
"""
Enhanced MEG LCS-CTC v4 with KAN integration and spectral decomposition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from torchmetrics import F1Score
from collections import defaultdict

# Import KANLayer - adjust path as needed
from kan.KANLayer import KANLayer

# ============================================
# KAN-based FFN Module
# ============================================


class KANFFN(nn.Module):
    """Feed-forward network using KAN layers."""
    
    def __init__(self, in_dim, hidden_dim, out_dim, num=5, k=3, 
                 grid_range=[-1, 1], dropout=0.1, activation='silu'):
        super().__init__()
        
        # Select base function for KAN
        if activation == 'silu':
            base_fun = nn.SiLU()
        elif activation == 'gelu':
            base_fun = nn.GELU()
        elif activation == 'relu':
            base_fun = nn.ReLU()
        else:
            base_fun = nn.SiLU()
        
        self.kan1 = KANLayer(
            in_dim=in_dim,
            out_dim=hidden_dim,
            num=num,
            k=k,
            grid_range=grid_range,
            base_fun=base_fun,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        self.dropout = nn.Dropout(dropout)
        
        self.kan2 = KANLayer(
            in_dim=hidden_dim,
            out_dim=out_dim,
            num=num,
            k=k,
            grid_range=grid_range,
            base_fun=base_fun,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    
    def forward(self, x):
        # Handle both 2D and 3D inputs
        original_shape = x.shape
        if len(original_shape) == 3:
            # Reshape (B, T, D) -> (B*T, D)
            B, T, D = original_shape
            x = x.reshape(B * T, D)
        
        x, _, _, _ = self.kan1(x)
        x = self.dropout(x)
        x, _, _, _ = self.kan2(x)
        
        if len(original_shape) == 3:
            # Reshape back (B*T, D_out) -> (B, T, D_out)
            x = x.reshape(B, T, -1)
        
        return x

# ============================================
# MEG-adapted Conformer Layer with KAN
# ============================================

class MEGKANConformerLayer(nn.Module):
    """Conformer layer with KAN-based FFN."""
    
    def __init__(self, dim: int, num_heads: int = 4, ff_dim: int = None, 
                 kernel_size: int = 3, dropout: float = 0.1, 
                 use_macaron_style: bool = False, kan_num: int = 5, 
                 kan_k: int = 3, kan_grid_range: List[float] = [-1, 1]):
        super().__init__()
        ff_dim = ff_dim or 2 * dim
        self.use_macaron_style = use_macaron_style
        
        # Depthwise separable convolution for efficiency
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim),
            nn.BatchNorm1d(dim),
            nn.Conv1d(dim, dim, 1),
            nn.ReLU()
        )
        
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ln3 = nn.LayerNorm(dim)
        
        if use_macaron_style:
            self.ln4 = nn.LayerNorm(dim)
            self.ffn1 = KANFFN(
                dim, ff_dim, dim, 
                num=kan_num, k=kan_k, 
                grid_range=kan_grid_range,
                dropout=dropout
            )
        
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        # Replace standard FFN with KAN FFN
        self.ffn = KANFFN(
            dim, ff_dim, dim,
            num=kan_num, k=kan_k,
            grid_range=kan_grid_range,
            dropout=dropout
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (B, T, D)
        
        # Macaron-style FFN (if enabled)
        if self.use_macaron_style:
            res = x
            x = self.ffn1(x)  # KANFFN now handles 3D input
            x = self.ln4(x + res)
        
        # Convolution module
        res = x
        x_conv = x.transpose(1, 2)  # (B, D, T)
        x_conv = self.conv(x_conv).transpose(1, 2)  # (B, T, D)
        x = self.ln1(x_conv + res)
        
        # Self-attention module
        res = x
        attn_out, _ = self.attention(x, x, x)
        x = self.ln2(self.dropout(attn_out) + res)
        
        # Feed-forward module with KAN
        res = x
        x = self.ffn(x)  # KANFFN now handles 3D input
        x = self.ln3(x + res)
        
        return x

# ============================================
# Focal Loss Implementation (unchanged)
# ============================================

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        p = torch.exp(-ce_loss)
        focal_loss = (1 - p) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != focal_loss.device:
                self.alpha = self.alpha.to(focal_loss.device)
            focal_loss = self.alpha[targets] * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ============================================
# Dynamic Channel Selection Module (unchanged)
# ============================================

class DynamicChannelSelector(nn.Module):
    """
    Dynamically selects informative MEG channels based on phoneme class frequency.
    """
    
    def __init__(self, vocab_size: int, meg_channels: int = 306, 
                 coordinate_file: Optional[str] = None, num_clusters: int = 20):
        super().__init__()
        self.vocab_size = vocab_size
        self.meg_channels = meg_channels
        self.num_clusters = num_clusters
        
        self.register_buffer('channel_coords', self._load_coordinates(coordinate_file))
        self.register_buffer('channel_clusters', self._compute_clusters())
        
        self.channel_importance = nn.Parameter(torch.ones(vocab_size, meg_channels))
        
        self.selection_net = nn.Sequential(
            nn.Linear(meg_channels + vocab_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, meg_channels),
            nn.Sigmoid()
        )
        
        self.register_buffer('class_frequencies', torch.ones(vocab_size) / vocab_size)
        self.register_buffer('update_count', torch.tensor(0.0))
        
    def _load_coordinates(self, coordinate_file: Optional[str]):
        if coordinate_file is None:
            coords = torch.randn(self.meg_channels, 3)
            coords = F.normalize(coords, p=2, dim=1) * 0.1
        else:
            import json
            with open(coordinate_file, 'r') as f:
                coord_data = json.load(f)
            
            if isinstance(coord_data, list):
                coords = torch.tensor(coord_data, dtype=torch.float32)
            elif isinstance(coord_data, dict):
                if 'coordinates' in coord_data:
                    coords = torch.tensor(coord_data['coordinates'], dtype=torch.float32)
                else:
                    coords_list = []
                    for sensor_name in sorted(coord_data.keys()):
                        coords_list.append(coord_data[sensor_name])
                    coords = torch.tensor(coords_list, dtype=torch.float32)
            else:
                raise ValueError(f"Unexpected JSON structure in {coordinate_file}")
                
        return coords
    
    def _compute_clusters(self):
        clusters = torch.zeros(self.meg_channels, self.num_clusters)
        for i in range(self.meg_channels):
            cluster_idx = i % self.num_clusters
            clusters[i, cluster_idx] = 1.0
        return clusters
    
    def update_class_frequencies(self, labels: torch.Tensor):
        with torch.no_grad():
            batch_size = labels.size(0)
            for label in labels:
                self.class_frequencies[label] += 1
            self.update_count += batch_size
            
            total = self.class_frequencies.sum()
            if total > 0:
                self.class_frequencies = self.class_frequencies / total
    
    def get_channel_mask(self, labels: Optional[torch.Tensor] = None, 
                        features: Optional[torch.Tensor] = None):
        B = features.size(0) if features is not None else labels.size(0)
        
        if labels is not None:
            label_freqs = self.class_frequencies[labels]
            is_rare = label_freqs < 0.02
            base_mask = self.channel_importance[labels]
            boost_factor = torch.where(is_rare.unsqueeze(1), 
                                      torch.ones_like(base_mask) * 1.5,
                                      torch.ones_like(base_mask))
            channel_mask = torch.sigmoid(base_mask * boost_factor)
        else:
            if features is not None:
                channel_var = features.var(dim=2)
                channel_mask = torch.sigmoid(channel_var * 10)
            else:
                channel_mask = torch.ones(B, self.meg_channels, device=self.channel_importance.device)
        
        channel_mask = torch.clamp(channel_mask, min=0.1, max=1.0)
        return channel_mask
    
    def forward(self, features: torch.Tensor, labels: Optional[torch.Tensor] = None):
        channel_mask = self.get_channel_mask(labels, features)
        masked_features = features * channel_mask.unsqueeze(2)
        return masked_features, channel_mask

# ============================================
# Enhanced Zipf Weight Learner with KAN attention
# ============================================

class AdaptiveZipfWeightLearnerKAN(nn.Module):
    """
    Enhanced Zipf learner with KAN-based attention mechanism.
    """
    
    def __init__(self, vocab_size: int, meg_dim: int, alpha: float = 0.99,
                 min_boost: float = 0.3, max_boost: float = 0.7,
                 kan_num: int = 5, kan_k: int = 3):
        super().__init__()
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.min_boost = min_boost
        self.max_boost = max_boost
        
        self.register_buffer('phoneme_counts', torch.ones(vocab_size))
        self.register_buffer('total_count', torch.tensor(vocab_size, dtype=torch.float32))
        self.register_buffer('meg_prototypes', torch.zeros(vocab_size, meg_dim))
        self.register_buffer('prototype_counts', torch.ones(vocab_size))
        
        self.zipf_s = nn.Parameter(torch.tensor(1.0))
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # KAN-based attention mechanism
        self.phoneme_meg_attention_kan = KANLayer(
            in_dim=meg_dim * 2 + 1,
            out_dim=1,
            num=kan_num,
            k=kan_k,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    
    def get_adaptive_boost_factor(self, labels: torch.Tensor) -> torch.Tensor:
        frequencies = self.phoneme_counts / self.total_count
        label_freqs = frequencies[labels]
        boost_factors = self.max_boost * (1 - label_freqs) + self.min_boost
        boost_factors = torch.clamp(boost_factors, self.min_boost, self.max_boost)
        return boost_factors
    
    def update_statistics(self, phonemes: torch.Tensor, meg_features: torch.Tensor):
        with torch.no_grad():
            for phoneme in phonemes:
                self.phoneme_counts[phoneme] = self.alpha * self.phoneme_counts[phoneme] + (1 - self.alpha)
                self.total_count = self.alpha * self.total_count + (1 - self.alpha)
            
            for phoneme, meg_feat in zip(phonemes, meg_features):
                old_prototype = self.meg_prototypes[phoneme]
                self.meg_prototypes[phoneme] = (
                    self.alpha * old_prototype + (1 - self.alpha) * meg_feat
                )
                self.prototype_counts[phoneme] += 1
    
    def forward(self, meg_features: torch.Tensor, labels: Optional[torch.Tensor] = None,
                training: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        B = meg_features.size(0)
        
        frequencies = self.phoneme_counts / self.total_count
        zipf_priors = self.get_zipf_weights().unsqueeze(0).expand(B, -1)
        
        norm_prototypes = F.normalize(self.meg_prototypes, p=2, dim=1)
        norm_meg = F.normalize(meg_features, p=2, dim=1)
        similarity = torch.matmul(norm_meg, norm_prototypes.T) / self.temperature
        
        freq_expanded = frequencies.unsqueeze(0).unsqueeze(-1).expand(B, -1, 1)
        combined_features = torch.cat([
            meg_features.unsqueeze(1).expand(-1, self.vocab_size, -1),
            self.meg_prototypes.unsqueeze(0).expand(B, -1, -1),
            freq_expanded
        ], dim=-1)
        
        # Use KAN for attention
        combined_flat = combined_features.reshape(B * self.vocab_size, -1)
        attention_weights, _, _, _ = self.phoneme_meg_attention_kan(combined_flat)
        attention_weights = torch.sigmoid(attention_weights).reshape(B, self.vocab_size)
        
        weights = zipf_priors * (1 + similarity) * attention_weights
        weights = F.softmax(weights, dim=-1)
        
        if labels is not None:
            boost_factors = self.get_adaptive_boost_factor(labels)
        else:
            boost_factors = torch.tensor(self.min_boost).expand(B)
        
        return weights, boost_factors
    
    def get_zipf_weights(self) -> torch.Tensor:
        frequencies = self.phoneme_counts / self.total_count
        sorted_freqs, sorted_indices = torch.sort(frequencies, descending=True)
        ranks = torch.zeros_like(frequencies)
        ranks[sorted_indices] = torch.arange(1, self.vocab_size + 1, 
                                            dtype=torch.float32, device=frequencies.device)
        
        zipf_weights = 1.0 / torch.pow(ranks, self.zipf_s)
        zipf_weights = zipf_weights / zipf_weights.sum()
        
        return zipf_weights

# ============================================
# MEG-adapted Cost Matrix Learner (unchanged)
# ============================================

class MEGCostMatrixLearner(nn.Module):
    """Learn frame-phoneme cost matrix for MEG-phoneme alignment."""
    
    def __init__(self, vocab_size=39, meg_channels=306, hidden_dim=128, projection_dim=128):
        super().__init__()
        self.meg_encoder = nn.Sequential(
            nn.Conv1d(meg_channels, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        self.temporal_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.linear = nn.Linear(hidden_dim, projection_dim)
        self.label_embedding = nn.Embedding(vocab_size, projection_dim)
    
    def forward(self, meg_data, text_labels):
        meg_features = self.meg_encoder(meg_data)
        meg_features = meg_features.transpose(1, 2)
        meg_features, _ = self.temporal_attention(meg_features, meg_features, meg_features)
        meg_features = self.linear(meg_features)
        text_embeddings = self.label_embedding(text_labels)
        cost_matrix = -torch.matmul(text_embeddings, meg_features.transpose(1, 2))
        cost_matrix = F.softmax(cost_matrix, dim=1)
        return cost_matrix

# ============================================
# KAN-based Classifier
# ============================================

class KANClassifier(nn.Module):
    """Multi-layer KAN classifier."""
    
    def __init__(self, input_dim, hidden_dims, output_dim, 
                 kan_num=5, kan_k=3, dropout_rates=None,
                 grid_range=[-1, 1], activation='silu'):
        super().__init__()
        
        if dropout_rates is None:
            dropout_rates = [0.1] * len(hidden_dims)
        
        # Select base function
        if activation == 'silu':
            base_fun = nn.SiLU()
        elif activation == 'gelu':
            base_fun = nn.GELU()
        else:
            base_fun = nn.ReLU()
        
        layers = []
        current_dim = input_dim
        
        for i, (hidden_dim, dropout) in enumerate(zip(hidden_dims, dropout_rates)):
            layers.append(KANLayer(
                in_dim=current_dim,
                out_dim=hidden_dim,
                num=kan_num,
                k=kan_k,
                grid_range=grid_range,
                base_fun=base_fun,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            ))
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        
        # Final layer
        layers.append(KANLayer(
            in_dim=current_dim,
            out_dim=output_dim,
            num=kan_num,
            k=kan_k,
            grid_range=grid_range,
            base_fun=base_fun,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        ))
        
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, KANLayer):
                x, _, _, _ = layer(x)
            else:
                x = layer(x)
        return x




























# ============================================
# Spectral Decomposition Module for MEG
# ============================================

class MEGSpectralDecomposition(nn.Module):
    """
    Spectral decomposition for MEG signals with multiple frequency band analysis.
    Processes each channel independently to preserve spatial information.
    """
    
    def __init__(self, 
                 n_fft: int = 64,
                 hop_length: int = 16,
                 n_mels: int = 32,
                 freq_bands: List[Tuple[float, float]] = None,
                 use_wavelet: bool = False,
                 wavelet_scales: int = 8,
                 sampling_rate: float = 1000.0,
                 use_learnable_filters: bool = True,
                 normalize_spectrum: bool = True):
        super().__init__()
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.use_wavelet = use_wavelet
        self.wavelet_scales = wavelet_scales
        self.sampling_rate = sampling_rate
        self.normalize_spectrum = normalize_spectrum
        
        # Define physiologically relevant frequency bands for MEG
        if freq_bands is None:
            self.freq_bands = [
                (0.5, 4.0),    # Delta
                (4.0, 8.0),    # Theta
                (8.0, 12.0),   # Alpha
                (12.0, 30.0),  # Beta
                (30.0, 50.0),  # Low Gamma
                (50.0, 100.0), # High Gamma
            ]
        else:
            self.freq_bands = freq_bands
        
        # Learnable spectral filters for adaptive frequency selection
        if use_learnable_filters:
            self.spectral_filters = nn.Parameter(
                torch.ones(len(self.freq_bands), n_fft // 2 + 1)
            )
            self.channel_specific_scaling = nn.Parameter(
                torch.ones(306, len(self.freq_bands))
            )
        else:
            self.register_buffer('spectral_filters', 
                                torch.ones(len(self.freq_bands), n_fft // 2 + 1))
            self.register_buffer('channel_specific_scaling',
                                torch.ones(306, len(self.freq_bands)))
        
        # Mel filterbank for mel-spectrogram
        mel_fb = self._create_mel_filterbank()
        self.register_buffer('mel_fb', mel_fb)
        
        # Initialize frequency band masks
        self._init_frequency_masks()
        
        # Adaptive normalization layers
        self.band_norm = nn.ModuleList([
            nn.LayerNorm(n_fft // 2 + 1) for _ in range(len(self.freq_bands))
        ])
        
        # Channel-wise feature extraction after spectral decomposition
        self.channel_encoder = nn.Conv1d(
            len(self.freq_bands) * (n_fft // 2 + 1),
            64,
            kernel_size=1,
            groups=1  # Process all frequency features together per channel
        )
        
    def _create_mel_filterbank(self):
        """Create mel-scale filterbank."""
        # Simplified mel filterbank creation
        n_freqs = self.n_fft // 2 + 1
        mel_fb = torch.zeros(self.n_mels, n_freqs)
        
        # Linear spacing in mel scale
        for i in range(self.n_mels):
            center = (i + 1) * n_freqs / (self.n_mels + 1)
            width = n_freqs / self.n_mels
            start = max(0, int(center - width))
            end = min(n_freqs, int(center + width))
            mel_fb[i, start:end] = 1.0 / (end - start)
        
        return mel_fb
    
    def _init_frequency_masks(self):
        """Initialize masks for frequency band extraction."""
        n_freqs = self.n_fft // 2 + 1
        freq_bins = torch.linspace(0, self.sampling_rate / 2, n_freqs)
        
        masks = []
        for low, high in self.freq_bands:
            mask = (freq_bins >= low) & (freq_bins <= high)
            masks.append(mask.float())
        
        self.register_buffer('freq_masks', torch.stack(masks))
    
    def compute_stft(self, x):
        """
        Compute Short-Time Fourier Transform per channel.
        x: (B, C, T) - batch, channels, time
        Returns: (B, C, F, T') - batch, channels, frequencies, time frames
        """
        B, C, T = x.shape
        
        # Reshape to process all channels at once
        x_flat = x.reshape(B * C, T)
        
        # Apply window
        window = torch.hann_window(self.n_fft, device=x.device)
        
        # Compute STFT
        stft = torch.stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
            center=True,
            pad_mode='reflect'
        )
        
        # Convert to magnitude and phase
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        
        # Reshape back
        F, T_new = magnitude.shape[-2:]
        magnitude = magnitude.reshape(B, C, F, T_new)
        phase = phase.reshape(B, C, F, T_new)
        
        return magnitude, phase
    
    def compute_wavelet(self, x):
        """
        Compute Continuous Wavelet Transform per channel.
        x: (B, C, T) - batch, channels, time
        Returns: (B, C, S, T) - batch, channels, scales, time
        """
        B, C, T = x.shape
        
        # Create Morlet wavelet filters
        scales = torch.logspace(0, 2, self.wavelet_scales, device=x.device)
        wavelet_features = []
        
        for scale in scales:
            # Create Morlet wavelet kernel
            kernel_size = min(int(6 * scale), T)
            t = torch.arange(-kernel_size//2, kernel_size//2 + 1, device=x.device).float()
            kernel = torch.exp(-t**2 / (2 * scale**2)) * torch.cos(2 * np.pi * t / scale)
            kernel = kernel / kernel.sum()
            
            # Convolve with signal
            x_padded = F.pad(x, (kernel_size//2, kernel_size//2), mode='reflect')
            conv = F.conv1d(
                x_padded.reshape(B * C, 1, -1),
                kernel.unsqueeze(0).unsqueeze(0),
                padding=0
            )
            wavelet_features.append(conv.reshape(B, C, -1))
        
        return torch.stack(wavelet_features, dim=2)
    
    def extract_spectral_features(self, magnitude, phase):
        """
        Extract frequency band-specific features.
        magnitude: (B, C, F, T)
        Returns: (B, C, n_bands, F, T)
        """
        B, C, F, T = magnitude.shape
        
        band_features = []
        for i, (mask, norm) in enumerate(zip(self.freq_masks, self.band_norm)):
            # Apply frequency mask and learnable filter
            masked_mag = magnitude * mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            
            if hasattr(self, 'spectral_filters'):
                filter_weight = torch.sigmoid(self.spectral_filters[i]).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
                masked_mag = masked_mag * filter_weight
            
            # Normalize within band
            if self.normalize_spectrum:
                masked_mag_flat = masked_mag.permute(0, 1, 3, 2).reshape(B * C * T, F)
                masked_mag_norm = norm(masked_mag_flat)
                masked_mag = masked_mag_norm.reshape(B, C, T, F).permute(0, 1, 3, 2)
            
            band_features.append(masked_mag)
        
        return torch.stack(band_features, dim=2)
    
    def compute_spectral_statistics(self, band_features):
        """
        Compute statistical features from spectral bands.
        band_features: (B, C, n_bands, F, T)
        Returns: dict of spectral statistics
        """
        stats = {}
        
        # Power spectral density
        psd = (band_features ** 2).mean(dim=-1)  # Average over time
        stats['psd'] = psd  # (B, C, n_bands, F)
        
        # Spectral centroid per band
        freqs = torch.arange(band_features.shape[3], device=band_features.device).float()
        freqs = freqs.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        spectral_centroid = (band_features * freqs).sum(dim=3) / (band_features.sum(dim=3) + 1e-8)
        stats['spectral_centroid'] = spectral_centroid  # (B, C, n_bands, T)
        
        # Spectral bandwidth
        centroid_expanded = spectral_centroid.unsqueeze(3)
        spectral_variance = ((freqs - centroid_expanded) ** 2 * band_features).sum(dim=3) / (band_features.sum(dim=3) + 1e-8)
        stats['spectral_bandwidth'] = torch.sqrt(spectral_variance + 1e-8)  # (B, C, n_bands, T)
        
        # Spectral rolloff
        cumsum = torch.cumsum(band_features, dim=3)
        total = band_features.sum(dim=3, keepdim=True)
        rolloff_threshold = 0.85 * total
        rolloff = (cumsum <= rolloff_threshold).sum(dim=3).float()
        stats['spectral_rolloff'] = rolloff  # (B, C, n_bands, T)
        
        return stats
    
    def forward(self, x):
        """
        Forward pass with spectral decomposition.
        x: (B, C, T) - batch, channels, time
        Returns: Spectral features and statistics
        """
        B, C, T = x.shape
        
        # Compute spectral representation
        if self.use_wavelet:
            wavelet_features = self.compute_wavelet(x)
            # Create pseudo-magnitude for compatibility
            magnitude = wavelet_features.abs()
            phase = torch.zeros_like(magnitude)
        else:
            magnitude, phase = self.compute_stft(x)
        
        # Extract frequency band features
        band_features = self.extract_spectral_features(magnitude, phase)
        
        # Apply channel-specific scaling
        if hasattr(self, 'channel_specific_scaling'):
            scaling = self.channel_specific_scaling.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            band_features = band_features * scaling
        
        # Compute spectral statistics
        stats = self.compute_spectral_statistics(band_features)
        
        # Combine features for output
        # Flatten frequency bands and frequencies
        B, C, n_bands, F, T_new = band_features.shape
        features_flat = band_features.reshape(B, C, n_bands * F, T_new)
        
        # Apply channel encoder to reduce dimensionality
        features_encoded = self.channel_encoder(features_flat.reshape(B * C, n_bands * F, T_new))
        features_encoded = features_encoded.reshape(B, C, -1, T_new)
        
        return {
            'spectral_features': features_encoded,  # (B, C, 64, T')
            'band_features': band_features,  # (B, C, n_bands, F, T')
            'magnitude': magnitude,  # (B, C, F, T')
            'phase': phase,  # (B, C, F, T')
            'statistics': stats  # Dict of various statistics
        }


# ============================================
# Enhanced MEG Encoder with Spectral Input
# ============================================

class SpectralMEGEncoder(nn.Module):
    """
    MEG encoder that processes spectral features instead of raw signals.
    """
    
    def __init__(self, 
                 spectral_dim: int = 64,
                 hidden_dim: int = 128,
                 num_channels: int = 306,
                 use_spatial_attention: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        
        # Process spectral features
        self.spectral_conv = nn.Sequential(
            nn.Conv2d(num_channels, hidden_dim, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )
        
        # Spatial attention across channels
        if use_spatial_attention:
            self.spatial_attention = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim // 8, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(hidden_dim // 8, hidden_dim, kernel_size=1),
                nn.Sigmoid()
            )
        else:
            self.spatial_attention = None
        
        # Temporal processing
        self.temporal_conv = nn.Conv1d(
            hidden_dim * spectral_dim, 
            hidden_dim, 
            kernel_size=5, 
            padding=2
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, spectral_features):
        """
        Process spectral features.
        spectral_features: (B, C, F, T) - batch, channels, freq features, time
        """
        B, C, F, T = spectral_features.shape
        
        # Treat channels as input channels for 2D conv
        x = self.spectral_conv(spectral_features)
        
        # Apply spatial attention
        if self.spatial_attention is not None:
            attention = self.spatial_attention(x)
            x = x * attention
        
        # Reshape for temporal processing
        x = x.reshape(B, -1, T)  # (B, hidden_dim * F, T)
        
        # Temporal convolution
        x = self.temporal_conv(x)
        x = self.dropout(x)
        
        return x  # (B, hidden_dim, T)


# ============================================
# Modified Enhanced MEG LCS-CTC Model with Spectral Decomposition
# ============================================

"""
Enhanced MEG LCS-CTC v4 with KAN integration and Spectral Decomposition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from torchmetrics import F1Score
from collections import defaultdict

# Import KANLayer - adjust path as needed
from kan.KANLayer import KANLayer

# ============================================
# Spectral Decomposition Module for MEG
# ============================================

class MEGSpectralDecomposition(nn.Module):
    """
    Spectral decomposition for MEG signals with multiple frequency band analysis.
    Processes each channel independently to preserve spatial information.
    """
    
    def __init__(self, 
                 n_fft: int = 64,
                 hop_length: int = 16,
                 n_mels: int = 32,
                 freq_bands: List[Tuple[float, float]] = None,
                 use_wavelet: bool = False,
                 wavelet_scales: int = 8,
                 sampling_rate: float = 1000.0,
                 use_learnable_filters: bool = True,
                 normalize_spectrum: bool = True):
        super().__init__()
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.use_wavelet = use_wavelet
        self.wavelet_scales = wavelet_scales
        self.sampling_rate = sampling_rate
        self.normalize_spectrum = normalize_spectrum
        
        # Define physiologically relevant frequency bands for MEG
        if freq_bands is None:
            self.freq_bands = [
                (0.5, 4.0),    # Delta
                (4.0, 8.0),    # Theta
                (8.0, 12.0),   # Alpha
                (12.0, 30.0),  # Beta
                (30.0, 50.0),  # Low Gamma
                (50.0, 100.0), # High Gamma
            ]
        else:
            self.freq_bands = freq_bands
        
        # Learnable spectral filters for adaptive frequency selection
        if use_learnable_filters:
            self.spectral_filters = nn.Parameter(
                torch.ones(len(self.freq_bands), n_fft // 2 + 1)
            )
            self.channel_specific_scaling = nn.Parameter(
                torch.ones(306, len(self.freq_bands))
            )
        else:
            self.register_buffer('spectral_filters', 
                                torch.ones(len(self.freq_bands), n_fft // 2 + 1))
            self.register_buffer('channel_specific_scaling',
                                torch.ones(306, len(self.freq_bands)))
        
        # Mel filterbank for mel-spectrogram
        mel_fb = self._create_mel_filterbank()
        self.register_buffer('mel_fb', mel_fb)
        
        # Initialize frequency band masks
        self._init_frequency_masks()
        
        # Adaptive normalization layers
        self.band_norm = nn.ModuleList([
            nn.LayerNorm(n_fft // 2 + 1) for _ in range(len(self.freq_bands))
        ])
        
        # Channel-wise feature extraction after spectral decomposition
        self.channel_encoder = nn.Conv1d(
            len(self.freq_bands) * (n_fft // 2 + 1),
            64,
            kernel_size=1,
            groups=1  # Process all frequency features together per channel
        )
        
    def _create_mel_filterbank(self):
        """Create mel-scale filterbank."""
        n_freqs = self.n_fft // 2 + 1
        mel_fb = torch.zeros(self.n_mels, n_freqs)
        
        # Linear spacing in mel scale
        for i in range(self.n_mels):
            center = (i + 1) * n_freqs / (self.n_mels + 1)
            width = n_freqs / self.n_mels
            start = max(0, int(center - width))
            end = min(n_freqs, int(center + width))
            mel_fb[i, start:end] = 1.0 / (end - start)
        
        return mel_fb
    
    def _init_frequency_masks(self):
        """Initialize masks for frequency band extraction."""
        n_freqs = self.n_fft // 2 + 1
        freq_bins = torch.linspace(0, self.sampling_rate / 2, n_freqs)
        
        masks = []
        for low, high in self.freq_bands:
            mask = (freq_bins >= low) & (freq_bins <= high)
            masks.append(mask.float())
        
        self.register_buffer('freq_masks', torch.stack(masks))
    
    def compute_stft(self, x):
        """
        Compute Short-Time Fourier Transform per channel.
        x: (B, C, T) - batch, channels, time
        Returns: (B, C, F, T') - batch, channels, frequencies, time frames
        """
        B, C, T = x.shape
        
        # Reshape to process all channels at once
        x_flat = x.reshape(B * C, T)
        
        # Apply window
        window = torch.hann_window(self.n_fft, device=x.device)
        
        # Compute STFT
        stft = torch.stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
            center=True,
            pad_mode='reflect'
        )
        
        # Convert to magnitude and phase
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        
        # Reshape back
        F, T_new = magnitude.shape[-2:]
        magnitude = magnitude.reshape(B, C, F, T_new)
        phase = phase.reshape(B, C, F, T_new)
        
        return magnitude, phase
    
    def compute_wavelet(self, x):
        """
        Compute Continuous Wavelet Transform per channel.
        x: (B, C, T) - batch, channels, time
        Returns: (B, C, S, T) - batch, channels, scales, time
        """
        B, C, T = x.shape
        
        # Create Morlet wavelet filters
        scales = torch.logspace(0, 2, self.wavelet_scales, device=x.device)
        wavelet_features = []
        
        for scale in scales:
            # Create Morlet wavelet kernel
            kernel_size = min(int(6 * scale), T)
            t = torch.arange(-kernel_size//2, kernel_size//2 + 1, device=x.device).float()
            kernel = torch.exp(-t**2 / (2 * scale**2)) * torch.cos(2 * np.pi * t / scale)
            kernel = kernel / kernel.sum()
            
            # Convolve with signal
            x_padded = F.pad(x, (kernel_size//2, kernel_size//2), mode='reflect')
            conv = F.conv1d(
                x_padded.reshape(B * C, 1, -1),
                kernel.unsqueeze(0).unsqueeze(0),
                padding=0
            )
            wavelet_features.append(conv.reshape(B, C, -1))
        
        return torch.stack(wavelet_features, dim=2)
    
    def extract_spectral_features(self, magnitude, phase):
        """
        Extract frequency band-specific features.
        magnitude: (B, C, F, T)
        Returns: (B, C, n_bands, F, T)
        """
        B, C, F, T = magnitude.shape
        
        band_features = []
        for i, (mask, norm) in enumerate(zip(self.freq_masks, self.band_norm)):
            # Apply frequency mask and learnable filter
            masked_mag = magnitude * mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            
            if hasattr(self, 'spectral_filters'):
                filter_weight = torch.sigmoid(self.spectral_filters[i]).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
                masked_mag = masked_mag * filter_weight
            
            # Normalize within band
            if self.normalize_spectrum:
                masked_mag_flat = masked_mag.permute(0, 1, 3, 2).reshape(B * C * T, F)
                masked_mag_norm = norm(masked_mag_flat)
                masked_mag = masked_mag_norm.reshape(B, C, T, F).permute(0, 1, 3, 2)
            
            band_features.append(masked_mag)
        
        return torch.stack(band_features, dim=2)
    
    def compute_spectral_statistics(self, band_features):
        """
        Compute statistical features from spectral bands.
        band_features: (B, C, n_bands, F, T)
        Returns: dict of spectral statistics
        """
        stats = {}
        
        # Power spectral density
        psd = (band_features ** 2).mean(dim=-1)  # Average over time
        stats['psd'] = psd  # (B, C, n_bands, F)
        
        # Spectral centroid per band
        freqs = torch.arange(band_features.shape[3], device=band_features.device).float()
        freqs = freqs.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        spectral_centroid = (band_features * freqs).sum(dim=3) / (band_features.sum(dim=3) + 1e-8)
        stats['spectral_centroid'] = spectral_centroid  # (B, C, n_bands, T)
        
        # Spectral bandwidth
        centroid_expanded = spectral_centroid.unsqueeze(3)
        spectral_variance = ((freqs - centroid_expanded) ** 2 * band_features).sum(dim=3) / (band_features.sum(dim=3) + 1e-8)
        stats['spectral_bandwidth'] = torch.sqrt(spectral_variance + 1e-8)  # (B, C, n_bands, T)
        
        # Spectral rolloff
        cumsum = torch.cumsum(band_features, dim=3)
        total = band_features.sum(dim=3, keepdim=True)
        rolloff_threshold = 0.85 * total
        rolloff = (cumsum <= rolloff_threshold).sum(dim=3).float()
        stats['spectral_rolloff'] = rolloff  # (B, C, n_bands, T)
        
        return stats
    
    def forward(self, x):
        """
        Forward pass with spectral decomposition.
        x: (B, C, T) - batch, channels, time
        Returns: Spectral features and statistics
        """
        B, C, T = x.shape
        
        # Compute spectral representation
        if self.use_wavelet:
            wavelet_features = self.compute_wavelet(x)
            # Create pseudo-magnitude for compatibility
            magnitude = wavelet_features.abs()
            phase = torch.zeros_like(magnitude)
        else:
            magnitude, phase = self.compute_stft(x)
        
        # Extract frequency band features
        band_features = self.extract_spectral_features(magnitude, phase)
        
        # Apply channel-specific scaling
        if hasattr(self, 'channel_specific_scaling'):
            scaling = self.channel_specific_scaling.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            band_features = band_features * scaling
        
        # Compute spectral statistics
        stats = self.compute_spectral_statistics(band_features)
        
        # Combine features for output
        # Flatten frequency bands and frequencies
        B, C, n_bands, F, T_new = band_features.shape
        features_flat = band_features.reshape(B, C, n_bands * F, T_new)
        
        # Apply channel encoder to reduce dimensionality
        features_encoded = self.channel_encoder(features_flat.reshape(B * C, n_bands * F, T_new))
        features_encoded = features_encoded.reshape(B, C, -1, T_new)
        
        return {
            'spectral_features': features_encoded,  # (B, C, 64, T')
            'band_features': band_features,  # (B, C, n_bands, F, T')
            'magnitude': magnitude,  # (B, C, F, T')
            'phase': phase,  # (B, C, F, T')
            'statistics': stats  # Dict of various statistics
        }


# ============================================
# Enhanced MEG Encoder with Spectral Input
# ============================================

class SpectralMEGEncoder(nn.Module):
    """
    MEG encoder that processes spectral features instead of raw signals.
    """
    
    def __init__(self, 
                 spectral_dim: int = 64,
                 hidden_dim: int = 128,
                 num_channels: int = 306,
                 use_spatial_attention: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        
        # Process spectral features
        self.spectral_conv = nn.Sequential(
            nn.Conv2d(num_channels, hidden_dim, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )
        
        # Spatial attention across channels
        if use_spatial_attention:
            self.spatial_attention = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim // 8, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(hidden_dim // 8, hidden_dim, kernel_size=1),
                nn.Sigmoid()
            )
        else:
            self.spatial_attention = None
        
        # Temporal processing
        self.temporal_conv = nn.Conv1d(
            hidden_dim * spectral_dim, 
            hidden_dim, 
            kernel_size=5, 
            padding=2
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, spectral_features):
        """
        Process spectral features.
        spectral_features: (B, C, F, T) - batch, channels, freq features, time
        """
        B, C, F, T = spectral_features.shape
        
        # Treat channels as input channels for 2D conv
        x = self.spectral_conv(spectral_features)
        
        # Apply spatial attention
        if self.spatial_attention is not None:
            attention = self.spatial_attention(x)
            x = x * attention
        
        # Reshape for temporal processing
        x = x.reshape(B, -1, T)  # (B, hidden_dim * F, T)
        
        # Temporal convolution
        x = self.temporal_conv(x)
        x = self.dropout(x)
        
        return x  # (B, hidden_dim, T)


# ============================================
# Complete Enhanced MEG LCS-CTC Model with Spectral Decomposition
# ============================================

class EnhancedMEGLCSCTCKANSpectral(L.LightningModule):
    """
    Enhanced MEG LCS-CTC v4 with KAN integration and spectral decomposition.
    """
    
    def __init__(self,
                 # Basic parameters
                 meg_channels=306,
                 time_points=125,
                 vocab_size=39,
                 hidden_dim=128,
                 num_conformers=4,
                 learning_rate=1e-4,
                 weight_decay=0.01,
                 dropout_rate=0.3,
                 # Spectral decomposition parameters
                 use_spectral: bool = True,
                 spectral_config: Dict = None,
                 # Original parameters
                 use_alignment=True,
                 zipf_weights=True,
                 zipf_alpha=0.99,
                 zipf_boost_factor=0.3,
                 label_smoothing=0.05,
                 # CTC configuration
                 ctc_weight=0.3,
                 ctc_frequency=10,
                 # Focal loss parameters
                 use_focal_loss=True,
                 focal_gamma=2.0,
                 focal_alpha=None,
                 focal_alpha_smoothing=0.1,
                 # Dynamic channels
                 use_dynamic_channels=True,
                 channel_selection_ratio=0.8,
                 channel_selection_strategy="variance",
                 # Adaptive Zipf
                 adaptive_zipf_boost=True,
                 min_zipf_boost=0.3,
                 max_zipf_boost=0.7,
                 coordinate_file=None,
                 # KAN parameters
                 kan_num=5,
                 kan_k=3,
                 kan_grid_range=[-1, 1],
                 # v4 configurations
                 conformer_config=None,
                 temporal_config=None,
                 classifier_config=None,
                 spatial_config=None,
                 hierarchical_classification=None,
                 ensemble=None,
                 scheduler_config=None,
                 optimizer_config=None,
                 stochastic_depth_rate=0.0,
                 **kwargs):
        
        super().__init__()
        self.save_hyperparameters()
        
        # Store key parameters
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.ctc_weight = ctc_weight
        self.ctc_frequency = ctc_frequency
        self.stochastic_depth_rate = stochastic_depth_rate
        self.kan_num = kan_num
        self.kan_k = kan_k
        self.kan_grid_range = kan_grid_range
        self.use_spectral = use_spectral
        self.spectral_config = spectral_config or {}
        
        # Parse configs with defaults
        conformer_config = conformer_config or {}
        temporal_config = temporal_config or {}
        classifier_config = classifier_config or {}
        
        # Initialize spectral decomposition if enabled
        if self.use_spectral:
            self.spectral_decomposer = MEGSpectralDecomposition(
                n_fft=self.spectral_config.get('n_fft', 64),
                hop_length=self.spectral_config.get('hop_length', 16),
                n_mels=self.spectral_config.get('n_mels', 32),
                freq_bands=self.spectral_config.get('freq_bands', None),
                use_wavelet=self.spectral_config.get('use_wavelet', False),
                wavelet_scales=self.spectral_config.get('wavelet_scales', 8),
                sampling_rate=self.spectral_config.get('sampling_rate', 1000.0),
                use_learnable_filters=self.spectral_config.get('use_learnable_filters', True),
                normalize_spectrum=self.spectral_config.get('normalize_spectrum', True)
            )
            
            # Use spectral encoder
            self.meg_encoder = SpectralMEGEncoder(
                spectral_dim=64,  # Output dim from spectral decomposer
                hidden_dim=hidden_dim,
                num_channels=meg_channels,
                use_spatial_attention=self.spectral_config.get('use_spatial_attention', True),
                dropout=dropout_rate
            )
            
            # Additional spectral statistics processor
            self.stats_processor = nn.Sequential(
                nn.Linear(4, 16),  # 4 statistics types
                nn.ReLU(),
                nn.Linear(16, hidden_dim),
                nn.Dropout(dropout_rate)
            )
            
            # Fusion layer for combining spectral features and statistics
            self.feature_fusion = nn.Linear(
                hidden_dim * 2,
                hidden_dim
            )
        else:
            # Use original encoder for non-spectral processing
            if temporal_config.get('multi_scale_kernels'):
                self.meg_encoder = nn.Sequential(
                    MultiScaleTemporalEncoder(
                        meg_channels, hidden_dim,
                        kernel_sizes=temporal_config.get('multi_scale_kernels', [3, 7, 15]),
                        use_dilated_conv=temporal_config.get('use_dilated_conv', True),
                        dilation_rates=temporal_config.get('dilation_rates', [1, 2, 4])
                    ),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU()
                )
            else:
                self.meg_encoder = nn.Sequential(
                    nn.Conv1d(meg_channels, hidden_dim, kernel_size=7, padding=3),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU()
                )
        
        # Dynamic channel selector
        self.use_dynamic_channels = use_dynamic_channels
        if use_dynamic_channels:
            self.channel_selector = DynamicChannelSelector(
                vocab_size, meg_channels, coordinate_file
            )
        
        # Temporal modeling with KAN Conformers
        self.conformers = nn.ModuleList([
            MEGKANConformerLayer(
                hidden_dim,
                num_heads=conformer_config.get('num_heads', 4),
                ff_dim=hidden_dim * conformer_config.get('ff_expansion_factor', 2),
                kernel_size=conformer_config.get('conv_kernel_size', 3),
                dropout=conformer_config.get('dropout', dropout_rate),
                use_macaron_style=conformer_config.get('use_macaron_style', False),
                kan_num=kan_num,
                kan_k=kan_k,
                kan_grid_range=kan_grid_range
            )
            for _ in range(num_conformers)
        ])
        
        # CTC components
        self.ctc_projection = nn.Linear(hidden_dim, vocab_size + 1)
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        
        # Estimate time dimension after processing
        self.estimated_time_dim = self._estimate_time_dimension(time_points)
        
        # KAN-based classification head
        if classifier_config.get('hidden_dims'):
            self.classifier = KANClassifier(
                input_dim=hidden_dim * self.estimated_time_dim,
                hidden_dims=classifier_config['hidden_dims'],
                output_dim=vocab_size,
                kan_num=kan_num,
                kan_k=kan_k,
                dropout_rates=classifier_config.get('dropout_rates', 
                                                   [dropout_rate] * len(classifier_config['hidden_dims'])),
                grid_range=kan_grid_range,
                activation=classifier_config.get('activation', 'silu')
            )
        else:
            self.classifier = KANClassifier(
                input_dim=hidden_dim * self.estimated_time_dim,
                hidden_dims=[32],
                output_dim=vocab_size,
                kan_num=kan_num,
                kan_k=kan_k,
                dropout_rates=[dropout_rate],
                grid_range=kan_grid_range,
                activation='silu'
            )
        
        # Cost matrix learner
        self.use_alignment = use_alignment
        if use_alignment:
            self.cost_learner = MEGCostMatrixLearner(
                vocab_size, meg_channels, hidden_dim
            )
        
        # Enhanced Zipf weight learner with KAN
        self.use_zipf = zipf_weights
        self.adaptive_zipf_boost = adaptive_zipf_boost
        self.zipf_boost_factor = zipf_boost_factor
        
        if zipf_weights:
            self.zipf_learner = AdaptiveZipfWeightLearnerKAN(
                vocab_size, hidden_dim, zipf_alpha,
                min_zipf_boost, max_zipf_boost,
                kan_num=kan_num, kan_k=kan_k
            )
            
            # KAN-based MEG aggregator
            self.meg_aggregator = KANLayer(
                in_dim=hidden_dim * self.estimated_time_dim,
                out_dim=hidden_dim,
                num=kan_num,
                k=kan_k,
                grid_range=kan_grid_range,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
        
        # Loss functions
        self.use_focal_loss = use_focal_loss
        self.focal_alpha_smoothing = focal_alpha_smoothing
        if use_focal_loss:
            self.criterion = FocalLoss(
                alpha=focal_alpha,
                gamma=focal_gamma,
                label_smoothing=label_smoothing
            )
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        self.f1_macro = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
        
        # Track class frequencies for focal loss alpha
        self.register_buffer('class_counts', torch.ones(vocab_size))
        self.register_buffer('total_samples', torch.tensor(0.0))
        
        # Store config for optimizer/scheduler
        self.scheduler_config = scheduler_config or {}
        self.optimizer_config = optimizer_config or {}
    
    def _estimate_time_dimension(self, original_time_points):
        """Estimate time dimension after spectral processing."""
        if self.use_spectral:
            # After STFT with given hop_length
            hop_length = self.spectral_config.get('hop_length', 16)
            # Approximate time dimension after STFT
            return (original_time_points - 1) // hop_length + 1
        else:
            return original_time_points
    
    def process_spectral_statistics(self, stats, time_dim):
        """
        Process spectral statistics into features.
        """
        # Combine different statistics
        stat_features = []
        
        for key in ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff']:
            if key in stats:
                # Average across frequency bands
                feat = stats[key].mean(dim=2)  # (B, C, T)
                stat_features.append(feat)
        
        if stat_features:
            # Stack and process
            combined_stats = torch.stack(stat_features, dim=-1)  # (B, C, T, n_stats)
            B, C, T, n_stats = combined_stats.shape
            
            # Process through stats network
            stats_flat = combined_stats.reshape(B * C * T, n_stats)
            stats_processed = self.stats_processor(stats_flat)
            stats_processed = stats_processed.reshape(B, C, T, -1)
            
            # Average across channels
            stats_features = stats_processed.mean(dim=1)  # (B, T, hidden_dim)
            
            # Interpolate to match time dimension if necessary
            if stats_features.shape[1] != time_dim:
                stats_features = F.interpolate(
                    stats_features.transpose(1, 2),
                    size=time_dim,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            
            return stats_features
        
        return None
    
    def update_class_weights(self):
        """Update focal loss alpha weights with smoothing."""
        if self.use_focal_loss and self.total_samples > 0:
            frequencies = self.class_counts / self.total_samples
            alpha = 1.0 / (frequencies + 1e-5)
            alpha = alpha / alpha.sum() * self.hparams.vocab_size
            
            if self.focal_alpha_smoothing > 0:
                uniform_weight = torch.ones_like(alpha) * self.hparams.vocab_size / len(alpha)
                alpha = (1 - self.focal_alpha_smoothing) * alpha + self.focal_alpha_smoothing * uniform_weight
                alpha = torch.clamp(alpha, 0.1, 10.0)
            
            if hasattr(self.criterion, 'alpha'):
                self.criterion.alpha = alpha
    
    def forward(self, x, labels=None, use_ctc=False):
        B, C, T = x.shape
        
        if self.use_spectral:
            # Apply spectral decomposition
            spectral_output = self.spectral_decomposer(x)
            spectral_features = spectral_output['spectral_features']
            stats = spectral_output['statistics']
            
            # Apply dynamic channel selection on spectral features if enabled
            if self.use_dynamic_channels and not use_ctc:
                # Reshape spectral features for channel selection
                B_s, C_s, F_s, T_s = spectral_features.shape
                spectral_flat = spectral_features.reshape(B_s, C_s, -1)
                spectral_flat, channel_mask = self.channel_selector(spectral_flat, labels)
                spectral_features = spectral_flat.reshape(B_s, C_s, F_s, T_s)
            
            # Encode spectral features
            features = self.meg_encoder(spectral_features)  # (B, hidden_dim, T')
            features = features.transpose(1, 2)  # (B, T', hidden_dim)
            
            # Process spectral statistics
            stats_features = self.process_spectral_statistics(stats, features.shape[1])
            
            # Fuse features if statistics are available
            if stats_features is not None:
                features_combined = torch.cat([features, stats_features], dim=-1)
                features = self.feature_fusion(features_combined)
        else:
            # Fall back to original processing
            if self.use_dynamic_channels and not use_ctc:
                x, channel_mask = self.channel_selector(x, labels)
            
            features = self.meg_encoder(x)
            features = features.transpose(1, 2)
        
        # Apply conformers with stochastic depth
        for i, conformer in enumerate(self.conformers):
            if self.training and self.stochastic_depth_rate > 0:
                if torch.rand(1).item() > self.stochastic_depth_rate:
                    features = conformer(features)
            else:
                features = conformer(features)
        
        # Handle CTC output
        if use_ctc:
            logits = self.ctc_projection(features)
            return logits
        else:
            # Update estimated time dimension based on actual features
            actual_time_dim = features.shape[1]

            # Dynamically adjust classifier input dimension if needed
            expected_input_dim = self.hparams.hidden_dim * actual_time_dim
            current_input_dim = self.classifier.layers[0].in_dim if hasattr(self.classifier.layers[0], 'in_dim') else None
            
            # Reshape features for classification
            features_flat = features.reshape(B, -1)
            
            # If dimension mismatch, interpolate features to expected dimension
            if features_flat.shape[1] != self.hparams.hidden_dim * self.estimated_time_dim:
                # Reshape and interpolate
                features_reshaped = features.transpose(1, 2)  # (B, hidden_dim, T)
                features_interpolated = F.interpolate(
                    features_reshaped,
                    size=self.estimated_time_dim,
                    mode='linear',
                    align_corners=False
                )
                features_flat = features_interpolated.transpose(1, 2).reshape(B, -1)
            
            # Apply classifier
            logits = self.classifier(features_flat)
            
            # Apply Zipf weighting if enabled
            if self.use_zipf and not self.training:
                meg_agg, _, _, _ = self.meg_aggregator(features_flat)
                
                zipf_adjustments, boost_factors = self.zipf_learner(
                    meg_agg, labels, training=False
                )
                probs = F.softmax(logits, dim=-1)
                for i in range(B):
                    boost = boost_factors[i] if labels is not None else 0.5
                    probs[i] = (1 - boost) * probs[i] + boost * zipf_adjustments[i]
                logits = torch.log(probs + 1e-10)
            
            return logits
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # Update class statistics
        with torch.no_grad():
            for label in y:
                self.class_counts[label] += 1
            self.total_samples += y.size(0)
            
            # Update focal loss weights periodically
            if batch_idx % 100 == 0:
                self.update_class_weights()
            
            # Update channel selector statistics
            if self.use_dynamic_channels:
                self.channel_selector.update_class_frequencies(y)
        
        # Forward pass with labels for dynamic channel selection
        y_hat = self(x, labels=y, use_ctc=False)
        loss = self.criterion(y_hat, y)
        
        # Update Zipf statistics if enabled
        if self.use_zipf:
            with torch.no_grad():
                B, C, T = x.shape
                
                # Process through spectral decomposition if enabled
                if self.use_spectral:
                    spectral_output = self.spectral_decomposer(x)
                    spectral_features = spectral_output['spectral_features']
                    features = self.meg_encoder(spectral_features)
                    features = features.transpose(1, 2)
                else:
                    features = self.meg_encoder(x)
                    features = features.transpose(1, 2)
                
                # Apply conformers
                for conformer in self.conformers:
                    features = conformer(features)
                
                # Prepare features for aggregator
                actual_time_dim = features.shape[1]
                if features.shape[1] != self.estimated_time_dim:
                    features_reshaped = features.transpose(1, 2)
                    features_interpolated = F.interpolate(
                        features_reshaped,
                        size=self.estimated_time_dim,
                        mode='linear',
                        align_corners=False
                    )
                    features = features_interpolated.transpose(1, 2)
                
                features_flat = features.reshape(B, -1)
                meg_agg, _, _, _ = self.meg_aggregator(features_flat)
                
                self.zipf_learner.update_statistics(y, meg_agg)
        
        # Optional CTC loss with configurable frequency
        if self.use_alignment and batch_idx % self.ctc_frequency == 0:
            ctc_logits = self(x, use_ctc=True)
            log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)
            input_lengths = torch.full((x.size(0),), ctc_logits.size(1), dtype=torch.long)
            target_lengths = torch.ones(x.size(0), dtype=torch.long)
            ctc_targets = y.unsqueeze(1)
            
            ctc_loss = self.ctc_loss(log_probs, ctc_targets, input_lengths, target_lengths)
            loss = (1 - self.ctc_weight) * loss + self.ctc_weight * ctc_loss
            
            self.log('train_ctc_loss', ctc_loss, prog_bar=False)
        
        # Calculate F1 score
        f1_macro = self.f1_macro(y_hat, y)
        
        # Logging
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_f1_macro', f1_macro)
        
        # Log additional metrics
        if batch_idx % 100 == 0:
            if self.use_zipf:
                zipf_weights = self.zipf_learner.get_zipf_weights()
                self.log('zipf_entropy', -torch.sum(zipf_weights * torch.log(zipf_weights + 1e-10)))
                self.log('zipf_s', self.zipf_learner.zipf_s)
                self.log('zipf_temperature', self.zipf_learner.temperature)
            
            if self.use_focal_loss and hasattr(self.criterion, 'alpha') and self.criterion.alpha is not None:
                self.log('focal_alpha_std', self.criterion.alpha.std())
            
            # Log spectral-specific metrics
            if self.use_spectral and hasattr(self.spectral_decomposer, 'spectral_filters'):
                filters = self.spectral_decomposer.spectral_filters
                self.log('spectral_filter_mean', filters.mean())
                self.log('spectral_filter_std', filters.std())
                
                if hasattr(self.spectral_decomposer, 'channel_specific_scaling'):
                    scaling = self.spectral_decomposer.channel_specific_scaling
                    self.log('channel_scaling_mean', scaling.mean())
                    self.log('channel_scaling_std', scaling.std())
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x, labels=y, use_ctc=False)
        loss = self.criterion(y_hat, y)
        f1_macro = self.f1_macro(y_hat, y)
        
        # Calculate accuracy
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('val_loss', loss)
        self.log('val_f1_macro', f1_macro, prog_bar=True)
        self.log('val_accuracy', acc)
        
        # Compare with and without enhancements
        if self.use_zipf and batch_idx == 0:
            # Test without Zipf
            self.use_zipf = False
            y_hat_no_zipf = self(x, labels=y, use_ctc=False)
            self.use_zipf = True
            
            f1_no_zipf = self.f1_macro(y_hat_no_zipf, y)
            self.log('val_f1_no_zipf', f1_no_zipf)
            self.log('val_f1_zipf_gain', f1_macro - f1_no_zipf)
        
        # Test spectral vs non-spectral if spectral is enabled
        if self.use_spectral and batch_idx == 0:
            # Temporarily disable spectral
            self.use_spectral = False
            with torch.no_grad():
                try:
                    y_hat_no_spectral = self(x, labels=y, use_ctc=False)
                    f1_no_spectral = self.f1_macro(y_hat_no_spectral, y)
                    self.log('val_f1_no_spectral', f1_no_spectral)
                    self.log('val_f1_spectral_gain', f1_macro - f1_no_spectral)
                except:
                    pass  # Skip if incompatible
            self.use_spectral = True
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step with detailed metrics."""
        x, y = batch
        y_hat = self(x, labels=y, use_ctc=False)
        loss = self.criterion(y_hat, y)
        f1_macro = self.f1_macro(y_hat, y)
        
        # Calculate accuracy
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        
        # Per-class accuracy
        per_class_correct = torch.zeros(self.hparams.vocab_size, device=self.device)
        per_class_total = torch.zeros(self.hparams.vocab_size, device=self.device)
        
        for pred, target in zip(preds, y):
            per_class_total[target] += 1
            if pred == target:
                per_class_correct[target] += 1
        
        per_class_acc = per_class_correct / (per_class_total + 1e-8)
        
        self.log('test_loss', loss)
        self.log('test_f1_macro', f1_macro)
        self.log('test_accuracy', acc)
        self.log('test_per_class_acc_mean', per_class_acc.mean())
        self.log('test_per_class_acc_std', per_class_acc.std())
        
        return {'loss': loss, 'f1': f1_macro, 'accuracy': acc, 
                'per_class_acc': per_class_acc}
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler with support for spectral parameters."""
        # Separate parameters by type for different learning rates
        kan_param_ids = set()
        kan_params = []
        spectral_param_ids = set()
        spectral_params = []
        
        # Collect KAN parameters
        for module in self.modules():
            if isinstance(module, KANLayer):
                for param in module.parameters():
                    kan_param_ids.add(id(param))
                    kan_params.append(param)
        
        # Collect spectral parameters if spectral decomposition is used
        if self.use_spectral:
            for name, param in self.named_parameters():
                if 'spectral' in name and id(param) not in kan_param_ids:
                    spectral_param_ids.add(id(param))
                    spectral_params.append(param)
        
        # Collect other parameters
        other_params = []
        for name, param in self.named_parameters():
            param_id = id(param)
            if param_id not in kan_param_ids and param_id not in spectral_param_ids:
                other_params.append(param)
        
        # Create parameter groups with different learning rates
        params = [
            {'params': other_params, 'lr': self.hparams.learning_rate},
            {'params': kan_params, 'lr': self.hparams.learning_rate * 0.5},  # KAN layers often need lower LR
        ]
        
        # Add spectral parameters group if applicable
        if spectral_params:
            params.append({
                'params': spectral_params, 
                'lr': self.hparams.learning_rate * 0.8  # Slightly lower LR for spectral params
            })
        
        # Configure optimizer based on config
        optimizer_type = self.optimizer_config.get('type', 'adamw')
        
        if optimizer_type.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                params,
                weight_decay=self.weight_decay,
                betas=self.optimizer_config.get('betas', [0.9, 0.999]),
                eps=self.optimizer_config.get('eps', 1e-8),
                amsgrad=self.optimizer_config.get('amsgrad', False)
            )
        elif optimizer_type.lower() == 'adam':
            optimizer = torch.optim.Adam(
                params,
                betas=self.optimizer_config.get('betas', [0.9, 0.999]),
                eps=self.optimizer_config.get('eps', 1e-8),
                amsgrad=self.optimizer_config.get('amsgrad', False)
            )
        elif optimizer_type.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                params,
                momentum=self.optimizer_config.get('momentum', 0.9),
                weight_decay=self.weight_decay
            )
        else:
            optimizer = torch.optim.AdamW(params, weight_decay=self.weight_decay)
        
        # Configure scheduler based on config
        scheduler_type = self.scheduler_config.get('type', 'cosine_annealing')
        
        if scheduler_type == 'cosine_annealing':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.scheduler_config.get('t_max', 50),
                eta_min=self.scheduler_config.get('eta_min', 1e-6)
            )
        elif scheduler_type == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=self.scheduler_config.get('factor', 0.5),
                patience=self.scheduler_config.get('patience', 5),
                min_lr=self.scheduler_config.get('eta_min', 1e-6)
            )
        elif scheduler_type == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=self.scheduler_config.get('gamma', 0.95)
            )
        elif scheduler_type == 'one_cycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.learning_rate * 10,
                total_steps=self.scheduler_config.get('total_steps', 1000),
                pct_start=self.scheduler_config.get('pct_start', 0.3)
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=50, eta_min=1e-6
            )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_f1_macro' if scheduler_type == 'reduce_on_plateau' else None
        }
    
    def on_train_epoch_end(self):
        """Update parameters at the end of each epoch."""
        # Log epoch-level metrics
        if self.use_spectral:
            self.log('epoch', self.current_epoch)
            
            # Optionally update spectral filter ranges based on data statistics
            if hasattr(self.spectral_decomposer, 'spectral_filters'):
                with torch.no_grad():
                    # Apply soft clipping to prevent extreme values
                    self.spectral_decomposer.spectral_filters.data = torch.clamp(
                        self.spectral_decomposer.spectral_filters.data,
                        -5.0, 5.0
                    )
    
    def on_validation_epoch_end(self):
        """Validation epoch end hook."""
        # Reset F1 metric for next epoch
        self.f1_macro.reset()
    
    def predict_step(self, batch, batch_idx):
        """Prediction step for inference."""
        x, _ = batch if isinstance(batch, tuple) else (batch, None)
        logits = self(x, use_ctc=False)
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=-1)
        
        return {
            'predictions': preds,
            'probabilities': probs,
            'logits': logits
        }
