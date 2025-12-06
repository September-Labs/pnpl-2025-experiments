"""
Enhanced MEG LCS-CTC v4 with comprehensive configurability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from torchmetrics import F1Score
from collections import defaultdict

# ============================================
# MEG-adapted Conformer Layer (Enhanced for v4)
# ============================================

class MEGConformerLayer(nn.Module):
    """Conformer layer adapted for MEG data with v4 enhancements."""
    
    def __init__(self, dim: int, num_heads: int = 4, ff_dim: int = None, 
                 kernel_size: int = 3, dropout: float = 0.1, 
                 use_macaron_style: bool = False):
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
            self.ffn1 = nn.Sequential(
                nn.Linear(dim, ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, dim),
                nn.Dropout(dropout)
            )
        
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (B, T, D)
        
        # Macaron-style FFN (if enabled)
        if self.use_macaron_style:
            res = x
            x = self.ffn1(x)
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
        
        # Feed-forward module
        res = x
        x = self.ffn(x)
        x = self.ln3(x + res)
        
        return x

# ============================================
# Multi-Scale Temporal Encoder (NEW for v4)
# ============================================

class MultiScaleTemporalEncoder(nn.Module):
    """Multi-scale temporal encoding with configurable kernels."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_sizes: List[int] = [3, 7, 15],
                 use_dilated_conv: bool = True,
                 dilation_rates: List[int] = [1, 2, 4]):
        super().__init__()
        
        self.multi_scale_convs = nn.ModuleList()
        for kernel_size in kernel_sizes:
            conv_layers = []
            if use_dilated_conv:
                for dilation in dilation_rates:
                    conv_layers.append(
                        nn.Conv1d(in_channels if len(conv_layers) == 0 else out_channels,
                                out_channels, kernel_size, 
                                padding=(kernel_size-1)//2 * dilation,
                                dilation=dilation)
                    )
                    conv_layers.append(nn.BatchNorm1d(out_channels))
                    conv_layers.append(nn.ReLU())
            else:
                conv_layers.append(
                    nn.Conv1d(in_channels, out_channels, kernel_size, 
                            padding=(kernel_size-1)//2)
                )
                conv_layers.append(nn.BatchNorm1d(out_channels))
                conv_layers.append(nn.ReLU())
            
            self.multi_scale_convs.append(nn.Sequential(*conv_layers))
        
        # Fusion layer
        self.fusion = nn.Conv1d(out_channels * len(kernel_sizes), out_channels, 1)
        self.fusion_norm = nn.BatchNorm1d(out_channels)
    
    def forward(self, x):
        # x: (B, C, T)
        multi_scale_features = []
        for conv in self.multi_scale_convs:
            multi_scale_features.append(conv(x))
        
        # Concatenate and fuse
        fused = torch.cat(multi_scale_features, dim=1)
        output = self.fusion_norm(self.fusion(fused))
        
        return output

# ============================================
# NEW: Focal Loss Implementation
# ============================================

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Focuses learning on hard examples (rare phonemes).
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha  # Class weights (optional)
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, C) raw logits
            targets: (B,) target labels
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        
        # Get probabilities
        p = torch.exp(-ce_loss)
        
        # Apply focal term
        focal_loss = (1 - p) ** self.gamma * ce_loss
        
        # Apply class weights if provided
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
# NEW: Dynamic Channel Selection Module
# ============================================

class DynamicChannelSelector(nn.Module):
    """
    Dynamically selects informative MEG channels based on phoneme class frequency.
    Uses spatial coordinates to identify channel clusters.
    """
    
    def __init__(self, vocab_size: int, meg_channels: int = 306, 
                 coordinate_file: Optional[str] = None, num_clusters: int = 20):
        super().__init__()
        self.vocab_size = vocab_size
        self.meg_channels = meg_channels
        self.num_clusters = num_clusters
        
        # Load and process coordinates
        self.register_buffer('channel_coords', self._load_coordinates(coordinate_file))
        
        # Compute channel clusters based on spatial proximity
        self.register_buffer('channel_clusters', self._compute_clusters())
        
        # Learn channel importance per phoneme class
        self.channel_importance = nn.Parameter(torch.ones(vocab_size, meg_channels))
        
        # Adaptive selection network
        self.selection_net = nn.Sequential(
            nn.Linear(meg_channels + vocab_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, meg_channels),
            nn.Sigmoid()
        )
        
        # Track class frequencies
        self.register_buffer('class_frequencies', torch.ones(vocab_size) / vocab_size)
        self.register_buffer('update_count', torch.tensor(0.0))
        
    def _load_coordinates(self, coordinate_file: Optional[str]):
        """Load MEG sensor coordinates."""
        if coordinate_file is None:
            # Use default coordinates from your paste-2.txt
            # Simplified: create synthetic coordinates in hemisphere
            coords = torch.randn(self.meg_channels, 3)
            coords = F.normalize(coords, p=2, dim=1) * 0.1  # Normalize to head radius
        else:
            # Load from JSON file
            import json
            with open(coordinate_file, 'r') as f:
                coord_data = json.load(f)
            
            # Extract coordinates - adjust based on actual JSON structure
            # Assuming the JSON has a list of [x, y, z] coordinates
            if isinstance(coord_data, list):
                coords = torch.tensor(coord_data, dtype=torch.float32)
            elif isinstance(coord_data, dict):
                # If it's a dict, might have keys like 'coordinates' or sensor names
                # Adjust this based on your actual JSON structure
                if 'coordinates' in coord_data:
                    coords = torch.tensor(coord_data['coordinates'], dtype=torch.float32)
                else:
                    # If sensors are keys with xyz values
                    coords_list = []
                    for sensor_name in sorted(coord_data.keys()):
                        coords_list.append(coord_data[sensor_name])
                    coords = torch.tensor(coords_list, dtype=torch.float32)
            else:
                raise ValueError(f"Unexpected JSON structure in {coordinate_file}")
                
        return coords
    def _compute_clusters(self):
        """Compute spatial clusters of channels."""
        # Simple k-means style clustering based on coordinates
        # For now, just divide into regions
        clusters = torch.zeros(self.meg_channels, self.num_clusters)
        
        # Assign channels to clusters based on spatial location
        for i in range(self.meg_channels):
            # Simple spatial binning (you can use actual k-means here)
            cluster_idx = i % self.num_clusters
            clusters[i, cluster_idx] = 1.0
            
        return clusters
    
    def update_class_frequencies(self, labels: torch.Tensor):
        """Update class frequency statistics."""
        with torch.no_grad():
            batch_size = labels.size(0)
            for label in labels:
                self.class_frequencies[label] += 1
            self.update_count += batch_size
            
            # Normalize frequencies
            total = self.class_frequencies.sum()
            if total > 0:
                self.class_frequencies = self.class_frequencies / total
    
    def get_channel_mask(self, labels: Optional[torch.Tensor] = None, 
                        features: Optional[torch.Tensor] = None):
        """
        Get channel selection mask based on class frequency.
        
        Args:
            labels: (B,) phoneme labels (for supervised selection)
            features: (B, C, T) MEG features (for feature-based selection)
            
        Returns:
            (B, C) channel selection mask
        """
        B = features.size(0) if features is not None else labels.size(0)
        
        if labels is not None:
            # Get frequency-based importance
            label_freqs = self.class_frequencies[labels]  # (B,)
            
            # Rare classes (freq < 2%) get more channels
            is_rare = label_freqs < 0.02  # (B,)
            
            # Base channel selection from learned importance
            base_mask = self.channel_importance[labels]  # (B, C)
            
            # Boost channels for rare classes
            boost_factor = torch.where(is_rare.unsqueeze(1), 
                                      torch.ones_like(base_mask) * 1.5,
                                      torch.ones_like(base_mask))
            
            channel_mask = torch.sigmoid(base_mask * boost_factor)
            
        else:
            # Unsupervised selection based on feature variance
            if features is not None:
                # Compute channel variance across time
                channel_var = features.var(dim=2)  # (B, C)
                # Select high-variance channels
                channel_mask = torch.sigmoid(channel_var * 10)
            else:
                # Default: use all channels
                channel_mask = torch.ones(B, self.meg_channels, device=self.channel_importance.device)
        
        # Apply minimum channel threshold
        channel_mask = torch.clamp(channel_mask, min=0.1, max=1.0)
        
        return channel_mask
    
    def forward(self, features: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """
        Apply dynamic channel selection to features.
        
        Args:
            features: (B, C, T) MEG features
            labels: (B,) phoneme labels (optional)
            
        Returns:
            (B, C, T) channel-selected features
        """
        channel_mask = self.get_channel_mask(labels, features)  # (B, C)
        
        # Apply mask to features
        masked_features = features * channel_mask.unsqueeze(2)
        
        return masked_features, channel_mask

# ============================================
# Original Zipf Weight Learner (for compatibility)
# ============================================

class ZipfWeightLearner(nn.Module):
    """
    Learns Zipf distribution of phonemes and their MEG signatures.
    Uses exponential moving average for online learning.
    """
    
    def __init__(self, vocab_size: int, meg_dim: int, alpha: float = 0.99):
        super().__init__()
        self.vocab_size = vocab_size
        self.alpha = alpha  # EMA decay factor
        
        # Track phoneme frequencies (initialized uniformly)
        self.register_buffer('phoneme_counts', torch.ones(vocab_size))
        self.register_buffer('total_count', torch.tensor(vocab_size, dtype=torch.float32))
        
        # Learn prototypical MEG patterns per phoneme
        self.register_buffer('meg_prototypes', torch.zeros(vocab_size, meg_dim))
        self.register_buffer('prototype_counts', torch.ones(vocab_size))
        
        # Learnable Zipf parameters
        self.zipf_s = nn.Parameter(torch.tensor(1.0))  # Zipf exponent
        self.temperature = nn.Parameter(torch.tensor(1.0))  # Temperature for softmax
        
        # Phoneme-MEG attention module (outputs single weight per phoneme)
        self.phoneme_meg_attention = nn.Sequential(
            nn.Linear(meg_dim * 2, meg_dim),
            nn.ReLU(),
            nn.Linear(meg_dim, 1),  # Single attention weight per phoneme
            nn.Sigmoid()
        )
    
    def update_statistics(self, phonemes: torch.Tensor, meg_features: torch.Tensor):
        """
        Update phoneme frequency and MEG prototype statistics.
        
        Args:
            phonemes: (B,) phoneme labels
            meg_features: (B, meg_dim) aggregated MEG features
        """
        with torch.no_grad():
            # Update phoneme counts with EMA
            for phoneme in phonemes:
                self.phoneme_counts[phoneme] = self.alpha * self.phoneme_counts[phoneme] + (1 - self.alpha)
                self.total_count = self.alpha * self.total_count + (1 - self.alpha)
            
            # Update MEG prototypes with EMA
            for phoneme, meg_feat in zip(phonemes, meg_features):
                old_prototype = self.meg_prototypes[phoneme]
                self.meg_prototypes[phoneme] = (
                    self.alpha * old_prototype + (1 - self.alpha) * meg_feat
                )
                self.prototype_counts[phoneme] += 1
    
    def get_zipf_weights(self) -> torch.Tensor:
        """
        Compute Zipf-based prior probabilities for phonemes.
        
        Returns:
            (vocab_size,) tensor of prior probabilities
        """
        # Normalize counts to get frequencies
        frequencies = self.phoneme_counts / self.total_count
        
        # Sort by frequency to get ranks
        sorted_freqs, sorted_indices = torch.sort(frequencies, descending=True)
        ranks = torch.zeros_like(frequencies)
        ranks[sorted_indices] = torch.arange(1, self.vocab_size + 1, dtype=torch.float32, device=frequencies.device)
        
        # Apply Zipf's law: P(rank) ∝ 1 / rank^s
        zipf_weights = 1.0 / torch.pow(ranks, self.zipf_s)
        zipf_weights = zipf_weights / zipf_weights.sum()
        
        return zipf_weights
    
    def compute_meg_similarity(self, meg_features: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity between input MEG and learned prototypes.
        
        Args:
            meg_features: (B, meg_dim) MEG features
            
        Returns:
            (B, vocab_size) similarity scores
        """
        # Normalize prototypes
        norm_prototypes = F.normalize(self.meg_prototypes, p=2, dim=1)
        norm_meg = F.normalize(meg_features, p=2, dim=1)
        
        # Compute cosine similarity
        similarity = torch.matmul(norm_meg, norm_prototypes.T)  # (B, vocab_size)
        
        # Apply temperature scaling
        similarity = similarity / self.temperature
        
        return similarity
    
    def forward(self, meg_features: torch.Tensor, training: bool = False) -> torch.Tensor:
        """
        Compute Zipf-weighted adjustments for predictions.
        
        Args:
            meg_features: (B, meg_dim) aggregated MEG features
            training: Whether in training mode
            
        Returns:
            (B, vocab_size) weight adjustments
        """
        B = meg_features.size(0)
        
        # Get Zipf prior probabilities
        zipf_priors = self.get_zipf_weights().unsqueeze(0).expand(B, -1)  # (B, vocab_size)
        
        # Get MEG-based similarity scores
        meg_similarity = self.compute_meg_similarity(meg_features)  # (B, vocab_size)
        
        # Combine Zipf priors with MEG similarity
        # Use attention mechanism to balance the two
        combined_features = torch.cat([
            meg_features.unsqueeze(1).expand(-1, self.vocab_size, -1),
            self.meg_prototypes.unsqueeze(0).expand(B, -1, -1)
        ], dim=-1)  # (B, vocab_size, meg_dim * 2)
        
        attention_weights = self.phoneme_meg_attention(combined_features.reshape(B * self.vocab_size, -1))  # (B*vocab_size, 1)
        attention_weights = attention_weights.squeeze(-1).reshape(B, self.vocab_size)  # (B, vocab_size)
        
        # Weighted combination
        weights = zipf_priors * (1 + meg_similarity) * attention_weights
        weights = F.softmax(weights, dim=-1)
        
        return weights

# ============================================
# NEW: Enhanced Zipf Weight Learner
# ============================================

class AdaptiveZipfWeightLearner(nn.Module):
    """
    Enhanced Zipf learner with adaptive boost factor based on class frequency.
    """
    
    def __init__(self, vocab_size: int, meg_dim: int, alpha: float = 0.99,
                 min_boost: float = 0.3, max_boost: float = 0.7):
        super().__init__()
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.min_boost = min_boost
        self.max_boost = max_boost
        
        # Track phoneme frequencies
        self.register_buffer('phoneme_counts', torch.ones(vocab_size))
        self.register_buffer('total_count', torch.tensor(vocab_size, dtype=torch.float32))
        
        # MEG prototypes
        self.register_buffer('meg_prototypes', torch.zeros(vocab_size, meg_dim))
        self.register_buffer('prototype_counts', torch.ones(vocab_size))
        
        # Learnable parameters
        self.zipf_s = nn.Parameter(torch.tensor(1.0))
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # Enhanced attention with frequency awareness
        self.phoneme_meg_attention = nn.Sequential(
            nn.Linear(meg_dim * 2 + 1, meg_dim),  # +1 for frequency
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(meg_dim, 1),
            nn.Sigmoid()
        )
    
    def get_adaptive_boost_factor(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive boost factor based on class frequency.
        
        Args:
            labels: (B,) phoneme labels
            
        Returns:
            (B,) boost factors
        """
        frequencies = self.phoneme_counts / self.total_count
        label_freqs = frequencies[labels]
        
        # Inverse frequency scaling: rarer classes get higher boost
        # boost = max_boost * (1 - freq) + min_boost
        boost_factors = self.max_boost * (1 - label_freqs) + self.min_boost
        boost_factors = torch.clamp(boost_factors, self.min_boost, self.max_boost)
        
        return boost_factors
    
    def update_statistics(self, phonemes: torch.Tensor, meg_features: torch.Tensor):
        """Update phoneme frequency and MEG prototype statistics."""
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
        """
        Compute adaptive Zipf-weighted adjustments.
        
        Returns:
            (weights, boost_factors)
        """
        B = meg_features.size(0)
        
        # Get frequencies
        frequencies = self.phoneme_counts / self.total_count
        zipf_priors = self.get_zipf_weights().unsqueeze(0).expand(B, -1)
        
        # Compute MEG similarity
        norm_prototypes = F.normalize(self.meg_prototypes, p=2, dim=1)
        norm_meg = F.normalize(meg_features, p=2, dim=1)
        similarity = torch.matmul(norm_meg, norm_prototypes.T) / self.temperature
        
        # Enhanced attention with frequency information
        freq_expanded = frequencies.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1)  # (B, vocab_size, 1)
        combined_features = torch.cat([
            meg_features.unsqueeze(1).expand(-1, self.vocab_size, -1),
            self.meg_prototypes.unsqueeze(0).expand(B, -1, -1),
            freq_expanded
        ], dim=-1)  # (B, vocab_size, meg_dim * 2 + 1)
        
        attention_weights = self.phoneme_meg_attention(
            combined_features.reshape(B * self.vocab_size, -1)
        ).squeeze(-1).reshape(B, self.vocab_size)
        
        # Compute weights
        weights = zipf_priors * (1 + similarity) * attention_weights
        weights = F.softmax(weights, dim=-1)
        
        # Get adaptive boost factors if labels provided
        if labels is not None:
            boost_factors = self.get_adaptive_boost_factor(labels)
        else:
            boost_factors = torch.tensor(self.min_boost).expand(B)
        
        return weights, boost_factors
    
    def get_zipf_weights(self) -> torch.Tensor:
        """Compute Zipf-based prior probabilities."""
        frequencies = self.phoneme_counts / self.total_count
        sorted_freqs, sorted_indices = torch.sort(frequencies, descending=True)
        ranks = torch.zeros_like(frequencies)
        ranks[sorted_indices] = torch.arange(1, self.vocab_size + 1, 
                                            dtype=torch.float32, device=frequencies.device)
        
        zipf_weights = 1.0 / torch.pow(ranks, self.zipf_s)
        zipf_weights = zipf_weights / zipf_weights.sum()
        
        return zipf_weights

# ============================================
# MEG-adapted Cost Matrix Learner (from original)
# ============================================

class MEGCostMatrixLearner(nn.Module):
    """Learn frame-phoneme cost matrix for MEG-phoneme alignment."""
    
    def __init__(self, vocab_size=39, meg_channels=306, hidden_dim=256, projection_dim=64):
        super().__init__()
        # MEG encoder (replacing wav2vec2)
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
        # meg_data: (B, channels, time_points)
        # Encode MEG features
        meg_features = self.meg_encoder(meg_data)  # (B, hidden_dim, T)
        meg_features = meg_features.transpose(1, 2)  # (B, T, hidden_dim)
        
        # Apply temporal attention
        meg_features, _ = self.temporal_attention(meg_features, meg_features, meg_features)
        meg_features = self.linear(meg_features)  # (B, T, projection_dim)
        
        # Encode text labels
        text_embeddings = self.label_embedding(text_labels)  # (B, L, projection_dim)
        
        # Compute cost matrix
        cost_matrix = -torch.matmul(text_embeddings, meg_features.transpose(1, 2))
        cost_matrix = F.softmax(cost_matrix, dim=1)  # (B, L, T)
        
        return cost_matrix

# ============================================
# Enhanced MEG LCS-CTC Model v4
# ============================================

class EnhancedMEGLCSCTC(L.LightningModule):
    """Enhanced MEG LCS-CTC v4 with comprehensive configurability."""
    
    def __init__(self,
                 meg_channels=306,
                 time_points=125,
                 vocab_size=39,
                 hidden_dim=256,
                 num_conformers=4,
                 learning_rate=1e-4,
                 weight_decay=0.01,
                 dropout_rate=0.3,
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
                 # NEW v4 configurations
                 conformer_config=None,
                 temporal_config=None,
                 classifier_config=None,
                 spatial_config=None,
                 hierarchical_classification=None,  # ADD THIS
                 ensemble=None,                      # ADD THIS
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
        
        # Parse configs with defaults
        conformer_config = conformer_config or {}
        temporal_config = temporal_config or {}
        classifier_config = classifier_config or {}
        
        # Dynamic channel selector
        self.use_dynamic_channels = use_dynamic_channels
        if use_dynamic_channels:
            self.channel_selector = DynamicChannelSelector(
                vocab_size, meg_channels, coordinate_file
            )
        
        # MEG feature extraction with multi-scale temporal encoding
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
            # Original encoder
            self.meg_encoder = nn.Sequential(
                nn.Conv1d(meg_channels, hidden_dim, kernel_size=7, padding=3),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )
        
        # Temporal modeling with Conformers
        self.conformers = nn.ModuleList([
            MEGConformerLayer(
                hidden_dim, 
                num_heads=conformer_config.get('num_heads', 4),
                ff_dim=hidden_dim * conformer_config.get('ff_expansion_factor', 2),
                kernel_size=conformer_config.get('conv_kernel_size', 3),
                dropout=conformer_config.get('dropout', dropout_rate),
                use_macaron_style=conformer_config.get('use_macaron_style', False)
            )
            for _ in range(num_conformers)
        ])
        
        # CTC components
        self.ctc_projection = nn.Linear(hidden_dim, vocab_size + 1)
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        
        # Enhanced classification head
        if classifier_config.get('hidden_dims'):
            layers = []
            input_dim = hidden_dim * time_points
            hidden_dims = classifier_config['hidden_dims']
            dropout_rates = classifier_config.get('dropout_rates', [dropout_rate] * len(hidden_dims))
            activation = classifier_config.get('activation', 'relu')
            use_batch_norm = classifier_config.get('use_batch_norm', False)
            
            for hidden_dim_cls, dropout in zip(hidden_dims, dropout_rates):
                layers.append(nn.Linear(input_dim, hidden_dim_cls))
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim_cls))
                
                if activation == 'gelu':
                    layers.append(nn.GELU())
                elif activation == 'silu':
                    layers.append(nn.SiLU())
                else:
                    layers.append(nn.ReLU())
                
                layers.append(nn.Dropout(dropout))
                input_dim = hidden_dim_cls
            
            layers.append(nn.Linear(input_dim, vocab_size))
            self.classifier = nn.Sequential(*layers)
        else:
            # Original classifier
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * time_points, 512),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(512, vocab_size)
            )
        
        # Cost matrix learner
        self.use_alignment = use_alignment
        if use_alignment:
            self.cost_learner = MEGCostMatrixLearner(
                vocab_size, meg_channels, hidden_dim
            )
        
        # Enhanced Zipf weight learner
        self.use_zipf = zipf_weights
        self.adaptive_zipf_boost = adaptive_zipf_boost
        self.zipf_boost_factor = zipf_boost_factor
        
        if zipf_weights:
            if adaptive_zipf_boost:
                self.zipf_learner = AdaptiveZipfWeightLearner(
                    vocab_size, hidden_dim, zipf_alpha,
                    min_zipf_boost, max_zipf_boost
                )
            else:
                self.zipf_learner = ZipfWeightLearner(
                    vocab_size, hidden_dim, zipf_alpha
                )
            
            self.meg_aggregator = nn.Sequential(
                nn.Linear(hidden_dim * time_points, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
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
    
    def update_class_weights(self):
        """Update focal loss alpha weights with smoothing."""
        if self.use_focal_loss and self.total_samples > 0:
            frequencies = self.class_counts / self.total_samples
            # Inverse frequency weighting
            alpha = 1.0 / (frequencies + 1e-5)
            alpha = alpha / alpha.sum() * self.hparams.vocab_size
            
            # Apply smoothing
            if self.focal_alpha_smoothing > 0:
                uniform_weight = torch.ones_like(alpha) * self.hparams.vocab_size / len(alpha)
                alpha = (1 - self.focal_alpha_smoothing) * alpha + self.focal_alpha_smoothing * uniform_weight
            
            alpha = torch.clamp(alpha, 0.1, 10.0)
            
            if hasattr(self.criterion, 'alpha'):
                self.criterion.alpha = alpha
    
    def forward(self, x, labels=None, use_ctc=False):
        B, C, T = x.shape
        
        # Apply dynamic channel selection
        if self.use_dynamic_channels and not use_ctc:
            x, channel_mask = self.channel_selector(x, labels)
        
        # Encode MEG features
        features = self.meg_encoder(x)
        features = features.transpose(1, 2)
        
        # Apply conformers with stochastic depth
        for i, conformer in enumerate(self.conformers):
            if self.training and self.stochastic_depth_rate > 0:
                # Stochastic depth: randomly skip layers
                if torch.rand(1).item() > self.stochastic_depth_rate:
                    features = conformer(features)
            else:
                features = conformer(features)
        
        if use_ctc:
            logits = self.ctc_projection(features)
            return logits
        else:
            features_flat = features.reshape(B, -1)
            logits = self.classifier(features_flat)
            
            # Apply Zipf weighting
            if self.use_zipf and not self.training:
                meg_agg = self.meg_aggregator(features_flat)
                
                if self.adaptive_zipf_boost:
                    zipf_adjustments, boost_factors = self.zipf_learner(
                        meg_agg, labels, training=False
                    )
                    probs = F.softmax(logits, dim=-1)
                    for i in range(B):
                        boost = boost_factors[i] if labels is not None else 0.5
                        probs[i] = (1 - boost) * probs[i] + boost * zipf_adjustments[i]
                    logits = torch.log(probs + 1e-10)
                else:
                    zipf_adjustments = self.zipf_learner(meg_agg, training=False)
                    probs = F.softmax(logits, dim=-1)
                    adjusted_probs = (1 - self.zipf_boost_factor) * probs + \
                                   self.zipf_boost_factor * zipf_adjustments
                    logits = torch.log(adjusted_probs + 1e-10)
            
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
        
        # Update Zipf statistics
        if self.use_zipf:
            with torch.no_grad():
                B, C, T = x.shape
                features = self.meg_encoder(x)
                features = features.transpose(1, 2)
                for conformer in self.conformers:
                    features = conformer(features)
                features_flat = features.reshape(B, -1)
                meg_agg = self.meg_aggregator(features_flat)
                
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
        
        f1_macro = self.f1_macro(y_hat, y)
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
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x, labels=y, use_ctc=False)
        loss = self.criterion(y_hat, y)
        f1_macro = self.f1_macro(y_hat, y)
        
        self.log('val_loss', loss)
        self.log('val_f1_macro', f1_macro, prog_bar=True)
        
        # Compare with and without enhancements
        if self.use_zipf and batch_idx == 0:
            # Test without Zipf
            self.use_zipf = False
            y_hat_no_zipf = self(x, labels=y, use_ctc=False)
            self.use_zipf = True
            
            f1_no_zipf = self.f1_macro(y_hat_no_zipf, y)
            self.log('val_f1_no_zipf', f1_no_zipf)
            self.log('val_f1_zipf_gain', f1_macro - f1_no_zipf)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler with v4 config support."""
        params = [
            {'params': self.meg_encoder.parameters(), 'lr': self.hparams.learning_rate},
            {'params': self.conformers.parameters(), 'lr': self.hparams.learning_rate},
            {'params': self.classifier.parameters(), 'lr': self.hparams.learning_rate},
        ]
        
        if self.use_alignment:
            params.append({'params': self.cost_learner.parameters(), 
                          'lr': self.hparams.learning_rate})
        
        if self.use_zipf:
            params.append({'params': self.zipf_learner.parameters(), 
                          'lr': self.hparams.learning_rate * 0.1})
            params.append({'params': self.meg_aggregator.parameters(), 
                          'lr': self.hparams.learning_rate})
        
        if self.use_dynamic_channels:
            params.append({'params': self.channel_selector.parameters(), 
                          'lr': self.hparams.learning_rate * 0.5})
        
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
            # Default to AdamW
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
        else:
            # Default to cosine annealing
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=50, eta_min=1e-6
            )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_f1_macro' if scheduler_type == 'reduce_on_plateau' else None
        }