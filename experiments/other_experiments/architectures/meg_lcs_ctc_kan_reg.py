"""
Enhanced MEG LCS-CTC v4 with KAN integration and Regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from torchmetrics import F1Score
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d

# Import KANLayer - adjust path as needed
from kan.KANLayer import KANLayer


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
# Multi-Scale Temporal Encoder (unchanged)
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
# MEG-adapted Cost Matrix Learner (unchanged)
# ============================================

class MEGCostMatrixLearner(nn.Module):
    """Learn frame-phoneme cost matrix for MEG-phoneme alignment."""
    
    def __init__(self, vocab_size=39, meg_channels=306, hidden_dim=128, projection_dim=64):
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
# Regularization Components
# ============================================

class ConfidencePenalty(nn.Module):
    """
    Penalizes overconfident predictions to prevent overfitting.
    """
    def __init__(self, penalty_weight=0.1, confidence_threshold=0.9):
        super().__init__()
        self.penalty_weight = penalty_weight
        self.confidence_threshold = confidence_threshold
    
    def forward(self, logits):
        """
        Args:
            logits: Raw model outputs (B, num_classes)
        Returns:
            penalty: Scalar penalty value
        """
        probs = F.softmax(logits, dim=-1)
        max_probs, _ = torch.max(probs, dim=-1)
        
        # Penalize predictions above threshold
        overconfident = (max_probs > self.confidence_threshold).float()
        penalty = self.penalty_weight * torch.mean(overconfident * (max_probs - self.confidence_threshold) ** 2)
        
        return penalty

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

class MEGBoundarySmoothing(nn.Module):
    """
    Applies Gaussian smoothing to MEG signal boundaries to reduce edge artifacts.
    """
    def __init__(self, sigma=1.0, boundary_ratio=0.1):
        super().__init__()
        self.sigma = sigma
        self.boundary_ratio = boundary_ratio
    
    def forward(self, x):
        """
        Args:
            x: MEG signals (B, C, T)
        Returns:
            Smoothed MEG signals
        """
        B, C, T = x.shape
        boundary_size = int(T * self.boundary_ratio)
        
        if boundary_size > 0:
            # Create weight mask for blending
            weights = torch.ones(T, device=x.device)
            
            # Smooth the boundaries
            for i in range(boundary_size):
                # Linear decay from edge to interior
                weight = i / boundary_size
                weights[i] = weight
                weights[T - 1 - i] = weight
            
            # Apply Gaussian smoothing to entire signal
            x_smoothed = x.clone()
            for b in range(B):
                for c in range(C):
                    # Convert to numpy for scipy processing
                    signal = x[b, c].cpu().numpy()
                    smoothed = gaussian_filter1d(signal, sigma=self.sigma, mode='reflect')
                    x_smoothed[b, c] = torch.from_numpy(smoothed).to(x.device)
            
            # Blend original and smoothed based on weights
            weights = weights.view(1, 1, -1)
            x = x * weights + x_smoothed * (1 - weights)
        
        return x


class TemperatureScaling(nn.Module):
    """
    Applies temperature scaling to model outputs for calibration.
    """
    def __init__(self, initial_temperature=1.0, learnable=True):
        super().__init__()
        if learnable:
            self.temperature = nn.Parameter(torch.tensor(initial_temperature))
        else:
            self.register_buffer('temperature', torch.tensor(initial_temperature))
    
    def forward(self, logits):
        """Scale logits by temperature."""
        return logits / self.temperature
    
    def get_temperature(self):
        """Get current temperature value."""
        return self.temperature.item()


def kan_spline_regularization(model, l2_weight=1e-4):
    """
    Compute L2 regularization for KAN spline coefficients.
    
    Args:
        model: The model containing KAN layers
        l2_weight: Weight for L2 regularization
    Returns:
        Regularization loss
    """
    reg_loss = 0.0
    for module in model.modules():
        if isinstance(module, KANLayer):
            # KAN layers typically have spline coefficients as parameters
            for param_name, param in module.named_parameters():
                if 'spline' in param_name.lower() or 'coef' in param_name.lower():
                    reg_loss += l2_weight * torch.sum(param ** 2)
    return reg_loss


# ============================================
# Modified KANFFN with regularization
# ============================================

class KANFFN(nn.Module):
    """Feed-forward network using KAN layers with spline regularization."""
    
    def __init__(self, in_dim, hidden_dim, out_dim, num=5, k=3, 
                 grid_range=[-1, 1], dropout=0.1, activation='silu',
                 spline_weight_init_scale=0.1):
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
        
        # Initialize spline weights with smaller values for regularization
        self._initialize_spline_weights(spline_weight_init_scale)
    
    def _initialize_spline_weights(self, scale):
        """Initialize spline weights with controlled scale."""
        for module in [self.kan1, self.kan2]:
            for param_name, param in module.named_parameters():
                if 'spline' in param_name.lower() or 'coef' in param_name.lower():
                    with torch.no_grad():
                        param.data *= scale
    
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
# Enhanced MEG LCS-CTC Model v4 with KAN and Regularization
# ============================================

class EnhancedMEGLCSCTCKAN(L.LightningModule):
    """Enhanced MEG LCS-CTC v4 with KAN integration and regularization."""
    
    def __init__(self,
                 meg_channels=306,
                 time_points=125,
                 vocab_size=39,
                 hidden_dim=128,
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
                 # KAN parameters
                 kan_num=5,
                 kan_k=3,
                 kan_grid_range=[-1, 1],
                 kan_spline_init_scale=0.1,
                 # Regularization parameters (NEW)
                 use_confidence_penalty=True,
                 confidence_penalty_weight=0.1,
                 confidence_threshold=0.9,
                 use_boundary_smoothing=True,
                 boundary_smooth_sigma=1.0,
                 boundary_smooth_ratio=0.1,
                 use_temperature_scaling=True,
                 initial_temperature=1.5,
                 learnable_temperature=True,
                 kan_spline_l2_weight=1e-4,
                 # NEW v4 configurations
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
        self.kan_spline_l2_weight = kan_spline_l2_weight
        
        # Initialize regularization components
        self.use_confidence_penalty = use_confidence_penalty
        if use_confidence_penalty:
            self.confidence_penalty = ConfidencePenalty(
                penalty_weight=confidence_penalty_weight,
                confidence_threshold=confidence_threshold
            )
        
        self.use_boundary_smoothing = use_boundary_smoothing
        if use_boundary_smoothing:
            self.boundary_smoother = MEGBoundarySmoothing(
                sigma=boundary_smooth_sigma,
                boundary_ratio=boundary_smooth_ratio
            )
        
        self.use_temperature_scaling = use_temperature_scaling
        if use_temperature_scaling:
            self.temperature_scaler = TemperatureScaling(
                initial_temperature=initial_temperature,
                learnable=learnable_temperature
            )
        
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
        
        # KAN-based classification head with spline init scale
        if classifier_config.get('hidden_dims'):
            self.classifier = KANClassifier(
                input_dim=hidden_dim * time_points,
                hidden_dims=classifier_config['hidden_dims'],
                output_dim=vocab_size,
                kan_num=kan_num,
                kan_k=kan_k,
                dropout_rates=classifier_config.get('dropout_rates', [dropout_rate] * len(classifier_config['hidden_dims'])),
                grid_range=kan_grid_range,
                activation=classifier_config.get('activation', 'silu')
            )
        else:
            # Default KAN classifier
            self.classifier = KANClassifier(
                input_dim=hidden_dim * time_points,
                hidden_dims=[128],
                output_dim=vocab_size,
                kan_num=kan_num,
                kan_k=kan_k,
                dropout_rates=[dropout_rate],
                grid_range=kan_grid_range,
                activation='silu'
            )
        
        # Initialize KAN spline weights with controlled scale
        self._initialize_kan_splines(kan_spline_init_scale)
        
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
            
            # KAN-based MEG aggregator with spline init
            self.meg_aggregator = KANLayer(
                in_dim=hidden_dim * time_points,
                out_dim=hidden_dim,
                num=kan_num,
                k=kan_k,
                grid_range=kan_grid_range,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            # Initialize aggregator splines
            for param_name, param in self.meg_aggregator.named_parameters():
                if 'spline' in param_name.lower() or 'coef' in param_name.lower():
                    with torch.no_grad():
                        param.data *= kan_spline_init_scale
        
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
    
    def _initialize_kan_splines(self, scale):
        """Initialize KAN spline weights with controlled scale for regularization."""
        for module in self.modules():
            if isinstance(module, KANLayer):
                for param_name, param in module.named_parameters():
                    if 'spline' in param_name.lower() or 'coef' in param_name.lower():
                        with torch.no_grad():
                            param.data *= scale
    
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
        
        # Apply boundary smoothing to MEG signals
        if self.use_boundary_smoothing and self.training:
            x = self.boundary_smoother(x)
        
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
            
            # Apply temperature scaling
            if self.use_temperature_scaling:
                logits = self.temperature_scaler(logits)
            
            # Apply Zipf weighting
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
        
        # Main classification loss
        loss = self.criterion(y_hat, y)
        
        # Add confidence penalty
        if self.use_confidence_penalty:
            conf_penalty = self.confidence_penalty(y_hat)
            loss = loss + conf_penalty
            self.log('train_confidence_penalty', conf_penalty, prog_bar=False)
        
        # Add KAN spline regularization
        kan_reg = kan_spline_regularization(self, self.kan_spline_l2_weight)
        loss = loss + kan_reg
        self.log('train_kan_regularization', kan_reg, prog_bar=False)
        
        # Update Zipf statistics
        if self.use_zipf:
            with torch.no_grad():
                B, C, T = x.shape
                features = self.meg_encoder(x)
                features = features.transpose(1, 2)
                for conformer in self.conformers:
                    features = conformer(features)
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
        
        f1_macro = self.f1_macro(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_f1_macro', f1_macro)
        
        # Log regularization metrics
        if batch_idx % 100 == 0:
            # Log temperature if using temperature scaling
            if self.use_temperature_scaling:
                self.log('temperature', self.temperature_scaler.get_temperature())
            
            # Log confidence statistics
            with torch.no_grad():
                probs = F.softmax(y_hat, dim=-1)
                max_probs, _ = torch.max(probs, dim=-1)
                self.log('train_mean_confidence', max_probs.mean())
                self.log('train_max_confidence', max_probs.max())
                
                # Log entropy for monitoring prediction diversity
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                self.log('train_prediction_entropy', entropy.mean())
            
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
        
        # Compute base loss without regularization for fair comparison
        loss = self.criterion(y_hat, y)
        f1_macro = self.f1_macro(y_hat, y)
        
        self.log('val_loss', loss)
        self.log('val_f1_macro', f1_macro, prog_bar=True)
        
        # Log confidence statistics for validation
        with torch.no_grad():
            probs = F.softmax(y_hat, dim=-1)
            max_probs, _ = torch.max(probs, dim=-1)
            self.log('val_mean_confidence', max_probs.mean())
            
            # Log entropy
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            self.log('val_prediction_entropy', entropy.mean())
        
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
        # Separate KAN and non-KAN parameters for different learning rates
        kan_params = []
        other_params = []
        
        for name, param in self.named_parameters():
            if any(kan_module in name for kan_module in ['kan1', 'kan2', 'kan_layer', 'meg_aggregator', 'phoneme_meg_attention_kan']):
                kan_params.append(param)
            else:
                other_params.append(param)
        
        # Create parameter groups with potentially different learning rates
        params = [
            {'params': other_params, 'lr': self.hparams.learning_rate},
            {'params': kan_params, 'lr': self.hparams.learning_rate * 0.5}  # KAN layers often need lower LR
        ]
        
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

    def on_train_epoch_end(self):
        """Update KAN grids and log regularization statistics at the end of each epoch."""
        # Log regularization statistics
        if self.use_temperature_scaling:
            self.log('epoch_temperature', self.temperature_scaler.get_temperature())
        
        # Monitor KAN spline magnitudes
        total_spline_norm = 0.0
        num_spline_params = 0
        for module in self.modules():
            if isinstance(module, KANLayer):
                for param_name, param in module.named_parameters():
                    if 'spline' in param_name.lower() or 'coef' in param_name.lower():
                        total_spline_norm += torch.norm(param, p=2).item()
                        num_spline_params += 1
        
        if num_spline_params > 0:
            avg_spline_norm = total_spline_norm / num_spline_params
            self.log('avg_kan_spline_norm', avg_spline_norm)
    
    def test_step(self, batch, batch_idx):
        """Test step with detailed metrics."""
        x, y = batch
        y_hat = self(x, labels=y, use_ctc=False)
        loss = self.criterion(y_hat, y)
        f1_macro = self.f1_macro(y_hat, y)
        
        # Calculate accuracy
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        
        # Calculate confidence metrics
        with torch.no_grad():
            probs = F.softmax(y_hat, dim=-1)
            max_probs, _ = torch.max(probs, dim=-1)
            
            # Expected Calibration Error (ECE) - simplified version
            n_bins = 10
            bin_boundaries = torch.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = torch.zeros(1, device=x.device)
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (max_probs > bin_lower) * (max_probs <= bin_upper)
                prop_in_bin = in_bin.float().mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = ((preds == y) * in_bin).float().sum() / in_bin.float().sum()
                    avg_confidence_in_bin = (max_probs * in_bin).sum() / in_bin.float().sum()
                    ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            self.log('test_ece', ece.item())
        
        self.log('test_loss', loss)
        self.log('test_f1_macro', f1_macro)
        self.log('test_accuracy', acc)
        self.log('test_mean_confidence', max_probs.mean())
        
        return {'loss': loss, 'f1': f1_macro, 'accuracy': acc, 'ece': ece.item()}
    
    def on_test_epoch_end(self, outputs=None):
        """Aggregate test metrics at the end of testing."""
        if outputs:
            avg_ece = np.mean([x['ece'] for x in outputs])
            self.log('test_avg_ece', avg_ece)
            print(f"Average Expected Calibration Error: {avg_ece:.4f}")
