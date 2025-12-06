# spd_geometric.py
"""
SPD Geometric Network for MEG Phoneme Classification
Adapts the EMG-to-speech geometric approach with spatial attention and residual dilated convolutions
Designed for robust out-of-domain generalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np
from typing import Optional, Tuple
from torchmetrics import F1Score
import math


# ============================================
# SPD Manifold Operations
# ============================================

class SPDOperations:
    """Operations on the manifold of Symmetric Positive Definite matrices."""
    
    @staticmethod
    def matrix_log(X: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
        """Compute matrix logarithm for SPD matrices."""
        # Add regularization for numerical stability
        X = X + epsilon * torch.eye(X.size(-1), device=X.device, dtype=X.dtype)
        
        # Eigendecomposition
        eigvals, eigvecs = torch.linalg.eigh(X)
        eigvals = eigvals.clamp(min=epsilon)
        
        # Compute log
        log_eigvals = torch.log(eigvals)
        return eigvecs @ torch.diag_embed(log_eigvals) @ eigvecs.transpose(-2, -1)
    
    @staticmethod
    def matrix_exp(X: torch.Tensor) -> torch.Tensor:
        """Compute matrix exponential for symmetric matrices."""
        eigvals, eigvecs = torch.linalg.eigh(X)
        exp_eigvals = torch.exp(eigvals)
        return eigvecs @ torch.diag_embed(exp_eigvals) @ eigvecs.transpose(-2, -1)
    
    @staticmethod
    def frechet_mean(matrices: torch.Tensor, max_iters: int = 10, epsilon: float = 1e-6) -> torch.Tensor:
        """Compute Fréchet mean of SPD matrices."""
        B, N, D, _ = matrices.shape  # B: batch, N: num matrices, D: dimension
        
        # Initialize with arithmetic mean
        mean = matrices.mean(dim=1)
        mean = mean + epsilon * torch.eye(D, device=matrices.device).unsqueeze(0)
        
        for _ in range(max_iters):
            # Compute tangent space representations
            tangent_sum = torch.zeros_like(mean)
            for i in range(N):
                diff = SPDOperations.matrix_log(
                    torch.linalg.solve(mean, matrices[:, i]) @ mean
                )
                tangent_sum += diff
            
            # Update mean
            tangent_mean = tangent_sum / N
            mean = mean @ SPDOperations.matrix_exp(tangent_mean)
        
        return mean


# ============================================
# SPD Feature Extractor
# ============================================

class SPDFeatureExtractor(nn.Module):
    """Extract SPD matrix features from MEG signals."""
    
    def __init__(self, n_channels: int, window_size: int, window_stride: int, 
                 eigenbasis_rank: int = 64):
        super().__init__()
        self.n_channels = n_channels
        self.window_size = window_size
        self.window_stride = window_stride
        self.eigenbasis_rank = min(eigenbasis_rank, n_channels)
        
        # Learnable shrinkage parameter for regularization
        self.shrinkage = nn.Parameter(torch.tensor(0.1))
        
        # Channel attention for weighted covariance
        self.channel_attention = nn.Sequential(
            nn.Linear(n_channels, n_channels // 4),
            nn.ReLU(),
            nn.Linear(n_channels // 4, n_channels),
            nn.Sigmoid()
        )
        
    def compute_spd_matrices(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute SPD covariance matrices from sliding windows.
        
        Args:
            x: (B, C, T) MEG signals
            
        Returns:
            (B, N_windows, C, C) SPD matrices
        """
        B, C, T = x.shape
        
        # Apply channel attention
        channel_weights = self.channel_attention(x.mean(dim=-1))  # (B, C)
        x = x * channel_weights.unsqueeze(-1)
        
        # Extract sliding windows
        windows = x.unfold(dimension=2, size=self.window_size, step=self.window_stride)
        # windows: (B, C, N_windows, window_size)
        
        N_windows = windows.size(2)
        windows = windows.permute(0, 2, 1, 3)  # (B, N_windows, C, window_size)
        
        # Compute covariance matrices
        spd_matrices = []
        for i in range(N_windows):
            window = windows[:, i]  # (B, C, window_size)
            
            # Center the data
            window = window - window.mean(dim=-1, keepdim=True)
            
            # Compute covariance
            cov = torch.bmm(window, window.transpose(1, 2)) / (self.window_size - 1)
            
            # Apply shrinkage regularization (Ledoit-Wolf style)
            trace = torch.diagonal(cov, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
            eye = torch.eye(C, device=cov.device, dtype=cov.dtype).unsqueeze(0)
            shrinkage_target = (trace / C).unsqueeze(-1) * eye
            
            cov_regularized = (1 - self.shrinkage) * cov + self.shrinkage * shrinkage_target
            spd_matrices.append(cov_regularized)
        
        return torch.stack(spd_matrices, dim=1)  # (B, N_windows, C, C)
    
    def approximate_diagonalization(self, spd_matrices: torch.Tensor, 
                                   eigenbasis: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Transform SPD matrices to approximate diagonal form using common eigenbasis.
        
        Args:
            spd_matrices: (B, N, C, C) SPD matrices
            eigenbasis: Optional (C, R) eigenvector matrix
            
        Returns:
            (B, N, R) diagonal elements in spectral domain
        """
        B, N, C, _ = spd_matrices.shape
        
        if eigenbasis is None:
            # Compute Fréchet mean and extract eigenbasis
            frechet_mean = SPDOperations.frechet_mean(spd_matrices)
            eigvals, eigvecs = torch.linalg.eigh(frechet_mean)
            
            # Select top eigenvectors
            idx = torch.argsort(eigvals, dim=-1, descending=True)
            eigvecs = torch.gather(eigvecs, -1, idx.unsqueeze(-2).expand(-1, C, -1))
            eigenbasis = eigvecs[..., :self.eigenbasis_rank]  # (B, C, R)
        
        # Transform to spectral domain
        spectral_features = []
        for i in range(N):
            # Q^T @ SPD @ Q for approximate diagonalization
            transformed = eigenbasis.transpose(-2, -1) @ spd_matrices[:, i] @ eigenbasis
            # Extract diagonal (approximate eigenvalues)
            diag = torch.diagonal(transformed, dim1=-2, dim2=-1)
            spectral_features.append(diag)
        
        return torch.stack(spectral_features, dim=1)  # (B, N, R)


# ============================================
# Spatial Attention Module
# ============================================

class SpatialAttentionModule(nn.Module):
    """Multi-head spatial attention for MEG channels."""
    
    def __init__(self, n_channels: int, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.channel_embed = nn.Linear(n_channels, hidden_dim)
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, num_heads, 
                                                    dropout=0.1, batch_first=True)
        
        # Spatial relationship encoding
        self.spatial_encoder = nn.Sequential(
            nn.Conv1d(n_channels, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial attention to MEG channels.
        
        Args:
            x: (B, C, T) MEG signals
            
        Returns:
            (B, hidden_dim, T) attended features
        """
        B, C, T = x.shape
        
        # Encode spatial relationships
        spatial_features = self.spatial_encoder(x)  # (B, hidden_dim, T)
        
        # Prepare for attention
        x_transpose = x.transpose(1, 2)  # (B, T, C)
        x_embed = self.channel_embed(x_transpose)  # (B, T, hidden_dim)
        
        # Apply multi-head attention across time
        attended, _ = self.multihead_attn(x_embed, x_embed, x_embed)  # (B, T, hidden_dim)
        
        # Combine with spatial features
        attended = attended.transpose(1, 2) + spatial_features  # (B, hidden_dim, T)
        
        return attended


# ============================================
# Residual Dilated Convolution Block
# ============================================

class ResidualDilatedBlock(nn.Module):
    """Residual block with dilated convolutions for multi-scale temporal modeling."""
    
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, 
                 dropout: float = 0.1):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, 
                               dilation=dilation, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               dilation=dilation, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        
        self.dropout = nn.Dropout(dropout)
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply residual dilated convolution.
        
        Args:
            x: (B, C, T) input features
            
        Returns:
            (B, C, T) output features
        """
        residual = x
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.gelu(out)
        out = self.dropout(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Gating
        gate = self.gate(out)
        out = out * gate
        
        # Residual connection
        out = out + residual
        out = F.gelu(out)
        
        return out


# ============================================
# Manifold-aware GRU
# ============================================

class ManifoldGRU(nn.Module):
    """GRU that operates on SPD manifold features."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        
        # Standard GRU for spectral features
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, 
                         batch_first=True, bidirectional=True, dropout=0.2)
        
        # Projection for manifold features
        self.manifold_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process spectral features with manifold-aware GRU.
        
        Args:
            x: (B, T, D) spectral features
            
        Returns:
            (B, hidden_dim) aggregated features
        """
        # Apply manifold projection
        x_proj = self.manifold_proj(x)
        
        # Process with GRU
        gru_out, _ = self.gru(x_proj)  # (B, T, hidden_dim * 2)
        
        # Project and aggregate
        out = self.output_proj(gru_out)  # (B, T, hidden_dim)
        
        # Temporal pooling with attention
        attention_weights = F.softmax(out.mean(dim=-1), dim=1).unsqueeze(-1)
        out = (out * attention_weights).sum(dim=1)  # (B, hidden_dim)
        
        return out


# ============================================
# Main SPD Geometric Network
# ============================================

class SPDGeometricNetwork(L.LightningModule):
    """
    SPD Geometric Network combining manifold representation with spatial attention
    and residual dilated convolutions for robust MEG phoneme classification.
    """
    
    def __init__(self,
                 time_points: int = 125,
                 n_channels: int = 306,
                 n_classes: int = 39,
                 learning_rate: float = 0.0003,
                 weight_decay: float = 0.01,
                 label_smoothing: float = 0.0,
                 # SPD configuration
                 window_size: int = 25,
                 window_stride: int = 10,
                 eigenbasis_rank: int = 64,
                 # Architecture configuration
                 hidden_dim: int = 256,
                 num_dilated_blocks: int = 3,
                 dilation_rates: Optional[list] = None,
                 spatial_attention_heads: int = 8,
                 dropout_rate: float = 0.3,
                 use_mixup: bool = True,
                 mixup_alpha: float = 0.2,
                 # Manifold configuration
                 use_manifold_gru: bool = True,
                 manifold_type: str = "spd"):
        
        super().__init__()
        self.save_hyperparameters()
        
        if dilation_rates is None:
            dilation_rates = [1, 2, 4, 8]
        
        # SPD feature extraction
        self.spd_extractor = SPDFeatureExtractor(
            n_channels, window_size, window_stride, eigenbasis_rank
        )
        
        # Spatial attention
        self.spatial_attention = SpatialAttentionModule(
            n_channels, hidden_dim, spatial_attention_heads
        )
        
        # Calculate temporal dimension after SPD extraction
        n_windows = (time_points - window_size) // window_stride + 1
        
        # Temporal feature projection
        self.temporal_proj = nn.Sequential(
            nn.Linear(eigenbasis_rank, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Residual dilated convolution blocks
        self.dilated_blocks = nn.ModuleList()
        for _ in range(num_dilated_blocks):
            block = nn.ModuleList([
                ResidualDilatedBlock(hidden_dim, kernel_size=3, 
                                   dilation=d, dropout=dropout_rate)
                for d in dilation_rates
            ])
            self.dilated_blocks.append(block)
        
        # Manifold-aware GRU
        if use_manifold_gru:
            self.manifold_gru = ManifoldGRU(hidden_dim, hidden_dim, num_layers=2)
        else:
            self.manifold_gru = None
        
        # Feature fusion
        fusion_input_dim = hidden_dim * 3 if use_manifold_gru else hidden_dim * 2
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Classification head with regularization
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 1.5),  # Extra dropout before final layer
            nn.Linear(hidden_dim // 2, n_classes)
        )
        
        # Loss and metrics
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.f1_macro = F1Score(num_classes=n_classes, average='macro', task="multiclass")
        
        # Store eigenbasis for consistency - initialize with proper tensor to avoid loading issues
        self.register_buffer('eigenbasis', torch.zeros(n_channels, eigenbasis_rank), persistent=True)
        
    def mixup_data(self, x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
        """Apply mixup augmentation."""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: (B, C, T) MEG signals
            
        Returns:
            (B, n_classes) logits
        """
        B, C, T = x.shape
        
        # Extract SPD matrices
        spd_matrices = self.spd_extractor.compute_spd_matrices(x)  # (B, N_windows, C, C)
        
        # Check if eigenbasis needs initialization (all zeros means uninitialized)
        eigenbasis_to_use = self.eigenbasis
        if torch.allclose(self.eigenbasis, torch.zeros_like(self.eigenbasis)):
            # Eigenbasis not initialized, compute it
            with torch.no_grad():
                frechet_mean = SPDOperations.frechet_mean(spd_matrices)
                eigvals, eigvecs = torch.linalg.eigh(frechet_mean)
                idx = torch.argsort(eigvals, dim=-1, descending=True)
                eigvecs = torch.gather(eigvecs, -1, idx.unsqueeze(-2).expand(-1, C, -1))
                eigenbasis_to_use = eigvecs[..., :self.hparams.eigenbasis_rank].mean(dim=0)
                
                # Store it for consistency if in training mode
                if self.training:
                    self.eigenbasis.copy_(eigenbasis_to_use)
        
        # Approximate diagonalization to spectral features
        spectral_features = self.spd_extractor.approximate_diagonalization(
            spd_matrices, eigenbasis_to_use
        )  # (B, N_windows, eigenbasis_rank)
        
        # Project spectral features
        spectral_features = self.temporal_proj(spectral_features)  # (B, N_windows, hidden_dim)
        spectral_features = spectral_features.transpose(1, 2)  # (B, hidden_dim, N_windows)
        
        # Apply spatial attention to original signal
        spatial_features = self.spatial_attention(x)  # (B, hidden_dim, T)
        
        # Resize spatial features to match spectral features temporal dimension
        spatial_features = F.adaptive_avg_pool1d(spatial_features, spectral_features.size(-1))
        
        # Apply residual dilated convolutions to both feature types
        dilated_spectral = spectral_features
        dilated_spatial = spatial_features
        
        for block_group in self.dilated_blocks:
            # Multi-scale processing for spectral features
            spectral_multiscale = []
            spatial_multiscale = []
            
            for dilated_block in block_group:
                spectral_multiscale.append(dilated_block(dilated_spectral))
                spatial_multiscale.append(dilated_block(dilated_spatial))
            
            # Aggregate multi-scale features
            dilated_spectral = sum(spectral_multiscale) / len(spectral_multiscale)
            dilated_spatial = sum(spatial_multiscale) / len(spatial_multiscale)
        
        # Global pooling
        spectral_pooled = F.adaptive_avg_pool1d(dilated_spectral, 1).squeeze(-1)  # (B, hidden_dim)
        spatial_pooled = F.adaptive_avg_pool1d(dilated_spatial, 1).squeeze(-1)  # (B, hidden_dim)
        
        # Process with manifold GRU if enabled
        if self.manifold_gru is not None:
            # Prepare features for GRU
            combined_features = (dilated_spectral + dilated_spatial) / 2
            combined_features = combined_features.transpose(1, 2)  # (B, N_windows, hidden_dim)
            manifold_features = self.manifold_gru(combined_features)  # (B, hidden_dim)
            
            # Combine all features
            fused_features = torch.cat([spectral_pooled, spatial_pooled, manifold_features], dim=-1)
        else:
            fused_features = torch.cat([spectral_pooled, spatial_pooled], dim=-1)
        
        # Feature fusion
        fused_features = self.feature_fusion(fused_features)  # (B, hidden_dim)
        
        # Classification
        logits = self.classifier(fused_features)  # (B, n_classes)
        
        return logits
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # Apply mixup if enabled
        if self.hparams.use_mixup and np.random.random() > 0.5:
            x, y_a, y_b, lam = self.mixup_data(x, y, self.hparams.mixup_alpha)
            
            logits = self(x)
            loss = lam * self.criterion(logits, y_a) + (1 - lam) * self.criterion(logits, y_b)
            
            # Compute F1 on the primary target
            f1_macro = self.f1_macro(logits, y_a)
        else:
            logits = self(x)
            loss = self.criterion(logits, y)
            f1_macro = self.f1_macro(logits, y)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_f1_macro', f1_macro)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        f1_macro = self.f1_macro(logits, y)
        
        self.log('val_loss', loss)
        self.log('val_f1_macro', f1_macro, prog_bar=True)
        
        # Log per-class accuracy for analysis
        if batch_idx == 0:
            preds = torch.argmax(logits, dim=-1)
            for class_idx in range(self.hparams.n_classes):
                class_mask = y == class_idx
                if class_mask.sum() > 0:
                    class_acc = (preds[class_mask] == y[class_mask]).float().mean()
                    self.log(f'val_class_{class_idx}_acc', class_acc)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        f1_macro = self.f1_macro(logits, y)
        
        self.log('test_loss', loss)
        self.log('test_f1_macro', f1_macro)
        
        return loss
    
    def configure_optimizers(self):
        # Different learning rates for different components
        param_groups = [
            {'params': self.spd_extractor.parameters(), 'lr': self.hparams.learning_rate * 0.5},
            {'params': self.spatial_attention.parameters(), 'lr': self.hparams.learning_rate},
            {'params': self.dilated_blocks.parameters(), 'lr': self.hparams.learning_rate},
            {'params': self.classifier.parameters(), 'lr': self.hparams.learning_rate * 2}
        ]
        
        if self.manifold_gru is not None:
            param_groups.append({
                'params': self.manifold_gru.parameters(), 
                'lr': self.hparams.learning_rate * 0.75
            })
        
        # Add remaining parameters
        other_params = []
        for name, param in self.named_parameters():
            if not any(name.startswith(module) for module in 
                      ['spd_extractor', 'spatial_attention', 'dilated_blocks', 
                       'classifier', 'manifold_gru']):
                other_params.append(param)
        
        if other_params:
            param_groups.append({'params': other_params, 'lr': self.hparams.learning_rate})
        
        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.hparams.weight_decay)
        
        # Cosine annealing with warm restarts for better generalization
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }