"""
Multi-Scale RNN Architecture for MEG Phoneme Classification
Inspired by NVIDIA's multi-scale diarization approach
Features:
- Multi-scale temporal processing
- Cross-channel attention and features
- Spectral-temporal feature extraction
- Data augmentation
- Dynamic scale weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import F1Score, Accuracy
import numpy as np
from typing import Optional, List, Dict, Tuple
import math

# ============================================
# Data Augmentation Module
# ============================================

class MEGAugmentation(nn.Module):
    """Data augmentation for MEG signals during training."""
    
    def __init__(self,
                 noise_level: float = 0.05,
                 time_jitter: float = 0.02,
                 channel_dropout: float = 0.1,
                 time_masking: float = 0.1,
                 freq_masking: float = 0.1):
        super().__init__()
        self.noise_level = noise_level
        self.time_jitter = time_jitter
        self.channel_dropout = channel_dropout
        self.time_masking = time_masking
        self.freq_masking = freq_masking
    
    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Apply augmentation to MEG data.
        x: (batch, channels, time)
        """
        if not training:
            return x
        
        B, C, T = x.shape
        device = x.device
        
        # Gaussian noise
        if self.noise_level > 0 and torch.rand(1).item() < 0.5:
            noise = torch.randn_like(x) * self.noise_level
            x = x + noise
        
        # Time jitter (small temporal shifts)
        if self.time_jitter > 0 and torch.rand(1).item() < 0.3:
            max_shift = int(T * self.time_jitter)
            if max_shift > 0:
                shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
                if shift != 0:
                    x = torch.roll(x, shifts=shift, dims=2)
        
        # Channel dropout
        if self.channel_dropout > 0 and torch.rand(1).item() < 0.3:
            channel_mask = torch.rand(B, C, 1, device=device) > self.channel_dropout
            x = x * channel_mask.float()
        
        # Time masking
        if self.time_masking > 0 and torch.rand(1).item() < 0.3:
            mask_size = int(T * self.time_masking)
            if mask_size > 0:
                for b in range(B):
                    start_idx = torch.randint(0, T - mask_size + 1, (1,)).item()
                    x[b, :, start_idx:start_idx + mask_size] = 0
        
        return x

# ============================================
# Spectral-Temporal Feature Extraction
# ============================================

class SpectralTemporalFeatures(nn.Module):
    """Extract spectral and temporal features from MEG signals."""
    
    def __init__(self,
                 num_channels: int = 306,
                 num_freq_bins: int = 20,
                 freq_min: float = 1.0,
                 freq_max: float = 40.0,
                 sfreq: float = 250.0):
        super().__init__()
        self.num_channels = num_channels
        self.num_freq_bins = num_freq_bins
        self.sfreq = sfreq
        
        # Create frequency bins (log-spaced)
        self.freq_bins = torch.logspace(
            np.log10(freq_min), np.log10(freq_max),
            num_freq_bins
        )
        
        # Learnable spectral filters (like formant extraction)
        self.spectral_conv = nn.Conv1d(
            num_channels, num_channels * num_freq_bins,
            kernel_size=11, padding=5, groups=num_channels
        )
        
        # Temporal envelope extraction
        self.envelope_conv = nn.Conv1d(
            num_channels, num_channels,
            kernel_size=5, padding=2
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract spectral-temporal features.
        x: (batch, channels, time)
        """
        B, C, T = x.shape
        
        # Spectral features via convolution (simulating filter banks)
        spectral = self.spectral_conv(x)  # (B, C*F, T)
        spectral = spectral.view(B, C, self.num_freq_bins, T)
        spectral = F.relu(spectral)
        
        # Temporal envelope
        envelope = torch.abs(self.envelope_conv(x))
        
        # Collapse spectral to single feature per channel per time
        # Average across frequency bins to get spectral centroid-like feature
        spectral_collapsed = spectral.mean(dim=2)  # (B, C, T)
        
        return {
            'spectral': spectral_collapsed,  # (B, C, T)
            'envelope': envelope,   # (B, C, T)
        }

# ============================================
# Multi-Scale Temporal Processing
# ============================================

class MultiScaleTemporalEncoder(nn.Module):
    """Process MEG signals at multiple temporal scales."""
    
    def __init__(self,
                 input_dim: int,
                 temporal_scales: List[int],
                 scale_hidden_dims: List[int],
                 rnn_type: str = "LSTM",
                 dropout: float = 0.1):
        super().__init__()
        assert len(temporal_scales) == len(scale_hidden_dims)
        
        self.temporal_scales = temporal_scales
        self.scale_hidden_dims = scale_hidden_dims
        self.num_scales = len(temporal_scales)
        
        # Create RNN for each scale
        self.scale_encoders = nn.ModuleList()
        for scale, hidden_dim in zip(temporal_scales, scale_hidden_dims):
            if rnn_type == "LSTM":
                encoder = nn.LSTM(
                    input_dim, hidden_dim,
                    num_layers=1, batch_first=True,
                    bidirectional=True, dropout=dropout
                )
            else:  # GRU
                encoder = nn.GRU(
                    input_dim, hidden_dim,
                    num_layers=1, batch_first=True,
                    bidirectional=True, dropout=dropout
                )
            self.scale_encoders.append(encoder)
        
        # Output dimension (sum of bidirectional hidden dims)
        self.output_dim = sum([dim * 2 for dim in scale_hidden_dims])
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Process input at multiple scales.
        x: (batch, time, features)
        Returns list of tensors, one per scale
        """
        B, T, F = x.shape
        scale_outputs = []
        
        for scale, encoder in zip(self.temporal_scales, self.scale_encoders):
            # Process entire sequence through RNN with different context sizes
            # Use the full sequence but let the RNN learn at different scales
            out, _ = encoder(x)  # Process full sequence
            
            # Pool over time dimension based on scale
            if scale < T:
                # Take outputs at regular intervals based on scale
                # This simulates looking at different temporal resolutions
                stride = max(1, scale // 4)  # Adaptive stride based on scale
                pooled = out[:, ::stride, :]  # Sample at intervals
                scale_out = pooled.mean(dim=1)  # Average over time
            else:
                # For larger scales, just take the mean of all outputs
                scale_out = out.mean(dim=1)
            
            scale_outputs.append(scale_out)
        
        return scale_outputs

# ============================================
# Cross-Channel Attention
# ============================================

class CrossChannelAttention(nn.Module):
    """Simplified attention mechanism across MEG channels."""
    
    def __init__(self,
                 num_channels: int = 306,
                 channel_groups: int = 6,
                 attention_dim: int = 64):
        super().__init__()
        self.num_channels = num_channels
        self.channel_groups = channel_groups
        self.channels_per_group = num_channels // channel_groups
        self.attention_dim = attention_dim
        
        # Channel projections
        self.channel_proj = nn.Linear(num_channels, attention_dim * channel_groups)
        
        # Group-wise attention (simplified)
        self.group_attention = nn.ModuleList([
            nn.Linear(self.channels_per_group, attention_dim)
            for _ in range(channel_groups)
        ])
        
        # Cross-group mixing
        self.cross_group_mixer = nn.Sequential(
            nn.Linear(attention_dim * channel_groups, attention_dim * channel_groups // 2),
            nn.ReLU(),
            nn.Linear(attention_dim * channel_groups // 2, attention_dim * channel_groups)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply cross-channel attention.
        x: (batch, channels, time) - we'll use the actual input shape
        """
        B, C, T = x.shape
        
        # Global channel statistics
        channel_mean = x.mean(dim=2)  # (B, C)
        channel_std = x.std(dim=2)  # (B, C)
        
        # Combine statistics as channel features
        channel_features = torch.cat([channel_mean, channel_std], dim=-1) if C == 306 else channel_mean
        
        # Project to attention dimension
        if channel_features.shape[-1] != C:
            # If we concatenated mean and std, project back
            channel_features = channel_mean  # Just use mean for simplicity
        
        projected = self.channel_proj(channel_features)  # (B, attention_dim * groups)
        projected = projected.view(B, self.channel_groups, self.attention_dim)
        
        # Group processing
        x_grouped = x.view(B, self.channel_groups, self.channels_per_group, T)
        group_features = []
        
        for g in range(self.channel_groups):
            # Get group data and compute features
            group_data = x_grouped[:, g]  # (B, channels_per_group, T)
            group_mean = group_data.mean(dim=2)  # (B, channels_per_group)
            
            # Project to attention dim
            group_feat = self.group_attention[g](group_mean)  # (B, attention_dim)
            group_features.append(group_feat)
        
        # Stack and mix groups
        all_groups = torch.stack(group_features, dim=1)  # (B, groups, attention_dim)
        all_groups_flat = all_groups.view(B, -1)  # (B, groups * attention_dim)
        
        # Cross-group interaction
        mixed = self.cross_group_mixer(all_groups_flat)  # (B, groups * attention_dim)
        
        return mixed  # (B, groups * attention_dim)

# ============================================
# Scale Weighting Mechanism (NVIDIA-inspired)
# ============================================

class ScaleWeightingModule(nn.Module):
    """Dynamic scale weighting inspired by NVIDIA's MSDD."""
    
    def __init__(self,
                 num_scales: int,
                 scale_dims: List[int],
                 weighting_type: str = "attention",
                 attention_dim: int = 64):
        super().__init__()
        self.num_scales = num_scales
        self.weighting_type = weighting_type
        
        if weighting_type == "attention":
            # Attention-based weighting
            self.scale_attention = nn.MultiheadAttention(
                sum(scale_dims), num_heads=4, batch_first=True
            )
            self.weight_proj = nn.Linear(sum(scale_dims), num_scales)
        elif weighting_type == "learned":
            # Learned static weights
            self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
        else:  # fixed
            self.register_buffer('scale_weights', torch.ones(num_scales) / num_scales)
        
        # 1D CNN for temporal weighting (NVIDIA approach)
        if weighting_type == "attention":
            self.temporal_conv = nn.Conv1d(sum(scale_dims), attention_dim, kernel_size=3, padding=1)
    
    def forward(self, scale_features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scale weights and combine features.
        scale_features: List of tensors from different scales
        Returns: (combined_features, scale_weights)
        """
        B = scale_features[0].shape[0]
        
        if self.weighting_type == "attention":
            # Concatenate scale features for attention computation
            combined = torch.cat(scale_features, dim=-1)  # (B, total_dim)
            combined_expanded = combined.unsqueeze(1)  # (B, 1, total_dim)
            attended, attention_weights = self.scale_attention(
                combined_expanded, combined_expanded, combined_expanded
            )
            
            # Project to scale weights
            scale_weights = F.softmax(self.weight_proj(attended.squeeze(1)), dim=-1)  # (B, num_scales)
            
            # Since features have different dimensions, we concatenate instead of sum
            # Apply weights and concatenate
            weighted_features = []
            for i, features in enumerate(scale_features):
                weight = scale_weights[:, i:i+1]  # (B, 1)
                weighted_features.append(features * weight.expand_as(features))
            
            # Concatenate weighted features instead of summing (due to different dims)
            combined_weighted = torch.cat(weighted_features, dim=-1)
            
        elif self.weighting_type == "learned":
            # Use learned static weights
            scale_weights = F.softmax(self.scale_weights, dim=0)
            weighted_features = []
            for i, features in enumerate(scale_features):
                weighted_features.append(features * scale_weights[i])
            # Concatenate instead of sum
            combined_weighted = torch.cat(weighted_features, dim=-1)
            scale_weights = scale_weights.unsqueeze(0).expand(B, -1)
            
        else:  # fixed
            # Equal weights - just concatenate
            combined_weighted = torch.cat(scale_features, dim=-1)
            scale_weights = self.scale_weights.unsqueeze(0).expand(B, -1)
        
        return combined_weighted, scale_weights

# ============================================
# Main Multi-Scale RNN Classifier
# ============================================

class MultiScaleRNNClassifier(L.LightningModule):
    """Multi-scale RNN classifier for MEG phoneme classification."""
    
    def __init__(self,
                 meg_channels: int = 306,
                 time_points: int = 125,
                 vocab_size: int = 39,
                 temporal_scales: List[int] = [8, 16, 32, 64],
                 scale_hidden_dims: List[int] = [32, 48, 64, 80],
                 rnn_type: str = "LSTM",
                 rnn_hidden_dim: int = 128,
                 rnn_num_layers: int = 3,
                 bidirectional: bool = True,
                 use_cross_channel: bool = True,
                 channel_groups: int = 6,
                 cross_channel_dim: int = 64,
                 use_spectral_features: bool = True,
                 num_freq_bins: int = 20,
                 freq_min: float = 1.0,
                 freq_max: float = 40.0,
                 augmentation: Optional[Dict] = None,
                 scale_weighting: str = "attention",
                 scale_attention_dim: int = 64,
                 learning_rate: float = 0.0005,
                 weight_decay: float = 0.01,
                 dropout_rate: float = 0.2,
                 label_smoothing: float = 0.1,
                 loss_type: str = "focal",
                 focal_gamma: float = 2.0,
                 warmup_epochs: int = 3,
                 scheduler_type: str = "cosine",
                 fusion_method: str = "weighted_sum",
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        # Data augmentation
        if augmentation and augmentation.get('enable', False):
            self.augmentation = MEGAugmentation(
                noise_level=augmentation.get('noise_level', 0.05),
                time_jitter=augmentation.get('time_jitter', 0.02),
                channel_dropout=augmentation.get('channel_dropout', 0.1),
                time_masking=augmentation.get('time_masking', 0.1),
                freq_masking=augmentation.get('freq_masking', 0.1)
            )
        else:
            self.augmentation = None
        
        # Spectral-temporal features
        if use_spectral_features:
            self.spectral_features = SpectralTemporalFeatures(
                num_channels=meg_channels,
                num_freq_bins=num_freq_bins,
                freq_min=freq_min,
                freq_max=freq_max
            )
            # Now we have: original (C) + spectral (C) + envelope (C) = 3*C channels
            # Project to a reasonable dimension
            self.spectral_projection = nn.Linear(
                meg_channels * 3,  # original + spectral + envelope
                meg_channels * 2  # Double the channel dimension
            )
            encoder_input_dim = meg_channels * 2
        else:
            self.spectral_features = None
            self.spectral_projection = None
            encoder_input_dim = meg_channels
        
        # Multi-scale temporal encoder
        self.multi_scale_encoder = MultiScaleTemporalEncoder(
            input_dim=encoder_input_dim,
            temporal_scales=temporal_scales,
            scale_hidden_dims=scale_hidden_dims,
            rnn_type=rnn_type,
            dropout=dropout_rate
        )
        
        # Scale weighting
        self.scale_weighting = ScaleWeightingModule(
            num_scales=len(temporal_scales),
            scale_dims=[dim * 2 for dim in scale_hidden_dims],  # *2 for bidirectional
            weighting_type=scale_weighting,
            attention_dim=scale_attention_dim
        )
        
        # Cross-channel attention
        if use_cross_channel:
            self.cross_channel = CrossChannelAttention(
                num_channels=meg_channels,
                channel_groups=channel_groups,
                attention_dim=cross_channel_dim
            )
            cross_channel_output_dim = channel_groups * cross_channel_dim
        else:
            self.cross_channel = None
            cross_channel_output_dim = 0
        
        # Main RNN for temporal modeling
        rnn_input_dim = sum([dim * 2 for dim in scale_hidden_dims]) + cross_channel_output_dim
        
        if rnn_type == "LSTM":
            self.main_rnn = nn.LSTM(
                rnn_input_dim, rnn_hidden_dim,
                num_layers=rnn_num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout_rate if rnn_num_layers > 1 else 0
            )
        else:  # GRU
            self.main_rnn = nn.GRU(
                rnn_input_dim, rnn_hidden_dim,
                num_layers=rnn_num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout_rate if rnn_num_layers > 1 else 0
            )
        
        rnn_output_dim = rnn_hidden_dim * (2 if bidirectional else 1)
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(rnn_output_dim, rnn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(rnn_hidden_dim, vocab_size)
        )
        
        # Loss and metrics
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        
        self.train_f1 = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
        self.val_f1 = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
        self.test_f1 = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
        
        # Training parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.scheduler_type = scheduler_type
        self.fusion_method = fusion_method
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        x: (batch, channels, time)
        """
        B, C, T = x.shape
        
        # Apply augmentation during training
        if self.augmentation and self.training:
            x = self.augmentation(x, training=True)
        
        # Extract spectral-temporal features
        if self.spectral_features:
            spectral_feats = self.spectral_features(x)
            
            # Combine features
            spectral = spectral_feats['spectral']  # (B, C, T)
            envelope = spectral_feats['envelope']  # (B, C, T)
            
            # Concatenate all features
            x_combined = torch.cat([x, spectral, envelope], dim=1)  # (B, 3*C, T)
            
            # Transpose and project to reasonable dimension
            x_combined = x_combined.transpose(1, 2)  # (B, T, 3*C)
            x_combined = self.spectral_projection(x_combined)  # (B, T, C*2)
        else:
            x_combined = x
            # Transpose for RNN processing
            x_combined = x_combined.transpose(1, 2)  # (B, T, C)
        
        # Multi-scale temporal encoding
        scale_outputs = self.multi_scale_encoder(x_combined)
        
        # Dynamic scale weighting
        combined_scales, scale_weights = self.scale_weighting(scale_outputs)
        
        # Cross-channel attention
        if self.cross_channel:
            # Use original x for cross-channel processing (B, C, T)
            cross_channel_feats = self.cross_channel(x)  # (B, groups * attention_dim)
            
            # Combine with scale features
            combined_features = torch.cat([combined_scales, cross_channel_feats], dim=-1)
        else:
            combined_features = combined_scales
        
        # Process through main RNN
        # Expand to sequence for RNN (use same features for all timesteps as context)
        combined_features = combined_features.unsqueeze(1).expand(-1, T, -1)
        rnn_out, _ = self.main_rnn(combined_features)
        
        # Take the last output
        final_features = rnn_out[:, -1, :]
        
        # Classification
        logits = self.classifier(self.dropout(final_features))
        
        return logits
    
    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss with focal loss option."""
        if self.loss_type == "focal":
            # Focal loss implementation
            ce_loss = F.cross_entropy(logits, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** self.focal_gamma) * ce_loss
            loss = focal_loss.mean()
            
            # Add label smoothing if specified
            if self.label_smoothing > 0:
                smooth_loss = F.cross_entropy(
                    logits, targets,
                    label_smoothing=self.label_smoothing
                )
                loss = 0.7 * loss + 0.3 * smooth_loss
        else:
            # Standard cross-entropy
            loss = F.cross_entropy(
                logits, targets,
                label_smoothing=self.label_smoothing if self.label_smoothing > 0 else 0.0
            )
        
        return loss
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.compute_loss(logits, y)
        
        # Metrics
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = (preds == y).float().mean()
            f1 = self.train_f1(logits, y)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_f1_macro', f1, prog_bar=True)
        self.log('train_acc', acc)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.compute_loss(logits, y)
        
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()
        f1 = self.val_f1(logits, y)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_f1_macro', f1, prog_bar=True)
        self.log('val_acc', acc)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.compute_loss(logits, y)
        
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()
        f1 = self.test_f1(logits, y)
        
        self.log('test_loss', loss)
        self.log('test_f1_macro', f1)
        self.log('test_acc', acc)
        
        return loss
    
    def configure_optimizers(self):
        # Optimizer with weight decay
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        if self.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=1e-6
            )
        else:  # step
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=10,
                gamma=0.5
            )
        
        # Warmup
        if self.warmup_epochs > 0:
            def warmup_lambda(epoch):
                if epoch < self.warmup_epochs:
                    return float(epoch) / float(max(1, self.warmup_epochs))
                return 1.0
            
            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=warmup_lambda
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': warmup_scheduler,
                    'monitor': 'val_loss'
                }
            }
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }