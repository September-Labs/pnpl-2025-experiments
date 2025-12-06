"""
MEG Transformer Phoneme Classification Model
Adapted for LibriBrain MEG phoneme decoding task with PyTorch Lightning.
Full architecture with configurable components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import F1Score
from typing import Optional, Tuple, List, Dict
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        # x shape: (batch, time, features)
        return x + self.pe[:, :x.size(1)]


class ChannelGroupProjector(nn.Module):
    """Projects channel groups to a common representation space."""
    
    def __init__(self, input_channels: int, hidden_dim: int, dropout: float = 0.1,
                 use_batch_norm: bool = True):
        super().__init__()
        layers = [
            nn.Conv1d(input_channels, hidden_dim // 2, kernel_size=1),
        ]
        
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim // 2))
        
        layers.extend([
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=1),
        ])
        
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        
        self.projection = nn.Sequential(*layers)
    
    def forward(self, x):
        # x shape: (batch, channels, time)
        return self.projection(x)


class TemporalFeatureExtractor(nn.Module):
    """Extracts multi-scale temporal features from MEG signals."""
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1,
                 kernel_sizes: List[int] = [3, 5, 7, 11], use_batch_norm: bool = True):
        super().__init__()
        
        self.kernel_sizes = kernel_sizes
        self.convs = nn.ModuleList()
        
        # Create convolution for each kernel size
        channels_per_kernel = hidden_dim // len(kernel_sizes)
        for i, kernel_size in enumerate(kernel_sizes):
            # Handle channel allocation for last kernel to ensure exact hidden_dim
            if i == len(kernel_sizes) - 1:
                out_channels = hidden_dim - (channels_per_kernel * (len(kernel_sizes) - 1))
            else:
                out_channels = channels_per_kernel
            
            conv = nn.Conv1d(input_dim, out_channels, kernel_size=kernel_size, 
                           padding=kernel_size // 2)
            self.convs.append(conv)
        
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch, channels, time)
        conv_outputs = []
        for conv in self.convs:
            conv_outputs.append(conv(x))
        
        # Concatenate multi-scale features
        multi_scale = torch.cat(conv_outputs, dim=1)
        
        if self.use_batch_norm:
            multi_scale = self.batch_norm(multi_scale)
        multi_scale = F.gelu(multi_scale)
        multi_scale = self.dropout(multi_scale)
        
        return multi_scale


class SpatialAttention(nn.Module):
    """Attention mechanism across MEG channel groups."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        # x shape: (batch, num_groups, hidden_dim)
        batch_size, num_groups, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, num_groups, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, num_groups, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, num_groups, self.num_heads, self.head_dim)
        
        q = q.transpose(1, 2)  # (batch, heads, groups, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        
        attended = torch.matmul(attn_weights, v)
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, num_groups, self.hidden_dim)
        
        return self.out_proj(attended)


class TransformerBlock(nn.Module):
    """Transformer encoder block with pre-norm."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, 
                 ff_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        # Pre-norm architecture
        norm_x = self.norm1(x)
        attn_out, _ = self.attention(norm_x, norm_x, norm_x)
        x = x + attn_out
        
        norm_x = self.norm2(x)
        ff_out = self.ff(norm_x)
        x = x + ff_out
        
        return x


class MEGTransformerPhoneme(L.LightningModule):
    """
    PyTorch Lightning MEG phoneme classification model with full configurability.
    
    Architecture:
    1. Channel grouping (magnetometers vs gradiometers)
    2. Multi-scale temporal feature extraction
    3. Spatial attention across channel groups
    4. Transformer encoding for temporal dependencies
    5. Hierarchical pooling and classification
    """
    
    def __init__(
        self,
        # Required by platform
        time_points: Optional[int] = None,
        learning_rate: float = 0.0001,
        
        # Architecture parameters
        num_meg_channels: int = 306,
        num_phonemes: int = 39,
        hidden_dim: int = 256,
        
        # Feature extraction
        use_channel_groups: bool = True,
        use_temporal_features: bool = True,
        temporal_kernel_sizes: List[int] = [3, 5, 7, 11],
        
        # Spatial attention
        use_spatial_attention: bool = True,
        spatial_attention_heads: int = 4,
        
        # Transformer parameters
        use_transformer: bool = True,
        num_transformer_layers: int = 4,
        transformer_heads: int = 8,
        transformer_ff_dim: Optional[int] = None,
        
        # Regularization
        dropout: float = 0.2,
        channel_dropout: float = 0.1,
        use_batch_norm: bool = True,
        
        # Positional encoding
        use_positional_encoding: bool = True,
        
        # Aggregation
        aggregate_method: str = "attention",  # "mean", "max", "attention"
        
        # Auxiliary loss
        use_auxiliary_loss: bool = True,
        auxiliary_loss_weight: float = 0.3,
        
        # Optimizer settings
        optimizer_type: str = "adam",  # "adam", "adamw", "sgd"
        weight_decay: float = 0.01,
        scheduler_type: Optional[str] = None,  # None, "onecycle", "cosine", "reduce"
        scheduler_params: Optional[Dict] = None,
        
        # Additional regularization
        label_smoothing: float = 0.0,  # Label smoothing for CrossEntropyLoss
        
        # Data preprocessing
        force_time_points: int = 125,  # Force input to this many time points
        
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Force time points to 125
        self.force_time_points = force_time_points
        
        # MEG channel grouping (102 magnetometers + 204 gradiometers)
        self.mag_channels = 102
        self.grad_channels = 204
        
        # Metrics
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.hparams.label_smoothing)
        self.f1_macro = F1Score(num_classes=num_phonemes, average='macro', task="multiclass")
        
        # Channel dropout for regularization
        if channel_dropout > 0:
            self.channel_dropout = nn.Dropout2d(channel_dropout)
        else:
            self.channel_dropout = nn.Identity()
        
        # Build model components based on configuration
        self._build_model()
        
    def _build_model(self):
        """Build model components based on configuration."""
        hidden_dim = self.hparams.hidden_dim
        dropout = self.hparams.dropout
        
        if self.hparams.use_channel_groups:
            # Channel group projectors
            self.mag_projector = ChannelGroupProjector(
                self.mag_channels, hidden_dim, dropout, self.hparams.use_batch_norm
            )
            self.grad_projector = ChannelGroupProjector(
                self.grad_channels, hidden_dim, dropout, self.hparams.use_batch_norm
            )
            
            if self.hparams.use_temporal_features:
                # Temporal feature extractors
                self.mag_temporal = TemporalFeatureExtractor(
                    hidden_dim, hidden_dim, dropout, 
                    self.hparams.temporal_kernel_sizes, self.hparams.use_batch_norm
                )
                self.grad_temporal = TemporalFeatureExtractor(
                    hidden_dim, hidden_dim, dropout,
                    self.hparams.temporal_kernel_sizes, self.hparams.use_batch_norm
                )
            
            if self.hparams.use_spatial_attention:
                # Spatial attention across channel groups
                self.spatial_attention = SpatialAttention(
                    hidden_dim, self.hparams.spatial_attention_heads
                )
            
            # Combine channel groups
            self.channel_fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        else:
            # Simple projection without channel grouping
            self.input_projection = nn.Sequential(
                nn.Conv1d(self.hparams.num_meg_channels, hidden_dim, kernel_size=1),
                nn.BatchNorm1d(hidden_dim) if self.hparams.use_batch_norm else nn.Identity(),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        
        # Positional encoding for temporal dimension
        if self.hparams.use_positional_encoding:
            self.positional_encoding = PositionalEncoding(hidden_dim, self.force_time_points)
        
        # Transformer encoder for temporal modeling
        if self.hparams.use_transformer:
            ff_dim = self.hparams.transformer_ff_dim or hidden_dim * 4
            self.transformer_layers = nn.ModuleList([
                TransformerBlock(
                    hidden_dim, self.hparams.transformer_heads, ff_dim, dropout
                )
                for _ in range(self.hparams.num_transformer_layers)
            ])
        
        # Temporal aggregation
        if self.hparams.aggregate_method == "attention":
            self.temporal_attention = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Softmax(dim=1)
            )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, self.hparams.num_phonemes)
        )
        
        # Auxiliary classifier for multi-scale features
        if self.hparams.use_auxiliary_loss:
            self.aux_classifier = nn.Linear(hidden_dim, self.hparams.num_phonemes)
        
        # Layer normalization for final features
        self.final_norm = nn.LayerNorm(hidden_dim)
    
    def extract_channel_groups(self, x):
        """Split MEG channels into magnetometers and gradiometers."""
        # x shape: (batch, channels, time)
        mag = x[:, :self.mag_channels, :]
        grad = x[:, self.mag_channels:, :]
        return mag, grad
    
    def adjust_time_points(self, x):
        """Adjust input to have exactly force_time_points."""
        # x shape: (batch, channels, time)
        current_time = x.shape[2]
        target_time = self.force_time_points
        
        if current_time == target_time:
            return x
        elif current_time > target_time:
            # Crop centered
            start = (current_time - target_time) // 2
            return x[:, :, start:start + target_time]
        else:
            # Pad with zeros
            pad_total = target_time - current_time
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            return F.pad(x, (pad_left, pad_right), mode='constant', value=0)
    
    def forward(self, x, return_features: bool = False):
        """
        Forward pass through the model.
        
        Args:
            x: Input MEG data of shape (batch, 306, time)
            return_features: If True, also return intermediate features
            
        Returns:
            logits: Shape (batch, 39) - unnormalized phoneme predictions
            features: Optional dict of intermediate features if return_features=True
        """
        # Adjust time points to exactly 125
        x = self.adjust_time_points(x)
        
        # Apply channel dropout for regularization
        x = self.channel_dropout(x.unsqueeze(1)).squeeze(1)
        
        if self.hparams.use_channel_groups:
            # Split into channel groups
            mag, grad = self.extract_channel_groups(x)
            
            # Project channel groups to hidden dimension
            mag_proj = self.mag_projector(mag)  # (batch, hidden_dim, time)
            grad_proj = self.grad_projector(grad)
            
            if self.hparams.use_temporal_features:
                # Extract multi-scale temporal features
                mag_temporal = self.mag_temporal(mag_proj)
                grad_temporal = self.grad_temporal(grad_proj)
            else:
                mag_temporal = mag_proj
                grad_temporal = grad_proj
            
            if self.hparams.use_spatial_attention:
                # Apply spatial attention across channel groups
                # First, create group representations by pooling over time
                mag_pooled = mag_temporal.mean(dim=2).unsqueeze(1)  # (batch, 1, hidden_dim)
                grad_pooled = grad_temporal.mean(dim=2).unsqueeze(1)
                
                spatial_features = torch.cat([mag_pooled, grad_pooled], dim=1)
                spatial_features = self.spatial_attention(spatial_features)
            else:
                spatial_features = None
            
            # Combine channel groups
            combined = torch.cat([
                mag_temporal.transpose(1, 2),  # (batch, time, hidden_dim)
                grad_temporal.transpose(1, 2)
            ], dim=2)
            
            features = self.channel_fusion(combined)  # (batch, time, hidden_dim)
        else:
            # Simple processing without channel grouping
            proj = self.input_projection(x)  # (batch, hidden_dim, time)
            features = proj.transpose(1, 2)  # (batch, time, hidden_dim)
            spatial_features = None
            mag_temporal = None
            grad_temporal = None
        
        # Add positional encoding
        if self.hparams.use_positional_encoding:
            features = self.positional_encoding(features)
        
        # Store features for auxiliary loss
        pre_transformer_features = features.mean(dim=1)
        
        # Apply transformer layers
        if self.hparams.use_transformer:
            for transformer in self.transformer_layers:
                features = transformer(features)
        
        # Temporal aggregation
        if self.hparams.aggregate_method == "mean":
            aggregated = features.mean(dim=1)
        elif self.hparams.aggregate_method == "max":
            aggregated = features.max(dim=1)[0]
        elif self.hparams.aggregate_method == "attention":
            attn_weights = self.temporal_attention(features)
            aggregated = (features * attn_weights).sum(dim=1)
        else:
            raise ValueError(f"Unknown aggregation method: {self.hparams.aggregate_method}")
        
        # Final normalization
        aggregated = self.final_norm(aggregated)
        
        # Classification
        logits = self.classifier(aggregated)
        
        if return_features:
            features_dict = {
                'aggregated_features': aggregated,
                'pre_transformer_features': pre_transformer_features,
                'spatial_attention': spatial_features,
                'mag_features': mag_temporal,
                'grad_features': grad_temporal
            }
            
            # Auxiliary predictions for training with intermediate supervision
            if self.hparams.use_auxiliary_loss:
                features_dict['aux_logits'] = self.aux_classifier(pre_transformer_features)
            
            return logits, features_dict
        
        return logits
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        if self.hparams.use_auxiliary_loss:
            logits, features = self(x, return_features=True)
            main_loss = self.criterion(logits, y)
            aux_loss = self.criterion(features['aux_logits'], y)
            loss = main_loss + self.hparams.auxiliary_loss_weight * aux_loss
            
            # Log losses
            self.log('train_main_loss', main_loss)
            self.log('train_aux_loss', aux_loss)
        else:
            logits = self(x)
            loss = self.criterion(logits, y)
        
        # Calculate F1 score
        f1 = self.f1_macro(logits, y)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_f1_macro', f1)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        if self.hparams.use_auxiliary_loss:
            logits, features = self(x, return_features=True)
            main_loss = self.criterion(logits, y)
            aux_loss = self.criterion(features['aux_logits'], y)
            loss = main_loss + self.hparams.auxiliary_loss_weight * aux_loss
            
            # Log losses
            self.log('val_main_loss', main_loss)
            self.log('val_aux_loss', aux_loss)
        else:
            logits = self(x)
            loss = self.criterion(logits, y)
        
        # Calculate F1 score
        f1 = self.f1_macro(logits, y)
        
        # Log metrics
        self.log('val_loss', loss)
        self.log('val_f1_macro', f1, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        f1 = self.f1_macro(logits, y)
        
        self.log('test_loss', loss)
        self.log('test_f1_macro', f1)
        
        return loss
    
    def configure_optimizers(self):
        # Select optimizer
        if self.hparams.optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), 
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), 
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.hparams.optimizer_type}")
        
        # Configure scheduler if specified
        if self.hparams.scheduler_type is None:
            return optimizer
        
        scheduler_params = self.hparams.scheduler_params or {}
        
        if self.hparams.scheduler_type == "onecycle":
            # Filter out only valid OneCycleLR parameters
            valid_onecycle_params = ['epochs', 'steps_per_epoch', 'pct_start', 'anneal_strategy', 
                                     'cycle_momentum', 'base_momentum', 'max_momentum', 'div_factor', 
                                     'final_div_factor', 'three_phase']
            onecycle_kwargs = {k: v for k, v in scheduler_params.items() 
                               if k in valid_onecycle_params and k not in ['epochs', 'steps_per_epoch']}
            
            # Calculate total steps with a small buffer to avoid stepping beyond limit
            epochs = scheduler_params.get('epochs', 100)
            steps_per_epoch = scheduler_params.get('steps_per_epoch', 100)
            total_steps = epochs * steps_per_epoch
            
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.learning_rate,
                total_steps=total_steps + 1,  # Add 1 to avoid the exact steps issue
                **onecycle_kwargs
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step'
                }
            }
        elif self.hparams.scheduler_type == "cosine":
            # Filter out only valid CosineAnnealingLR parameters
            valid_cosine_params = ['T_max', 'eta_min', 'last_epoch', 'verbose']
            cosine_kwargs = {k: v for k, v in scheduler_params.items() 
                             if k in valid_cosine_params and k != 'T_max'}
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_params.get('T_max', 100),
                **cosine_kwargs
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler
            }
        elif self.hparams.scheduler_type == "reduce":
            # Filter out only valid ReduceLROnPlateau parameters
            valid_reduce_params = ['mode', 'factor', 'patience', 'threshold', 'threshold_mode', 
                                   'cooldown', 'min_lr', 'eps', 'verbose']
            reduce_kwargs = {k: v for k, v in scheduler_params.items() 
                             if k in valid_reduce_params and k not in ['factor', 'patience']}
            
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=scheduler_params.get('factor', 0.5),
                patience=scheduler_params.get('patience', 10),
                **reduce_kwargs
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_f1_macro'
                }
            }
        else:
            raise ValueError(f"Unknown scheduler type: {self.hparams.scheduler_type}")