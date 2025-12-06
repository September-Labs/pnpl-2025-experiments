# hierarchical_dynamic.py
"""
Hierarchical Dynamic Network for MEG Phoneme Classification.

Inspired by "From Thought to Action: How a Hierarchy of Neural Dynamics Supports Language Production"
This architecture implements:
1. Multi-scale temporal processing (different linguistic levels have different dynamics)
2. Dynamic neural codes (representations that evolve over time)
3. Temporal attention for overlapping representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import F1Score
import math


class DynamicConv1d(nn.Module):
    """Dynamic convolution that adapts weights based on input."""
    
    def __init__(self, in_channels, out_channels, kernel_size, reduction=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Static convolution baseline
        self.static_conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        
        # Dynamic weight generation
        self.weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, in_channels // reduction, 1),
            nn.ReLU(),
            nn.Conv1d(in_channels // reduction, out_channels * kernel_size, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch_size, _, time_points = x.shape
        
        # Generate dynamic weights
        weights = self.weight_gen(x)  # (B, out_channels * kernel_size, 1)
        weights = weights.view(batch_size, self.out_channels, self.kernel_size, 1)
        
        # Apply static convolution
        static_out = self.static_conv(x)
        
        # Apply dynamic modulation
        # Unfold input for dynamic conv
        x_unfold = F.unfold(x.unsqueeze(2), (self.kernel_size, 1), padding=(self.kernel_size//2, 0))
        x_unfold = x_unfold.view(batch_size, self.in_channels, self.kernel_size, time_points)
        
        # Apply dynamic weights (simplified version)
        dynamic_mod = (weights * x_unfold.mean(dim=1, keepdim=True)).sum(dim=2)
        
        return static_out + dynamic_mod


class TemporalScaleBlock(nn.Module):
    """Process MEG signals at a specific temporal scale."""
    
    def __init__(self, in_channels, out_channels, scale_factor, kernel_size, use_dynamic=True, reduction=4):
        super().__init__()
        self.scale_factor = scale_factor
        
        # Temporal pooling for this scale
        if scale_factor > 1:
            self.pool = nn.AvgPool1d(scale_factor, stride=scale_factor)
        else:
            self.pool = nn.Identity()
        
        # Dynamic or static convolution
        if use_dynamic:
            self.conv = DynamicConv1d(in_channels, out_channels, kernel_size, reduction)
        else:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # Apply temporal scaling
        x = self.pool(x)
        
        # Apply convolution
        x = self.conv(x)
        
        # Normalize and activate
        x = x.transpose(1, 2)  # (B, T, C) for LayerNorm
        x = self.norm(x)
        x = x.transpose(1, 2)  # Back to (B, C, T)
        
        x = self.activation(x)
        x = self.dropout(x)
        
        return x


class HierarchicalTemporalEncoder(nn.Module):
    """Encode MEG signals at multiple temporal scales."""
    
    def __init__(self, n_channels, temporal_scales, scale_dims, dynamic_kernel_sizes, 
                 use_dynamic_conv=True, dynamic_reduction=4):
        super().__init__()
        
        self.scales = nn.ModuleList()
        for scale, dim, kernel in zip(temporal_scales, scale_dims, dynamic_kernel_sizes):
            self.scales.append(
                TemporalScaleBlock(
                    n_channels, dim, scale, kernel,
                    use_dynamic=use_dynamic_conv,
                    reduction=dynamic_reduction
                )
            )
        
        # Fusion layer
        self.fusion = nn.Conv1d(sum(scale_dims), scale_dims[-1], 1)
        self.fusion_norm = nn.LayerNorm(scale_dims[-1])
        
    def forward(self, x):
        # Process at each scale
        multi_scale_features = []
        for scale_block in self.scales:
            scale_features = scale_block(x)
            # Upsample to original time dimension if needed
            if scale_features.shape[-1] != x.shape[-1]:
                scale_features = F.interpolate(
                    scale_features, size=x.shape[-1], mode='linear', align_corners=False
                )
            multi_scale_features.append(scale_features)
        
        # Concatenate and fuse
        fused = torch.cat(multi_scale_features, dim=1)
        fused = self.fusion(fused)
        
        # Normalize
        fused = fused.transpose(1, 2)
        fused = self.fusion_norm(fused)
        fused = fused.transpose(1, 2)
        
        return fused


class TemporalAttention(nn.Module):
    """Multi-head attention for temporal dependencies."""
    
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class DynamicLayer(nn.Module):
    """Layer with dynamic neural codes and temporal attention."""
    
    def __init__(self, dim, num_heads=8, dropout=0.1, use_attention=True):
        super().__init__()
        
        # Temporal attention
        self.use_attention = use_attention
        if use_attention:
            self.attention = TemporalAttention(dim, num_heads, dropout)
            self.norm1 = nn.LayerNorm(dim)
        
        # Feedforward with dynamic modulation
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
        # Dynamic gate (inspired by the paper's dynamic codes)
        self.dynamic_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: (B, C, T) -> convert to (B, T, C) for attention
        x = x.transpose(1, 2)
        
        # Attention with residual
        if self.use_attention:
            x = x + self.attention(self.norm1(x))
        
        # Feedforward with dynamic gating
        ff_out = self.ff(x)
        gate = self.dynamic_gate(x)
        x = x + ff_out * gate
        
        # Convert back to (B, C, T)
        x = x.transpose(1, 2)
        
        return x


class HierarchicalDynamicNetwork(L.LightningModule):
    """
    Hierarchical Dynamic Network for MEG phoneme classification.
    
    Implements multi-scale temporal processing with dynamic neural codes,
    inspired by the hierarchical language production findings.
    """
    
    def __init__(
        self,
        time_points=None,
        n_channels=306,
        n_classes=39,
        learning_rate=0.0003,
        weight_decay=0.0001,
        label_smoothing=0.0,
        temporal_scales=[1, 2, 4, 8],
        scale_dims=[128, 256, 384, 512],
        dynamic_kernel_sizes=[3, 5, 7, 11],
        use_dynamic_conv=True,
        dynamic_reduction=4,
        hidden_dim=512,
        num_dynamic_layers=3,
        use_temporal_attention=True,
        attention_heads=8,
        attention_dropout=0.1,
        dropout=0.3,
        use_layer_norm=True
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Multi-scale temporal encoder
        self.encoder = HierarchicalTemporalEncoder(
            n_channels=n_channels,
            temporal_scales=temporal_scales,
            scale_dims=scale_dims,
            dynamic_kernel_sizes=dynamic_kernel_sizes,
            use_dynamic_conv=use_dynamic_conv,
            dynamic_reduction=dynamic_reduction
        )
        
        # Dynamic layers for evolving representations
        self.dynamic_layers = nn.ModuleList([
            DynamicLayer(
                dim=scale_dims[-1],
                num_heads=attention_heads,
                dropout=attention_dropout,
                use_attention=use_temporal_attention
            )
            for _ in range(num_dynamic_layers)
        ])
        
        # Global pooling strategies
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Classification head
        classifier_input_dim = scale_dims[-1] * 2  # avg + max pooling
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes)
        )
        
        # Loss and metrics
        if label_smoothing > 0:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        self.train_f1 = F1Score(num_classes=n_classes, average='macro', task='multiclass')
        self.val_f1 = F1Score(num_classes=n_classes, average='macro', task='multiclass')
        self.test_f1 = F1Score(num_classes=n_classes, average='macro', task='multiclass')
        
    def forward(self, x):
        # Multi-scale encoding
        x = self.encoder(x)
        
        # Apply dynamic layers (representations evolve through time)
        for layer in self.dynamic_layers:
            x = layer(x)
        
        # Global temporal pooling
        avg_pool = self.avg_pool(x).squeeze(-1)
        max_pool = self.max_pool(x).squeeze(-1)
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Metrics
        preds = torch.argmax(logits, dim=1)
        f1 = self.train_f1(preds, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_f1_macro', f1, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Metrics
        preds = torch.argmax(logits, dim=1)
        f1 = self.val_f1(preds, y)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1_macro', f1, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Metrics
        preds = torch.argmax(logits, dim=1)
        f1 = self.test_f1(preds, y)
        
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_f1_macro', f1, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs if self.trainer else 50,
            eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }