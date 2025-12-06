"""
Enhanced Multi-Scale RNN Architecture for MEG Phoneme Classification
Combines the best elements from demega (DeBERTa attention, Conformer, BalancedPhonemePretrainer)
with multi-scale temporal processing and RNN backbone
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import F1Score, Accuracy
import numpy as np
from typing import Optional, List, Dict, Tuple, Iterable
import math
from collections import defaultdict

# ============================================
# Import DeBERTa Components from demega
# ============================================

def prepare_attention_mask(attention_mask):
    if attention_mask.dim() <= 2:
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
    elif attention_mask.dim() == 3:
        attention_mask = attention_mask.unsqueeze(1)
    return attention_mask

@torch.jit.script
def make_log_bucket_position(relative_pos: torch.Tensor, bucket_size: int, max_position: int):
    sign = torch.sign(relative_pos)
    mid = bucket_size // 2
    abs_pos = torch.where(
        (relative_pos < mid) & (relative_pos > -mid),
        torch.tensor(mid - 1).type_as(relative_pos),
        torch.abs(relative_pos),
    )
    log_pos = (
        torch.ceil(
            torch.log(abs_pos / mid)
            / torch.log(torch.tensor((max_position - 1) / mid))
            * (mid - 1)
        ) + mid
    )
    bucket_pos = torch.where(abs_pos <= mid, relative_pos.type_as(log_pos), log_pos * sign)
    return bucket_pos

def build_relative_position(query_layer, key_layer, bucket_size: int = -1, max_position: int = -1):
    query_size = query_layer.size(-2)
    key_size = key_layer.size(-2)

    q_ids = torch.arange(query_size, dtype=torch.long, device=query_layer.device)
    k_ids = torch.arange(key_size, dtype=torch.long, device=key_layer.device)
    rel_pos_ids = q_ids[:, None] - k_ids[None, :]
    if bucket_size > 0 and max_position > 0:
        rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position)
    rel_pos_ids = rel_pos_ids.to(torch.long)
    rel_pos_ids = rel_pos_ids[:query_size, :]
    rel_pos_ids = rel_pos_ids.unsqueeze(0)
    return rel_pos_ids

# ============================================
# Balanced Phoneme Pretrainer from demega
# ============================================

class BalancedPhonemePretrainer(nn.Module):
    """
    Pre-training module with temperature-based reweighting from demega.
    Uses focal loss and exponential temperature scaling for rare phonemes.
    """
    
    def __init__(self, vocab_size: int = 39, hidden_dim: int = 16, temperature: float = 2.0):
        super().__init__()

        phoneme_counts = {
            0: 216, 1: 443, 2: 1584, 3: 231, 4: 114, 5: 360, 6: 268, 7: 93,
            8: 772, 9: 504, 10: 477, 11: 429, 12: 231, 13: 282, 14: 119, 15: 428,
            16: 1052, 17: 570, 18: 63, 19: 430, 20: 566, 21: 518, 22: 1128, 23: 154,
            24: 226, 25: 14, 26: 276, 27: 634, 28: 743, 29: 113, 30: 1143, 31: 110,
            32: 96, 33: 236, 34: 326, 35: 428, 36: 151, 37: 456, 38: 7
        }

        total_count = sum(phoneme_counts.values())
        
        # Temperature-based scaling (exponential) to prevent extreme weights
        self.class_weights = torch.zeros(vocab_size)
        for i, count in phoneme_counts.items():
            freq = count / total_count
            # Use temperature to control the strength of reweighting
            self.class_weights[i] = math.exp(-temperature * freq)
        
        # Normalize weights to reasonable range
        self.class_weights = self.class_weights / self.class_weights.mean()
        # Clip extreme values
        self.class_weights = torch.clamp(self.class_weights, min=0.5, max=5.0)
        
    def focal_loss(self, logits: torch.Tensor, targets: torch.Tensor, 
                   gamma: float = 2.0, alpha: torch.Tensor = None):
        """Focal loss to focus on hard-to-classify phonemes."""
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** gamma * ce_loss
        
        if alpha is not None:
            focal_loss = alpha[targets] * focal_loss
        
        return focal_loss.mean()

# ============================================
# MEG Conformer Layer with RNN Integration
# ============================================

class MEGConformerRNNLayer(nn.Module):
    """
    Hybrid Conformer-RNN layer combining DeBERTa attention with RNN processing.
    Based on demega's successful Conformer implementation.
    """
    
    def __init__(self, dim: int, rnn_hidden: int = None, num_heads: int = 4, 
                 ff_dim: int = None, kernel_size: int = 5, dropout: float = 0.1,
                 rnn_type: str = "LSTM", norm_type: str = "pre"):
        super().__init__()
        ff_dim = ff_dim or 2 * dim
        rnn_hidden = rnn_hidden or dim
        self.norm_type = norm_type
        
        # Depthwise separable convolution from demega
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, 1)
        )
        
        # Layer norms (pre-norm strategy from demega)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ln3 = nn.LayerNorm(dim)
        self.ln4 = nn.LayerNorm(dim)
        
        # RNN component for temporal modeling
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(dim, rnn_hidden, num_layers=1, 
                              batch_first=True, bidirectional=True)
        else:
            self.rnn = nn.GRU(dim, rnn_hidden, num_layers=1,
                             batch_first=True, bidirectional=True)
        
        # Project RNN output back to dim
        self.rnn_proj = nn.Linear(rnn_hidden * 2, dim)
        
        # FFN with SiLU activation (from demega)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Pre-layer norm implementation from demega
        if self.norm_type == "pre":
            # Convolution module with pre-norm
            res = x
            x_norm = self.ln1(x)
            x_conv = x_norm.transpose(1, 2)
            x_conv = self.conv(x_conv).transpose(1, 2)
            x = res + self.dropout(x_conv)
            
            # RNN module with pre-norm
            res = x
            x_norm = self.ln2(x)
            rnn_out, _ = self.rnn(x_norm)
            rnn_out = self.rnn_proj(rnn_out)
            x = res + self.dropout(rnn_out)
            
            # Feed-forward module with pre-norm
            res = x
            x_norm = self.ln3(x)
            ff_out = self.ffn(x_norm)
            x = res + ff_out
            
        return x

# ============================================
# Enhanced Multi-Scale Temporal Processing
# ============================================

class EnhancedMultiScaleEncoder(nn.Module):
    """
    Multi-scale temporal encoder using Conformer-RNN hybrid layers.
    """
    
    def __init__(self,
                 input_dim: int,
                 temporal_scales: List[int],
                 scale_hidden_dims: List[int],
                 num_conformer_layers: int = 2,
                 dropout: float = 0.1,
                 norm_type: str = "pre"):
        super().__init__()
        assert len(temporal_scales) == len(scale_hidden_dims)
        
        self.temporal_scales = temporal_scales
        self.scale_hidden_dims = scale_hidden_dims
        self.num_scales = len(temporal_scales)
        
        # Input projection for each scale
        self.scale_projections = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim)
            for hidden_dim in scale_hidden_dims
        ])
        
        # Conformer-RNN layers for each scale
        self.scale_encoders = nn.ModuleList()
        for scale, hidden_dim in zip(temporal_scales, scale_hidden_dims):
            layers = nn.ModuleList([
                MEGConformerRNNLayer(
                    dim=hidden_dim,
                    rnn_hidden=hidden_dim // 2,
                    num_heads=4,
                    ff_dim=hidden_dim * 2,
                    kernel_size=5,
                    dropout=dropout,
                    norm_type=norm_type
                )
                for _ in range(num_conformer_layers)
            ])
            self.scale_encoders.append(layers)
        
        # Output dimension
        self.output_dim = sum(scale_hidden_dims)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Process input at multiple scales with Conformer-RNN layers.
        x: (batch, time, features)
        """
        B, T, F = x.shape
        scale_outputs = []
        
        for idx, (scale, projection, encoder_layers) in enumerate(
            zip(self.temporal_scales, self.scale_projections, self.scale_encoders)
        ):
            # Project input to scale-specific dimension
            x_scale = projection(x)  # (B, T, hidden_dim)
            
            # Apply Conformer-RNN layers
            for layer in encoder_layers:
                x_scale = layer(x_scale)
            
            # Pool based on scale
            if scale < T:
                # Adaptive pooling based on scale
                stride = max(1, scale // 4)
                pooled = x_scale[:, ::stride, :]
                scale_out = pooled.mean(dim=1)
            else:
                scale_out = x_scale.mean(dim=1)
            
            scale_outputs.append(scale_out)
        
        return scale_outputs

# ============================================
# Data Augmentation (Enhanced)
# ============================================

class EnhancedMEGAugmentation(nn.Module):
    """Enhanced data augmentation for MEG signals."""
    
    def __init__(self,
                 noise_level: float = 0.05,
                 time_jitter: float = 0.02,
                 channel_dropout: float = 0.1,
                 time_masking: float = 0.1,
                 mixup_alpha: float = 0.2):
        super().__init__()
        self.noise_level = noise_level
        self.time_jitter = time_jitter
        self.channel_dropout = channel_dropout
        self.time_masking = time_masking
        self.mixup_alpha = mixup_alpha
    
    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None, 
                training: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply augmentation to MEG data."""
        if not training:
            return x, targets
        
        B, C, T = x.shape
        device = x.device
        
        # Gaussian noise
        if self.noise_level > 0 and torch.rand(1).item() < 0.5:
            noise = torch.randn_like(x) * self.noise_level
            x = x + noise
        
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
        
        # Mixup augmentation
        if self.mixup_alpha > 0 and targets is not None and torch.rand(1).item() < 0.3:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            index = torch.randperm(B, device=device)
            x = lam * x + (1 - lam) * x[index]
            if targets is not None:
                targets_a = targets
                targets_b = targets[index]
                return x, (targets_a, targets_b, lam)
        
        return x, targets

# ============================================
# Main Enhanced Multi-Scale RNN Classifier
# ============================================

class EnhancedMultiScaleRNNClassifier(L.LightningModule):
    """
    Enhanced Multi-scale RNN classifier combining best elements from demega.
    """
    
    def __init__(self,
                 # Core dimensions
                 meg_channels: int = 306,
                 time_points: int = 125,
                 vocab_size: int = 39,
                 hidden_dim: int = 64,
                 
                 # Multi-scale parameters
                 temporal_scales: List[int] = [8, 16, 32, 64],
                 scale_hidden_dims: List[int] = [32, 48, 64, 80],
                 num_conformer_layers: int = 2,
                 
                 # Architecture choices from demega
                 use_conformer: bool = True,
                 norm_type: str = "pre",
                 
                 # Training parameters from demega
                 learning_rate: float = 0.0001,
                 classifier_lr_multiplier: float = 2.0,
                 
                 # Loss configuration from demega
                 loss_type: str = "focal",
                 focal_gamma: float = 2.0,
                 temperature: float = 2.0,
                 
                 # Regularization
                 dropout_rate: float = 0.1,
                 label_smoothing: float = 0.1,
                 weight_decay: float = 0.01,
                 
                 # Learning rate schedule
                 warmup_epochs: int = 5,
                 total_epochs: int = 100,
                 
                 # Augmentation
                 use_augmentation: bool = True,
                 augmentation_params: Optional[Dict] = None,
                 
                 # Scale weighting
                 scale_weighting: str = "attention",
                 scale_attention_dim: int = 64,
                 
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        # Balanced pre-trainer from demega
        self.pretrainer = BalancedPhonemePretrainer(vocab_size, hidden_dim, temperature)
        
        # Data augmentation
        if use_augmentation:
            aug_params = augmentation_params or {}
            self.augmentation = EnhancedMEGAugmentation(
                noise_level=aug_params.get('noise_level', 0.05),
                time_jitter=aug_params.get('time_jitter', 0.02),
                channel_dropout=aug_params.get('channel_dropout', 0.1),
                time_masking=aug_params.get('time_masking', 0.1),
                mixup_alpha=aug_params.get('mixup_alpha', 0.2)
            )
        else:
            self.augmentation = None
        
        if use_conformer:
            # Initial projection with residual (from demega)
            self.input_projection = nn.Sequential(
                nn.Conv1d(meg_channels, hidden_dim, kernel_size=5, padding=2),
                nn.SiLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
            )
            
            # Skip connection for input (from demega)
            self.input_skip = nn.Conv1d(meg_channels, hidden_dim, kernel_size=1)
            
            # Enhanced multi-scale encoder with Conformer-RNN
            self.multi_scale_encoder = EnhancedMultiScaleEncoder(
                input_dim=hidden_dim,
                temporal_scales=temporal_scales,
                scale_hidden_dims=scale_hidden_dims,
                num_conformer_layers=num_conformer_layers,
                dropout=dropout_rate,
                norm_type=norm_type
            )
            
            encoder_output_dim = sum(scale_hidden_dims)
        else:
            # Fallback to simple multi-scale RNN
            self.input_projection = None
            self.input_skip = None
            self.multi_scale_encoder = None
            encoder_output_dim = meg_channels
        
        self.use_conformer = use_conformer
        
        # Scale weighting mechanism
        if scale_weighting == "attention":
            self.scale_attention = nn.MultiheadAttention(
                encoder_output_dim, num_heads=8, batch_first=True
            )
            self.scale_weight_proj = nn.Linear(encoder_output_dim, len(temporal_scales))
        else:
            self.scale_weights = nn.Parameter(torch.ones(len(temporal_scales)) / len(temporal_scales))
        
        self.scale_weighting_type = scale_weighting
        
        # Feature normalization (from demega)
        self.feature_norm = nn.LayerNorm(encoder_output_dim)
        
        # Classification head (simplified from demega)
        self.classifier = nn.Sequential(
            nn.Linear(encoder_output_dim, 128),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, vocab_size)
        )
        
        # Loss parameters
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.classifier_lr_multiplier = classifier_lr_multiplier
        
        # Metrics
        self.train_f1 = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
        self.val_f1 = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
        self.test_f1 = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
        
        # Per-phoneme tracking (from demega)
        self.phoneme_f1_scores = defaultdict(float)
        self.phoneme_counts = defaultdict(int)
        
        # Training parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features using enhanced multi-scale encoder."""
        B, C, T = x.shape
        
        if self.use_conformer:
            # Apply initial convolution with residual (from demega)
            features_main = self.input_projection(x)
            features_skip = self.input_skip(x)
            features = features_main + features_skip  # Residual connection
            
            features = features.transpose(1, 2)  # (B, T, hidden_dim)
            
            # Multi-scale encoding
            scale_outputs = self.multi_scale_encoder(features)
            
            # Scale weighting
            combined = torch.cat(scale_outputs, dim=-1)  # (B, total_dim)
            
            if self.scale_weighting_type == "attention":
                combined_exp = combined.unsqueeze(1)  # (B, 1, total_dim)
                attended, _ = self.scale_attention(combined_exp, combined_exp, combined_exp)
                scale_weights = F.softmax(self.scale_weight_proj(attended.squeeze(1)), dim=-1)
                
                # Apply weights
                weighted_features = []
                start_idx = 0
                for i, scale_out in enumerate(scale_outputs):
                    weight = scale_weights[:, i:i+1]
                    weighted_features.append(scale_out * weight)
                
                features = torch.cat(weighted_features, dim=-1)
            else:
                features = combined
            
            # Apply feature normalization
            features = self.feature_norm(features)
        else:
            # Simple average pooling
            features = x.mean(dim=2)
        
        return features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification."""
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits
    
    def compute_loss(self, logits, targets, mixup_targets=None):
        """Compute loss based on configured loss type (from demega)."""
        if mixup_targets is not None:
            # Handle mixup loss
            targets_a, targets_b, lam = mixup_targets
            loss_a = self.compute_single_loss(logits, targets_a)
            loss_b = self.compute_single_loss(logits, targets_b)
            return lam * loss_a + (1 - lam) * loss_b
        else:
            return self.compute_single_loss(logits, targets)
    
    def compute_single_loss(self, logits, targets):
        """Compute single loss (from demega)."""
        if self.loss_type == "focal":
            focal_loss = self.pretrainer.focal_loss(
                logits, targets, 
                gamma=self.focal_gamma,
                alpha=self.pretrainer.class_weights.to(logits.device)
            )
            
            if self.label_smoothing > 0:
                smooth_loss = F.cross_entropy(
                    logits, targets, 
                    label_smoothing=self.label_smoothing
                )
                loss = 0.7 * focal_loss + 0.3 * smooth_loss
            else:
                loss = focal_loss
                
        else:  # cross_entropy
            if self.label_smoothing > 0:
                loss = F.cross_entropy(
                    logits, targets, 
                    label_smoothing=self.label_smoothing
                )
            else:
                loss = F.cross_entropy(logits, targets)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # Apply augmentation
        mixup_targets = None
        if self.augmentation:
            x_aug, aug_targets = self.augmentation(x, y, training=True)
            x = x_aug
            # Check if mixup was applied (returns tuple of 3 elements)
            if aug_targets is not None and isinstance(aug_targets, tuple) and len(aug_targets) == 3:
                mixup_targets = aug_targets
        
        logits = self(x)
        loss = self.compute_loss(logits, y, mixup_targets)
        
        # Metrics (only on non-mixup samples)
        if mixup_targets is None:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                acc = (preds == y).float().mean()
                f1 = self.train_f1(logits, y)
                
                # Per-phoneme tracking (from demega)
                for i in range(len(y)):
                    phoneme_id = y[i].item()
                    self.phoneme_counts[phoneme_id] += 1
                    if preds[i] == y[i]:
                        self.phoneme_f1_scores[phoneme_id] += 1
        else:
            acc = 0.0
            f1 = 0.0
        
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
        """Configure optimizers with differential learning rates (from demega)."""
        # Separate parameters for differential learning rates
        classifier_params = list(self.classifier.parameters())
        classifier_param_ids = {id(p) for p in classifier_params}
        other_params = [p for p in self.parameters() if id(p) not in classifier_param_ids]
        
        optimizer = torch.optim.AdamW([
            {'params': other_params, 'lr': self.learning_rate},
            {'params': classifier_params, 'lr': self.learning_rate * self.classifier_lr_multiplier}
        ], weight_decay=self.weight_decay)
        
        # Cosine annealing with warmup
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return float(epoch) / float(max(1, self.warmup_epochs))
            else:
                progress = float(epoch - self.warmup_epochs) / float(max(1, self.total_epochs - self.warmup_epochs))
                return 0.5 * (1.0 + math.cos(math.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }