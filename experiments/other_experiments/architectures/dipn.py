"""
Domain-Invariant Phoneme Network (DIPN)
Designed for robustness to domain shift between train and holdout sets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import F1Score, Accuracy
import numpy as np
from typing import Optional, Dict, Tuple


class DomainInvariantPhonemeNet(L.LightningModule):
    """
    Architecture designed to handle domain shift between train and holdout.
    Key principles:
    1. Feature bottlenecks to prevent overfitting
    2. Multiple shallow paths instead of deep networks
    3. Ensemble-like predictions from different views
    4. Statistical features that should be domain-invariant
    """
    
    def __init__(
        self,
        # Data params
        meg_channels=306,
        time_points=125,
        num_classes=39,
        
        # Architecture params
        base_dim=64,  # Deliberately small to prevent overfitting
        num_paths=3,  # Number of parallel processing paths
        conformer_heads=4,
        
        # Training params
        learning_rate=1e-4,
        weight_decay=0.1,  # High regularization
        label_smoothing=0.1,
        
        # Regularization
        dropout_rate=0.3,
        feature_dropout=0.5,
        mixup_alpha=0.2,
        
        # Ensemble params
        use_uncertainty=True,
        uncertainty_weight=0.1,
        temperature=2.0,  # For softmax temperature scaling
        
        # Scheduler params
        use_scheduler='cosine_warmup',
        warmup_epochs=3,
        max_epochs=50,
        
        **kwargs  # Catch any extra params from config
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # === Path 1: Temporal Focus (captures time dynamics) ===
        self.temporal_conv = nn.Sequential(
            # Depthwise separable convolution for efficiency
            nn.Conv1d(meg_channels, meg_channels, kernel_size=15, padding=7, groups=meg_channels),
            nn.BatchNorm1d(meg_channels),
            nn.Conv1d(meg_channels, base_dim, kernel_size=1),  # Pointwise
            nn.GELU(),
            nn.Dropout1d(dropout_rate)
        )
        
        # === Path 2: Spatial Focus (channel relationships) ===
        self.spatial_conv = nn.Sequential(
            # Process across channels (after transpose)
            nn.Conv1d(time_points, base_dim, kernel_size=1),
            nn.BatchNorm1d(base_dim),
            nn.GELU(),
            nn.Dropout1d(dropout_rate)
        )
        
        # === Path 3: Raw Statistics (domain-invariant features) ===
        # 6 statistics per channel = meg_channels * 6 input features
        self.stat_extractor = nn.Sequential(
            nn.Linear(meg_channels * 6, base_dim * 2),
            nn.BatchNorm1d(base_dim * 2),
            nn.GELU(),
            nn.Dropout(feature_dropout),  # Heavy dropout on statistics
            nn.Linear(base_dim * 2, base_dim)
        )
        
        # === Conformer Block (inspired by LCS-CTC success) ===
        self.conformer = self._make_simple_conformer(base_dim * num_paths, conformer_heads)
        
        # === Multi-Head Classifiers (ensemble effect) ===
        classifier_dim = base_dim * num_paths
        self.head1 = nn.Linear(classifier_dim, num_classes)
        self.head2 = nn.Linear(classifier_dim, num_classes)
        self.head3 = nn.Linear(classifier_dim, num_classes)
        
        # Initialize heads differently for diversity
        nn.init.xavier_uniform_(self.head1.weight, gain=1.0)
        nn.init.xavier_uniform_(self.head2.weight, gain=0.5)
        nn.init.xavier_uniform_(self.head3.weight, gain=2.0)
        
        # === Uncertainty Estimation ===
        if use_uncertainty:
            self.uncertainty = nn.Sequential(
                nn.Linear(classifier_dim, base_dim),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(base_dim, 1),
                nn.Sigmoid()
            )
        
        # === Loss and Metrics ===
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.train_f1 = F1Score(num_classes=num_classes, average='macro', task='multiclass')
        self.val_f1 = F1Score(num_classes=num_classes, average='macro', task='multiclass')
        self.train_acc = Accuracy(num_classes=num_classes, task='multiclass')
        self.val_acc = Accuracy(num_classes=num_classes, task='multiclass')
        
        # For tracking
        self.validation_step_outputs = []
        
    def _make_simple_conformer(self, dim, num_heads):
        """Single conformer block similar to LCS-CTC"""
        return nn.ModuleDict({
            'mha': nn.MultiheadAttention(dim, num_heads, dropout=0.1, batch_first=True),
            'conv': nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim//num_heads),
                nn.BatchNorm1d(dim),
                nn.Conv1d(dim, dim, kernel_size=1),
                nn.GELU()
            ),
            'ffn': nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(dim * 2, dim),
                nn.Dropout(0.1)
            ),
            'ln1': nn.LayerNorm(dim),
            'ln2': nn.LayerNorm(dim),
            'ln3': nn.LayerNorm(dim)
        })
    
    def extract_robust_stats(self, x):
        """Extract statistics that should be domain-invariant"""
        # x: (B, C, T)
        stats = []
        
        # Basic statistics
        stats.append(x.mean(dim=2))  # Channel means
        stats.append(x.std(dim=2).clamp(min=1e-6))   # Channel stds (clamped for stability)
        stats.append(x.median(dim=2)[0])  # Channel medians
        
        # Quantiles (robust to outliers)
        stats.append(torch.quantile(x, 0.25, dim=2))  # Q1
        stats.append(torch.quantile(x, 0.75, dim=2))  # Q3
        
        # Differential features (relative changes, not absolute values)
        if x.size(2) > 1:
            diff = x[:, :, 1:] - x[:, :, :-1]
            stats.append(diff.std(dim=2).clamp(min=1e-6))  # Temporal variability
        else:
            stats.append(torch.zeros_like(stats[0]))
        
        return torch.cat(stats, dim=1)  # (B, C*6)
    
    def forward_conformer(self, x, conformer):
        """Apply conformer block with residual connections"""
        # x: (B, 1, dim) or (B, T, dim)
        
        # Self-attention
        res = x
        attn_out, _ = conformer['mha'](x, x, x)
        x = conformer['ln1'](attn_out + res)
        
        # Convolution (need to transpose for Conv1d)
        res = x
        x_conv = x.transpose(1, 2)  # (B, dim, T)
        x_conv = conformer['conv'](x_conv).transpose(1, 2)  # Back to (B, T, dim)
        x = conformer['ln2'](x_conv + res)
        
        # Feed-forward
        res = x
        x = conformer['ln3'](conformer['ffn'](x) + res)
        
        return x
    
    def forward(self, x, return_uncertainty=False):
        B, C, T = x.shape
        
        # === Path 1: Temporal Features ===
        temporal_feat = self.temporal_conv(x)  # (B, base_dim, T)
        temporal_pool = F.adaptive_avg_pool1d(temporal_feat, 1).squeeze(-1)  # (B, base_dim)
        
        # === Path 2: Spatial Features ===
        x_transposed = x.transpose(1, 2)  # (B, T, C)
        spatial_feat = self.spatial_conv(x_transposed)  # (B, base_dim, C)
        spatial_pool = F.adaptive_avg_pool1d(spatial_feat, 1).squeeze(-1)  # (B, base_dim)
        
        # === Path 3: Statistical Features ===
        stats = self.extract_robust_stats(x)  # (B, C*6)
        stat_feat = self.stat_extractor(stats)  # (B, base_dim)
        
        # === Combine all paths ===
        combined = torch.cat([temporal_pool, spatial_pool, stat_feat], dim=1)  # (B, base_dim*3)
        
        # Add sequence dimension for conformer
        combined = combined.unsqueeze(1)  # (B, 1, base_dim*3)
        
        # === Apply Conformer ===
        combined = self.forward_conformer(combined, self.conformer)
        combined = combined.squeeze(1)  # (B, base_dim*3)
        
        # === Multi-head predictions ===
        logits1 = self.head1(combined) / self.hparams.temperature
        logits2 = self.head2(combined) / self.hparams.temperature
        logits3 = self.head3(combined) / self.hparams.temperature
        
        # === Ensemble with uncertainty weighting ===
        if self.hparams.use_uncertainty and hasattr(self, 'uncertainty'):
            uncertainty = self.uncertainty(combined)  # (B, 1)
            
            # Dynamic weighting based on uncertainty
            w1 = (1 - uncertainty) * 0.5 + 0.25  # Range [0.25, 0.75]
            w2 = (1 - w1) / 2  # Share remaining weight
            w3 = (1 - w1) / 2
            
            logits = w1 * logits1 + w2 * logits2 + w3 * logits3
            
            if return_uncertainty:
                return logits, uncertainty.squeeze()
        else:
            # Simple averaging
            logits = (logits1 + logits2 + logits3) / 3
            
            if return_uncertainty:
                return logits, torch.zeros(B, device=x.device)
        
        return logits
    
    def apply_mixup(self, x, y):
        """Apply mixup in feature space"""
        if not self.training or self.hparams.mixup_alpha <= 0:
            return x, y, 1.0
        
        B = x.size(0)
        lam = np.random.beta(self.hparams.mixup_alpha, self.hparams.mixup_alpha)
        index = torch.randperm(B, device=x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        
        return mixed_x, (y_a, y_b, lam), lam
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # Apply mixup
        x, y_mixed, lam = self.apply_mixup(x, y)
        
        if self.hparams.use_uncertainty:
            logits, uncertainty = self(x, return_uncertainty=True)
        else:
            logits = self(x, return_uncertainty=False)
            uncertainty = torch.zeros(x.size(0), device=x.device)
        
        # Compute loss
        if isinstance(y_mixed, tuple):
            y_a, y_b, lam = y_mixed
            loss = lam * self.ce_loss(logits, y_a) + (1 - lam) * self.ce_loss(logits, y_b)
            # Use original labels for metrics
            y_for_metrics = y_a
        else:
            loss = self.ce_loss(logits, y)
            y_for_metrics = y
        
        # Add uncertainty regularization if enabled
        if self.hparams.use_uncertainty:
            with torch.no_grad():
                correct = logits.argmax(dim=1) == y_for_metrics
            
            # Low uncertainty when correct, high when wrong
            uncertainty_loss = torch.mean(
                correct.float() * uncertainty + 
                (1 - correct.float()) * (1 - uncertainty)
            )
            loss = loss + self.hparams.uncertainty_weight * uncertainty_loss
            
            self.log('train/uncertainty_loss', uncertainty_loss, prog_bar=False)
            self.log('train/uncertainty_mean', uncertainty.mean(), prog_bar=False)
        
        # Metrics
        f1 = self.train_f1(logits, y_for_metrics)
        acc = self.train_acc(logits, y_for_metrics)
        
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_f1_macro', f1, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x, return_uncertainty=False)
        loss = self.ce_loss(logits, y)
        
        f1 = self.val_f1(logits, y)
        acc = self.val_acc(logits, y)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_f1_macro', f1, prog_bar=True)
        self.log('val_acc', acc)
        
        # Store for epoch-end analysis
        self.validation_step_outputs.append({
            'loss': loss,
            'logits': logits.detach(),
            'labels': y.detach()
        })
        
        return loss
    
    def on_validation_epoch_end(self):
        # Additional epoch-level metrics if needed
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        # Different learning rates for different components
        param_groups = [
            {'params': self.temporal_conv.parameters(), 'lr': self.hparams.learning_rate},
            {'params': self.spatial_conv.parameters(), 'lr': self.hparams.learning_rate},
            {'params': self.stat_extractor.parameters(), 'lr': self.hparams.learning_rate * 0.1},
            {'params': self.conformer.parameters(), 'lr': self.hparams.learning_rate},
            {'params': self.head1.parameters(), 'lr': self.hparams.learning_rate * 2},
            {'params': self.head2.parameters(), 'lr': self.hparams.learning_rate * 2},
            {'params': self.head3.parameters(), 'lr': self.hparams.learning_rate * 2},
        ]
        
        if self.hparams.use_uncertainty and hasattr(self, 'uncertainty'):
            param_groups.append({
                'params': self.uncertainty.parameters(), 
                'lr': self.hparams.learning_rate * 0.5
            })
        
        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.hparams.weight_decay)
        
        # Scheduler
        if self.hparams.use_scheduler == 'cosine_warmup':
            # Cosine annealing with warm restarts
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=1e-6
            )
        elif self.hparams.use_scheduler == 'onecycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.learning_rate,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.1
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6
            )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step' if self.hparams.use_scheduler == 'onecycle' else 'epoch'
            }
        }