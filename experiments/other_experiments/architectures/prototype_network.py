# simplified_meg_classifier.py
"""
Simplified but powerful MEG phoneme classifier.
Focus on strong feature extraction and basic classification first.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np
from typing import Optional, Dict, List, Tuple
from torchmetrics import F1Score, Accuracy
import math


class PositionalEncoding(nn.Module):
    """Add positional encoding to capture temporal structure."""
    
    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        # x shape: (batch, time, features)
        return x + self.pe[:, :x.size(1), :]


class MEGFeatureExtractor(nn.Module):
    """
    Powerful MEG feature extractor using multi-scale convolutions and self-attention.
    """
    
    def __init__(self, n_channels: int = 306, n_timepoints: int = 125, 
                 hidden_dim: int = 512):
        super().__init__()
        
        # Initial projection to create channel embeddings
        self.channel_embedding = nn.Linear(n_channels, hidden_dim)
        
        # Multi-scale 1D convolutions over time
        kernel_sizes = [3, 5, 7]
        num_scales = len(kernel_sizes)
        # Ensure branch output channels sum exactly to hidden_dim
        branch_out_channels = [hidden_dim // num_scales] * num_scales
        branch_out_channels[-1] = hidden_dim - sum(branch_out_channels[:-1])
        self.conv_blocks = nn.ModuleList()
        for k, out_ch in zip(kernel_sizes, branch_out_channels):
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dim, out_ch, kernel_size=k, padding=k//2),
                    nn.BatchNorm1d(out_ch),
                    nn.GELU(),
                    nn.Dropout(0.1)
                )
            )
        
        # Combine multi-scale features
        self.combine_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        
        # Self-attention to capture temporal dependencies
        self.positional_encoding = PositionalEncoding(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        self.attention_norm = nn.LayerNorm(hidden_dim)
        
        # Channel attention to weight important sensors
        self.channel_attention = nn.Sequential(
            nn.Linear(n_channels, n_channels // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(n_channels // 4, n_channels),
            nn.Sigmoid()
        )
        
        # Final projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
    def forward(self, x):
        """
        Args:
            x: MEG signals (batch_size, n_channels, n_timepoints)
        Returns:
            features: (batch_size, hidden_dim // 2)
        """
        batch_size, n_channels, n_timepoints = x.shape
        
        # Apply channel attention
        channel_weights = self.channel_attention(x.mean(dim=2))  # (batch, n_channels)
        x = x * channel_weights.unsqueeze(2)
        
        # Transpose for channel embedding: (batch, time, channels)
        x = x.transpose(1, 2)  # (batch, n_timepoints, n_channels)
        
        # Project channels to hidden dimension
        x = self.channel_embedding(x)  # (batch, n_timepoints, hidden_dim)
        
        # Apply convolutions (need to transpose for Conv1d)
        x_conv = x.transpose(1, 2)  # (batch, hidden_dim, n_timepoints)
        
        # Multi-scale convolutions
        conv_outputs = []
        for conv_block in self.conv_blocks:
            conv_outputs.append(conv_block(x_conv))
        
        # Concatenate and combine
        x_conv = torch.cat(conv_outputs, dim=1)  # (batch, hidden_dim, n_timepoints)
        x_conv = self.combine_conv(x_conv)
        
        # Back to (batch, time, features) for attention
        x_conv = x_conv.transpose(1, 2)
        
        # Add positional encoding
        x_conv = self.positional_encoding(x_conv)
        
        # Self-attention
        attn_out, _ = self.self_attention(x_conv, x_conv, x_conv)
        x_conv = self.attention_norm(x_conv + attn_out)  # Residual connection
        
        # Global average pooling over time
        features = x_conv.mean(dim=1)  # (batch, hidden_dim)
        
        # Final projection
        features = self.output_projection(features)  # (batch, hidden_dim // 2)
        
        return features


class PhonemeClassifierHead(nn.Module):
    """
    Classification head with optional auxiliary tasks.
    """
    
    def __init__(self, input_dim: int, n_phonemes: int = 39, 
                 use_auxiliary: bool = True):
        super().__init__()
        
        self.use_auxiliary = use_auxiliary
        
        # Main classifier
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(input_dim, n_phonemes)
        )
        
        if use_auxiliary:
            # Auxiliary task: predict phoneme class (vowel/consonant/silence)
            self.phoneme_type_classifier = nn.Linear(input_dim, 3)
            
            # Auxiliary task: predict voicing
            self.voicing_classifier = nn.Linear(input_dim, 2)
    
    def forward(self, features):
        """
        Args:
            features: (batch_size, input_dim)
        Returns:
            Dictionary with predictions
        """
        outputs = {
            'phoneme_logits': self.classifier(features)
        }
        
        if self.use_auxiliary:
            outputs['phoneme_type_logits'] = self.phoneme_type_classifier(features)
            outputs['voicing_logits'] = self.voicing_classifier(features)
        
        return outputs


class SimplifiedMEGClassifier(L.LightningModule):
    """
    Simplified but powerful MEG phoneme classifier.
    Focus on strong feature extraction and classification.
    """
    
    def __init__(self,
                 # Architecture params
                 n_channels: int = 306,
                 time_points: int = 125,  # Compatible with train.py
                 n_classes: int = 39,  # Compatible with train.py
                 hidden_dim: int = 512,
                 
                 # Training params
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.01,
                 label_smoothing: float = 0.1,
                 warmup_epochs: int = 5,
                 max_epochs: int = 100,
                 
                 # Auxiliary tasks
                 use_auxiliary: bool = True,
                 auxiliary_weight: float = 0.1,
                 
                 # Augmentation
                 use_mixup: bool = True,
                 mixup_alpha: float = 0.2,
                 use_cutmix: bool = False,
                 cutmix_alpha: float = 1.0):
        
        super().__init__()
        self.save_hyperparameters()
        
        # MEG feature extractor
        self.feature_extractor = MEGFeatureExtractor(
            n_channels=n_channels,
            n_timepoints=time_points,
            hidden_dim=hidden_dim
        )
        
        # Classification head
        self.classifier_head = PhonemeClassifierHead(
            input_dim=hidden_dim // 2,
            n_phonemes=n_classes,
            use_auxiliary=use_auxiliary
        )
        
        # Losses
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # Metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=n_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=n_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=n_classes, average='macro')
        
        # For auxiliary tasks
        if use_auxiliary:
            self.phoneme_type_map = self._create_phoneme_type_map()
            self.voicing_map = self._create_voicing_map()
    
    def _create_phoneme_type_map(self):
        """Map phoneme indices to types (0: silence, 1: vowel, 2: consonant)."""
        phoneme_types = torch.zeros(39, dtype=torch.long)
        
        # Silence
        phoneme_types[0] = 0
        
        # Vowels
        vowel_indices = [1, 2, 6, 10, 11, 18, 19, 26, 27, 28, 35, 36, 37, 38]
        for idx in vowel_indices:
            phoneme_types[idx] = 1
        
        # Consonants (everything else)
        for idx in range(39):
            if idx != 0 and idx not in vowel_indices:
                phoneme_types[idx] = 2
        
        return phoneme_types
    
    def _create_voicing_map(self):
        """Map phoneme indices to voicing (0: unvoiced, 1: voiced)."""
        voicing = torch.zeros(39, dtype=torch.long)
        
        # Voiced phonemes
        voiced_indices = [1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 14, 15, 16, 17, 
                         18, 19, 21, 26, 27, 28, 29, 30, 31, 32, 33, 
                         35, 36, 37, 38]
        for idx in voiced_indices:
            voicing[idx] = 1
        
        return voicing
    
    def mixup_data(self, x, y, alpha=1.0):
        """Apply mixup augmentation."""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def cutmix_data(self, x, y, alpha=1.0):
        """Apply cutmix augmentation on MEG data."""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        # Cut on time dimension
        time_points = x.size(2)
        cut_len = int(time_points * (1 - lam))
        cut_start = np.random.randint(0, time_points - cut_len + 1)
        
        x_mixed = x.clone()
        x_mixed[:, :, cut_start:cut_start + cut_len] = x[index, :, cut_start:cut_start + cut_len]
        
        # Adjust lambda based on actual cut ratio
        lam = 1 - (cut_len / time_points)
        
        return x_mixed, y, y[index], lam
    
    def forward(self, x):
        """Forward pass for inference."""
        features = self.feature_extractor(x)
        outputs = self.classifier_head(features)
        return outputs['phoneme_logits']
    
    def training_step(self, batch, batch_idx):
        x, labels = batch
        
        # Apply augmentation
        if self.training and self.hparams.use_mixup and np.random.random() < 0.5:
            x, labels_a, labels_b, lam = self.mixup_data(x, labels, self.hparams.mixup_alpha)
            
            # Forward pass
            features = self.feature_extractor(x)
            outputs = self.classifier_head(features)
            
            # Mixup loss
            loss = lam * self.ce_loss(outputs['phoneme_logits'], labels_a) + \
                   (1 - lam) * self.ce_loss(outputs['phoneme_logits'], labels_b)
            
            # Use original labels for metrics
            acc = self.train_accuracy(outputs['phoneme_logits'], labels_a)
            
        elif self.training and self.hparams.use_cutmix and np.random.random() < 0.3:
            x, labels_a, labels_b, lam = self.cutmix_data(x, labels, self.hparams.cutmix_alpha)
            
            # Forward pass
            features = self.feature_extractor(x)
            outputs = self.classifier_head(features)
            
            # Cutmix loss
            loss = lam * self.ce_loss(outputs['phoneme_logits'], labels_a) + \
                   (1 - lam) * self.ce_loss(outputs['phoneme_logits'], labels_b)
            
            acc = self.train_accuracy(outputs['phoneme_logits'], labels_a)
            
        else:
            # Standard forward pass
            features = self.feature_extractor(x)
            outputs = self.classifier_head(features)
            
            loss = self.ce_loss(outputs['phoneme_logits'], labels)
            acc = self.train_accuracy(outputs['phoneme_logits'], labels)
        
        # Add auxiliary losses if enabled
        if self.hparams.use_auxiliary and not (self.hparams.use_mixup or self.hparams.use_cutmix):
            # Get auxiliary labels
            phoneme_types = self.phoneme_type_map[labels].to(x.device)
            voicing = self.voicing_map[labels].to(x.device)
            
            # Auxiliary losses
            type_loss = F.cross_entropy(outputs['phoneme_type_logits'], phoneme_types)
            voicing_loss = F.cross_entropy(outputs['voicing_logits'], voicing)
            
            loss = loss + self.hparams.auxiliary_weight * (type_loss + voicing_loss)
            
            self.log('train_type_loss', type_loss, prog_bar=False)
            self.log('train_voicing_loss', voicing_loss, prog_bar=False)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_accuracy', acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, labels = batch
        
        # Forward pass (no augmentation in validation)
        features = self.feature_extractor(x)
        outputs = self.classifier_head(features)
        
        loss = self.ce_loss(outputs['phoneme_logits'], labels)
        acc = self.val_accuracy(outputs['phoneme_logits'], labels)
        f1 = self.val_f1(outputs['phoneme_logits'], labels)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', acc, prog_bar=True)
        self.log('val_f1_macro', f1, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        # Group parameters with different learning rates
        params = [
            {'params': self.feature_extractor.parameters(), 'lr': self.hparams.learning_rate},
            {'params': self.classifier_head.parameters(), 'lr': self.hparams.learning_rate * 2}
        ]
        
        optimizer = torch.optim.AdamW(
            params,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing with warmup
        def lr_lambda(epoch):
            if epoch < self.hparams.warmup_epochs:
                # Linear warmup
                return (epoch + 1) / self.hparams.warmup_epochs
            else:
                # Cosine annealing
                progress = (epoch - self.hparams.warmup_epochs) / \
                          (self.hparams.max_epochs - self.hparams.warmup_epochs)
                return 0.5 * (1.0 + math.cos(math.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }