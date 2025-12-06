"""
Simple 1D Convolutional + Transformer architecture for MEG phoneme classification
Adapted to work with existing train.py script
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import F1Score
import numpy as np
from typing import Optional, Tuple
import math
from collections import Counter
from torch.utils.data import WeightedRandomSampler


class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance."""
    
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer('alpha', alpha)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
        
        return focal_loss.mean()


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1)]


class Conv1DFeatureExtractor(nn.Module):
    """1D Convolutional feature extractor for MEG data."""
    
    def __init__(self, 
                 input_channels: int = 306,
                 hidden_dim: int = 256,
                 num_conv_layers: int = 3,
                 kernel_size: int = 5,
                 dropout: float = 0.1):
        super().__init__()
        
        layers = []
        in_channels = input_channels
        
        for i in range(num_conv_layers):
            out_channels = hidden_dim // (2 ** (num_conv_layers - i - 1))
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        self.output_dim = hidden_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, channels, time_points]
        return self.conv_layers(x)  # [batch_size, hidden_dim, time_points]


class SimpleConvTransformer(L.LightningModule):
    """Simple Conv1D + Transformer model for MEG phoneme classification."""
    
    # Struggling phonemes (F1 = 0 from evaluation)
    STRUGGLING_PHONEMES = [
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
        30, 31, 32, 33, 34, 35, 36, 37, 38
    ]
    
    def __init__(self,
                 # REQUIRED by train.py
                 time_points: int = 125,  # This will be set by train.py
                 
                 # Data parameters
                 meg_channels: int = 306,
                 num_classes: int = 39,
                 
                 # Model architecture
                 hidden_dim: int = 256,
                 num_conv_layers: int = 3,
                 kernel_size: int = 5,
                 num_transformer_layers: int = 4,
                 num_heads: int = 8,
                 ff_dim: int = 1024,
                 dropout: float = 0.2,
                 
                 # Training parameters
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01,
                 label_smoothing: float = 0.1,
                 
                 # Masking strategies
                 use_focal_loss: bool = True,
                 focal_gamma: float = 2.0,
                 struggling_weight: float = 3.0,
                 use_mixup: bool = True,
                 mixup_alpha: float = 0.2,
                 augment_prob: float = 0.5,
                 use_balanced_sampling: bool = True,
                 
                 **kwargs):  # Catch any extra params from config
        
        super().__init__()
        self.save_hyperparameters()
        
        # Store time_points (required by train.py)
        self.time_points = time_points
        
        # 1D Convolutional feature extractor
        self.conv_extractor = Conv1DFeatureExtractor(
            input_channels=meg_channels,
            hidden_dim=hidden_dim,
            num_conv_layers=num_conv_layers,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Loss function
        if use_focal_loss:
            # Create class weights
            class_weights = torch.ones(num_classes)
            for idx in self.STRUGGLING_PHONEMES:
                class_weights[idx] = struggling_weight
            self.criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        else:
            if label_smoothing > 0:
                self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            else:
                self.criterion = nn.CrossEntropyLoss()
        
        # Metrics - IMPORTANT: train.py expects 'val_f1_macro' not 'val_f1'
        self.train_f1_macro = F1Score(num_classes=num_classes, average='macro', task='multiclass')
        self.val_f1_macro = F1Score(num_classes=num_classes, average='macro', task='multiclass')
        
        # Training params
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.augment_prob = augment_prob
        self.use_balanced_sampling = use_balanced_sampling
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, channels, time_points] MEG data
        Returns:
            logits: [batch_size, num_classes]
        """
        # Extract features with 1D convolutions
        features = self.conv_extractor(x)  # [B, hidden_dim, T]
        
        # Transpose for transformer (needs [B, T, hidden_dim])
        features = features.transpose(1, 2)  # [B, T, hidden_dim]
        
        # Add positional encoding
        features = self.pos_encoder(features)
        
        # Pass through transformer
        transformer_out = self.transformer(features)  # [B, T, hidden_dim]
        
        # Global average pooling over time
        pooled = transformer_out.mean(dim=1)  # [B, hidden_dim]
        
        # Classify
        logits = self.classifier(pooled)  # [B, num_classes]
        
        return logits
    
    def augment_struggling_phonemes(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Apply augmentation to struggling phoneme samples."""
        batch_size = x.shape[0]
        device = x.device
        
        for i in range(batch_size):
            if y[i].item() in self.STRUGGLING_PHONEMES and torch.rand(1) < self.augment_prob:
                aug_type = torch.randint(0, 4, (1,)).item()
                
                if aug_type == 0:
                    # Add Gaussian noise
                    noise = torch.randn_like(x[i]) * 0.1
                    x[i] = x[i] + noise
                
                elif aug_type == 1:
                    # Channel dropout
                    num_channels = x.shape[1]
                    channels_to_drop = torch.randperm(num_channels, device=device)[:num_channels//10]
                    x[i, channels_to_drop] = 0
                
                elif aug_type == 2:
                    # Time masking
                    T = x.shape[-1]
                    mask_len = T // 8
                    mask_start = torch.randint(0, T - mask_len, (1,)).item()
                    x[i, :, mask_start:mask_start+mask_len] = 0
                
                elif aug_type == 3:
                    # Time shift
                    shift = torch.randint(-5, 6, (1,)).item()
                    x[i] = torch.roll(x[i], shifts=shift, dims=-1)
        
        return x
    
    def mixup(self, x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2) -> Tuple:
        """Mixup augmentation."""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # Apply augmentation to struggling phonemes
        x = self.augment_struggling_phonemes(x, y)
        
        # Apply mixup
        if self.use_mixup and self.training:
            x, y_a, y_b, lam = self.mixup(x, y, self.mixup_alpha)
            logits = self(x)
            loss = lam * self.criterion(logits, y_a) + (1 - lam) * self.criterion(logits, y_b)
            
            # Calculate F1 on original labels
            f1_macro = self.train_f1_macro(logits, y_a)
        else:
            logits = self(x)
            loss = self.criterion(logits, y)
            f1_macro = self.train_f1_macro(logits, y)
        
        # Logging - use names that match train.py expectations
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_f1_macro', f1_macro, prog_bar=True)
        
        # Log struggling phoneme accuracy every 50 batches
        if batch_idx % 50 == 0:
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                for idx, phoneme_idx in enumerate(self.STRUGGLING_PHONEMES[:5]):
                    mask = y == phoneme_idx
                    if mask.any():
                        acc = (preds[mask] == phoneme_idx).float().mean()
                        self.log(f'train_acc_phoneme_{phoneme_idx}', acc)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        f1_macro = self.val_f1_macro(logits, y)
        
        # IMPORTANT: train.py monitors 'val_f1_macro'
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_f1_macro', f1_macro, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Get max epochs from trainer or use default
        max_epochs = self.trainer.max_epochs if hasattr(self, 'trainer') and self.trainer else 50
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epochs,
            eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }
    
    def on_train_start(self):
        """Hook to potentially modify dataloader for balanced sampling."""
        if self.use_balanced_sampling and hasattr(self.trainer, 'train_dataloader'):
            print("Note: For balanced sampling, modify your dataloader creation in train.py or config")
            print("The model supports it but cannot modify the dataloader after creation")