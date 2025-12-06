"""
Enhanced CBraMod architecture for phoneme classification.

This is a simplified and optimized version of CBraMod specifically designed for:
- Short sequences (0.5s / 125 time points)
- MEG phoneme classification with 39 classes
- Better regularization to prevent overfitting
- Efficient patch-based processing for small inputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import F1Score
from einops import rearrange
from einops.layers.torch import Rearrange


class PatchEmbedding(nn.Module):
    """Enhanced patch embedding for short MEG sequences."""
    
    def __init__(self, in_dim, d_model, num_patches=5, num_channels=306):
        super().__init__()
        self.d_model = d_model
        self.num_patches = num_patches
        
        # Channel mixing with spatial attention
        self.channel_mixer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(num_channels//6, 5), 
                     stride=(num_channels//12, 1), padding=(0, 2)),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, d_model, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(d_model),
        )
        
        # Learnable positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, d_model) * 0.02)
        
        # Spectral features branch (following original CBraMod)
        self.spectral_proj = nn.Sequential(
            nn.Linear(in_dim//2 + 1, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(0.1),
        )
        
    def forward(self, x):
        # x shape: (batch, channels, patches, patch_size)
        batch_size, channels, num_patches, patch_size = x.shape
        
        # Spatial processing branch
        x_spatial = x.unsqueeze(1)  # (batch, 1, channels, patches*patch_size)
        x_spatial = x_spatial.reshape(batch_size, 1, channels, -1)
        x_spatial = self.channel_mixer(x_spatial)  # (batch, d_model, H', W')
        
        # Pool over spatial dimension and reshape to patches
        x_spatial = F.adaptive_avg_pool2d(x_spatial, (1, num_patches * 8))
        x_spatial = x_spatial.squeeze(2).reshape(batch_size, self.d_model, num_patches, -1)
        x_spatial = x_spatial.mean(dim=-1).transpose(1, 2)  # (batch, num_patches, d_model)
        
        # Spectral features branch (FFT-based)
        x_fft = torch.fft.rfft(x.reshape(batch_size * channels * num_patches, patch_size), 
                               dim=-1, norm='ortho')
        x_spectral = torch.abs(x_fft).reshape(batch_size, channels, num_patches, -1)
        x_spectral = x_spectral.mean(dim=1)  # Average over channels
        x_spectral = self.spectral_proj(x_spectral)  # (batch, num_patches, d_model)
        
        # Combine spatial and spectral features
        x = x_spatial + x_spectral
        
        # Add positional encoding
        x = x + self.pos_embed[:, :num_patches, :]
        
        return x


class SimpleCBraMod(nn.Module):
    """Simplified CBraMod for phoneme classification."""
    
    def __init__(self, in_dim=25, out_dim=128, d_model=128, dim_feedforward=512,
                 n_layer=6, nhead=8, num_patches=5, dropout=0.3):
        super().__init__()
        
        self.patch_embedding = PatchEmbedding(in_dim, d_model, num_patches)
        
        # Transformer encoder with better regularization
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for better training stability
        )
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layer,
            norm=nn.LayerNorm(d_model)
        )
        
        # Global pooling options
        self.pool = nn.Sequential(
            Rearrange('b n d -> b d n'),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(1)
        )
        
        # Classification head with strong regularization
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 39)  # 39 phoneme classes
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        # x shape: (batch, channels, patches, patch_size)
        
        # Patch embedding with positional encoding
        x = self.patch_embedding(x)
        
        # Transformer encoding
        x = self.encoder(x)
        
        # Global pooling
        x = self.pool(x)
        
        # Classification
        logits = self.classifier(x)
        
        return logits


class PhonemeClassificationCBraMod(L.LightningModule):
    """Lightning module for phoneme classification using SimpleCBraMod."""
    
    def __init__(self, 
                 learning_rate=0.001,
                 time_points=None,  # Added to match other models
                 patch_size=25,
                 num_patches=5,
                 d_model=128,
                 dim_feedforward=512,
                 n_layer=6,
                 nhead=8,
                 dropout=0.3):
        super().__init__()
        self.save_hyperparameters()
        
        # Key parameters for phoneme task
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.learning_rate = learning_rate
        
        # Initialize model
        self.model = SimpleCBraMod(
            in_dim=self.patch_size,
            out_dim=d_model,
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            n_layer=n_layer,
            nhead=nhead,
            num_patches=self.num_patches,
            dropout=dropout
        )
        
        # Loss and metrics
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing
        self.train_f1 = F1Score(task="multiclass", num_classes=39, average='macro')
        self.val_f1 = F1Score(task="multiclass", num_classes=39, average='macro')
        
    def prepare_input(self, x):
        """Prepare input for patch-based processing."""
        batch_size, channels, time_points = x.shape
        
        # Ensure we have exactly patch_size * num_patches time points
        target_len = self.patch_size * self.num_patches
        
        if time_points < target_len:
            # Pad with edge values if too short
            pad_len = target_len - time_points
            x = F.pad(x, (0, pad_len), mode='replicate')
        elif time_points > target_len:
            # Center crop if too long
            start = (time_points - target_len) // 2
            x = x[:, :, start:start + target_len]
            
        # Reshape to patches
        x = x.reshape(batch_size, channels, self.num_patches, self.patch_size)
        
        return x
        
    def forward(self, x):
        x = self.prepare_input(x)
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        f1 = self.train_f1(preds, y)
        
        # Logging
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_f1_macro', f1, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        f1 = self.val_f1(preds, y)
        
        # Logging
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_f1_macro', f1, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        f1 = self.val_f1(preds, y)
        
        # Logging
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_f1_macro', f1, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        # Optimizer with weight decay
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.05,
            betas=(0.9, 0.95)
        )
        
        # Cosine annealing with warmup
        warmup_steps = 500
        total_steps = self.trainer.estimated_stepping_batches
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
    
    def on_train_epoch_end(self):
        # Reset metrics at epoch end
        self.train_f1.reset()
        
    def on_validation_epoch_end(self):
        # Reset metrics at epoch end
        self.val_f1.reset()