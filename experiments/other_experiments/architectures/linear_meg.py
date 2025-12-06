"""
Fixed Simplified MEG Model for Phoneme Classification
Corrects LayerNorm dimension issues with Conv1d outputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import F1Score
from typing import Optional

class SimpleLightweightConformer(nn.Module):
    """Ultra-lightweight conformer - just the essentials"""
    
    def __init__(self, dim, num_heads=2, ff_mult=2, dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        
        normed = self.norm2(x)
        x = x + self.ffn(normed)
        
        return x
        
class SimplifiedMEGModel(L.LightningModule):
    def __init__(
        self, 
        meg_channels=306,
        time_points=125,
        vocab_size=39,
        hidden_dim=128,  # Keep same as successful model
        num_conformers=2,  # Just 1-2, not 4
        conformer_heads=2,  # Fewer heads
        learning_rate=1e-4,
        weight_decay=0.01,
        label_smoothing=0.05,
        dropout=0.3,
        conv_kernel_size=5,
        pool_size=2,
        classifier_hidden=512,
        **kwargs):

        super().__init__()
        self.save_hyperparameters()
        
        # EXACT same encoder as SimplifiedMEGModel (0.342 score)
        self.meg_encoder = nn.Sequential(
            nn.Conv1d(meg_channels, hidden_dim, kernel_size=conv_kernel_size, 
                    padding=conv_kernel_size//2),
            nn.BatchNorm1d(hidden_dim),  # Keep BatchNorm, not LayerNorm
            nn.ReLU(),
            nn.MaxPool1d(pool_size)
        )
        
        time_after_pool = time_points // pool_size
        
        # Add MINIMAL conformers
        self.conformers = nn.ModuleList()
        for _ in range(num_conformers):
            self.conformers.append(
                SimpleLightweightConformer(
                    dim=hidden_dim,
                    num_heads=conformer_heads,
                    ff_mult=2,  # Small expansion
                    dropout=0.1  # Light dropout in conformer
                )
            )
        
        # EXACT same classifier as SimplifiedMEGModel
        classifier_input = hidden_dim * time_after_pool
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, vocab_size)
        )
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.f1_macro = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
        self.validation_step_outputs = []


    def forward(self, x):
        B = x.size(0)
        
        # Encode (same as 0.342 model)
        features = self.meg_encoder(x)  # (B, hidden_dim, T/2)
        
        # Apply lightweight conformers
        features = features.transpose(1, 2)  # (B, T/2, hidden_dim)
        for conformer in self.conformers:
            features = conformer(features)
        features = features.transpose(1, 2)  # (B, hidden_dim, T/2)
        
        # Flatten and classify (same as 0.342 model)
        features = features.reshape(B, -1)
        logits = self.classifier(features)
        
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # Forward pass
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Metrics
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == y).float().mean()
            f1 = self.f1_macro(logits, y)
        
        # Logging
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        self.log('train_f1_macro', f1)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        # Forward pass
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Metrics
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == y).float().mean()
        f1 = self.f1_macro(logits, y)
        
        # Logging
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_f1_macro', f1)
        
        self.validation_step_outputs.append({
            'loss': loss,
            'acc': acc,
            'f1': f1,
            'preds': preds,
            'targets': y
        })
        
        return loss
    
    def on_validation_epoch_end(self):
        """Compute epoch-level metrics"""
        if self.validation_step_outputs:
            avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
            avg_acc = torch.stack([x['acc'] for x in self.validation_step_outputs]).mean()
            avg_f1 = torch.stack([x['f1'] for x in self.validation_step_outputs]).mean()
            
            self.validation_step_outputs.clear()
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        
        # Forward pass
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Metrics
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == y).float().mean()
        f1 = self.f1_macro(logits, y)
        
        # Logging
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        self.log('test_f1_macro', f1)
        
        return loss
    
    def configure_optimizers(self):
        """Simple optimizer configuration"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def predict_step(self, batch, batch_idx):
        """For generating predictions on test data"""
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        logits = self(x)
        preds = torch.argmax(logits, dim=-1)
        return preds


class UltraSimpleMEGModel(L.LightningModule):
    """
    Ultra-minimal version: Single conv + global pooling + classifier
    For testing if extreme simplification helps
    """
    
    def __init__(self,
                 meg_channels: int = 306,
                 time_points: int = 125,  # Accept but don't necessarily use
                 vocab_size: int = 39,
                 hidden_dim: int = 256,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01,
                 label_smoothing: float = 0.05,
                 **kwargs):  # Catch extra params
        
        super().__init__()
        self.save_hyperparameters()
        
        # Ultra-simple: One conv + global pool
        self.encoder = nn.Sequential(
            nn.Conv1d(meg_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),  # BatchNorm1d for Conv1d output
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Global average pooling
        )
        
        # Direct classification from pooled features
        self.classifier = nn.Linear(hidden_dim, vocab_size)
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.f1_macro = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
    
    def forward(self, x):
        # x: (B, 306, 125)
        features = self.encoder(x).squeeze(-1)  # (B, hidden_dim)
        logits = self.classifier(features)  # (B, vocab_size)
        return logits
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        acc = (torch.argmax(logits, dim=-1) == y).float().mean()
        f1 = self.f1_macro(logits, y)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        self.log('train_f1_macro', f1)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        acc = (torch.argmax(logits, dim=-1) == y).float().mean()
        f1 = self.f1_macro(logits, y)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_f1_macro', f1)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        acc = (torch.argmax(logits, dim=-1) == y).float().mean()
        f1 = self.f1_macro(logits, y)
        
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        self.log('test_f1_macro', f1)
        
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )


class LayerNormConv1d(nn.Module):
    """
    Custom LayerNorm for Conv1d outputs
    Applies LayerNorm over the channel dimension for (B, C, T) tensors
    """
    def __init__(self, num_features):
        super().__init__()
        self.ln = nn.LayerNorm(num_features)
    
    def forward(self, x):
        # x: (B, C, T)
        x = x.transpose(1, 2)  # (B, T, C)
        x = self.ln(x)
        x = x.transpose(1, 2)  # (B, C, T)
        return x


class SimplifiedMEGModelWithLayerNorm(L.LightningModule):
    """
    Alternative version that properly uses LayerNorm with Conv1d
    """
    
    def __init__(self, 
                 meg_channels: int = 306,
                 time_points: int = 125,
                 vocab_size: int = 39,
                 hidden_dim: int = 128,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01,
                 label_smoothing: float = 0.05,
                 dropout: float = 0.3,
                 use_single_conv: bool = True,
                 conv_kernel_size: int = 5,
                 use_pooling: bool = True,
                 pool_size: int = 2,
                 classifier_hidden: int = 512,
                 **kwargs):
        
        super().__init__()
        self.save_hyperparameters()
        
        # Build encoder with proper LayerNorm
        encoder_layers = []
        
        if use_single_conv:
            encoder_layers.extend([
                nn.Conv1d(meg_channels, hidden_dim, kernel_size=conv_kernel_size, padding=conv_kernel_size//2),
                LayerNormConv1d(hidden_dim),  # Custom LayerNorm for Conv1d
                nn.ReLU()
            ])
            
            if use_pooling:
                encoder_layers.append(nn.MaxPool1d(pool_size))
                time_after_pool = time_points // pool_size
            else:
                time_after_pool = time_points
        else:
            encoder_layers.extend([
                nn.Conv1d(meg_channels, hidden_dim, kernel_size=conv_kernel_size, padding=conv_kernel_size//2),
                LayerNormConv1d(hidden_dim),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                LayerNormConv1d(hidden_dim),
                nn.ReLU()
            ])
            
            if use_pooling:
                encoder_layers.append(nn.MaxPool1d(pool_size))
                time_after_pool = time_points // pool_size
            else:
                time_after_pool = time_points
        
        self.meg_encoder = nn.Sequential(*encoder_layers)
        self.time_after_encoding = time_after_pool
        
        # Classifier
        classifier_input_dim = hidden_dim * time_after_pool
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, vocab_size)
        )
        
        # Loss and metrics
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.f1_macro = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
        self.validation_step_outputs = []
    
    def forward(self, x):
        B = x.size(0)
        features = self.meg_encoder(x)  # (B, hidden_dim, T_reduced)
        features = features.reshape(B, -1)  # (B, hidden_dim * T_reduced)
        logits = self.classifier(features)  # (B, vocab_size)
        return logits
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == y).float().mean()
            f1 = self.f1_macro(logits, y)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        self.log('train_f1_macro', f1)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == y).float().mean()
        f1 = self.f1_macro(logits, y)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_f1_macro', f1)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == y).float().mean()
        f1 = self.f1_macro(logits, y)
        
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        self.log('test_f1_macro', f1)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }