"""
5-Layer GRU Network for Phoneme Classification
Based on Kunz et al. 2025 architecture, adapted for classification task
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import F1Score
from collections import defaultdict

# ============================================
# Balanced Pre-training Module
# ============================================

class BalancedPhonemePretrainer(nn.Module):
    """
    Pre-training module to address class imbalance.
    Maintains phoneme statistics for monitoring training.
    """
    
    def __init__(self, vocab_size: int = 39, hidden_dim: int = 256):
        super().__init__()

        phoneme_counts = {
            0: 216,
            1: 443,
            2: 1584,
            3: 231,
            4: 114,
            5: 360,
            6: 268,
            7: 93,
            8: 772,
            9: 504,
            10: 477,
            11: 429,
            12: 231,
            13: 282,
            14: 119,
            15: 428,
            16: 1052,
            17: 570,
            18: 63,
            19: 430,
            20: 566,
            21: 518,
            22: 1128,
            23: 154,
            24: 226,
            25: 14,
            26: 276,
            27: 634,
            28: 743,
            29: 113,
            30: 1143,
            31: 110,
            32: 96,
            33: 236,
            34: 326,
            35: 428,
            36: 151,
            37: 456,
            38: 7
        }

        self.phoneme_performance = {
            i: 1.0 / (count/15991 + 0.001) 
            for i, count in phoneme_counts.items()
        }
        
        # Create class weights for potential weighting
        self.class_weights = torch.zeros(vocab_size)
        for i, score in self.phoneme_performance.items():
            self.class_weights[i] = 1.0 / (score + 0.1)
        
        # Normalize weights
        self.class_weights = self.class_weights / self.class_weights.mean()
        
    def focal_loss(self, logits: torch.Tensor, targets: torch.Tensor, 
                   gamma: float = 2.0, alpha: torch.Tensor = None):
        """
        Focal loss for addressing class imbalance.
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** gamma * ce_loss
        
        if alpha is not None:
            focal_loss = alpha[targets] * focal_loss
        
        return focal_loss.mean()

# ============================================
# 5-Layer Stacked GRU RNN
# ============================================

class StackedGRUEncoder(nn.Module):
    """5-layer stacked GRU as described in the paper."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, 
                 num_layers: int = 5, dropout: float = 0.3):
        super().__init__()
        
        self.gru_layers = nn.ModuleList()
        
        # First GRU layer
        self.gru_layers.append(
            nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        )
        
        # Remaining GRU layers (bidirectional, so input is 2*hidden_dim)
        for _ in range(num_layers - 1):
            self.gru_layers.append(
                nn.GRU(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
            )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim * 2) for _ in range(num_layers)
        ])
    
    def forward(self, x):
        # x: (B, T, D)
        for i, (gru, ln) in enumerate(zip(self.gru_layers, self.layer_norms)):
            # Apply GRU
            output, _ = gru(x)
            
            # Residual connection (if dimensions match)
            if output.shape == x.shape:
                output = output + x
            
            # Layer normalization and dropout
            output = ln(output)
            output = self.dropout(output)
            
            x = output
        
        return x

# ============================================
# Main GRU Classifier
# ============================================

class SingleStageMEGClassifier(L.LightningModule):
    """
    5-layer stacked GRU RNN for phoneme classification.
    Architecture inspired by Kunz et al. 2025, adapted for classification.
    """
    
    def __init__(self,
                 meg_channels: int = 306,
                 time_points: int = 125,
                 vocab_size: int = 39,
                 hidden_dim: int = 256,
                 num_conformers: int = 5,  # Reused as num_gru_layers
                 learning_rate: float = 1e-3,
                 use_conformer: bool = False,  # Ignored, always use GRU
                 loss_type: str = "cross_entropy",
                 focal_gamma: float = 2.0,
                 dropout_rate: float = 0.3,
                 label_smoothing: float = 0.0,
                 weight_decay: float = 1e-4,
                 classifier_lr_multiplier: float = 1.0,
                 warmup_epochs: int = 5,
                 total_epochs: int = 100,
                 add_noise: bool = True):
        super().__init__()
        self.save_hyperparameters()
        
        # Balanced pre-trainer (for monitoring and optional focal loss)
        self.pretrainer = BalancedPhonemePretrainer(vocab_size, hidden_dim)
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(meg_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5)  # Light dropout at input
        )
        
        # 5-layer stacked GRU RNN encoder
        self.meg_encoder = StackedGRUEncoder(
            hidden_dim, 
            hidden_dim, 
            num_layers=num_conformers,  # Using this param for GRU layers
            dropout=dropout_rate
        )
        
        self.encoder_output_dim = hidden_dim * 2  # Bidirectional
        
        # Temporal pooling options
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, vocab_size)
        )
        
        # Noise augmentation parameters
        self.add_noise = add_noise
        self.noise_std = 0.05
        
        # Loss type
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        
        # Metrics
        self.train_f1 = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
        self.val_f1 = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
        self.test_f1 = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
        self.phoneme_f1_scores = defaultdict(float)
        self.phoneme_counts = defaultdict(int)
    
    def add_artificial_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add artificial noise for regularization.
        """
        if not self.add_noise or not self.training:
            return x
        
        # Gaussian noise
        noise = torch.randn_like(x) * self.noise_std
        x = x + noise
        
        # Random temporal shift
        if torch.rand(1).item() < 0.3:
            shift = torch.randint(-2, 3, (1,)).item()
            if shift != 0:
                x = torch.roll(x, shifts=shift, dims=2)
        
        return x
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features using 5-layer stacked GRU RNN.
        """
        B, C, T = x.shape
        
        # Add noise for regularization
        x = self.add_artificial_noise(x)
        
        # Transpose for RNN: (B, T, C)
        x = x.transpose(1, 2)
        
        # Input projection
        x = self.input_projection(x)  # (B, T, hidden_dim)
        
        # Apply 5-layer stacked GRU
        features = self.meg_encoder(x)  # (B, T, hidden_dim*2)
        
        return features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification.
        """
        B, C, T = x.shape
        features = self.extract_features(x)  # (B, T, hidden_dim*2)
        
        # Temporal pooling: average across time
        features = features.transpose(1, 2)  # (B, hidden_dim*2, T)
        pooled = self.temporal_pool(features).squeeze(-1)  # (B, hidden_dim*2)
        
        # Classification
        logits = self.classifier(pooled)  # (B, vocab_size)
        
        return logits
    
    def compute_loss(self, logits, targets):
        """
        Compute loss based on configured loss type.
        """
        if self.loss_type == "focal":
            # Focal loss with class weights
            loss = self.pretrainer.focal_loss(
                logits, targets, 
                gamma=self.focal_gamma,
                alpha=self.pretrainer.class_weights.to(logits.device)
            )
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
        x, y = batch  # Standard classification format
        
        logits = self(x)
        loss = self.compute_loss(logits, y)
        
        # Metrics
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = (preds == y).float().mean()
            f1 = self.train_f1(logits, y)
            
            # Track per-phoneme performance
            for i in range(len(y)):
                phoneme_id = y[i].item()
                self.phoneme_counts[phoneme_id] += 1
                if preds[i] == y[i]:
                    self.phoneme_f1_scores[phoneme_id] += 1
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_f1', f1, prog_bar=True)
        self.log('train_acc', acc)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch  # Standard classification format
        
        logits = self(x)
        loss = self.compute_loss(logits, y)
        
        # Metrics
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()
        f1 = self.val_f1(logits, y)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_f1_macro', f1, prog_bar=True)
        self.log('val_acc', acc)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch  # Standard classification format
        
        logits = self(x)
        loss = self.compute_loss(logits, y)
        
        # Metrics
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()
        f1 = self.test_f1(logits, y)
        
        self.log('test_loss', loss)
        self.log('test_f1_macro', f1)
        self.log('test_acc', acc)
        
        return loss
    
    def on_train_epoch_end(self):
        """
        Log training statistics.
        """
        if self.current_epoch % 10 == 0:
            print(f"\n{'='*50}")
            print(f"Epoch {self.current_epoch} - 5-Layer GRU Network")
            print(f"Architecture: {self.hparams.num_conformers} stacked bidirectional GRU layers")
            print(f"Hidden dim: {self.hparams.hidden_dim}, Dropout: {self.hparams.dropout_rate}")
            print(f"{'='*50}\n")
            
            # Reset counters
            self.phoneme_f1_scores = defaultdict(float)
            self.phoneme_counts = defaultdict(int)
    
    def configure_optimizers(self):
        """
        Configure AdamW optimizer with warmup and cosine annealing.
        """
        params = []
        
        # Input projection
        params.append({
            'params': self.input_projection.parameters(), 
            'lr': self.hparams.learning_rate
        })
        
        # GRU encoder
        params.append({
            'params': self.meg_encoder.parameters(), 
            'lr': self.hparams.learning_rate
        })
        
        # Classifier
        params.append({
            'params': self.classifier.parameters(), 
            'lr': self.hparams.learning_rate * self.hparams.classifier_lr_multiplier
        })
        
        optimizer = torch.optim.AdamW(params, weight_decay=self.hparams.weight_decay)
        
        # Learning rate scheduling
        def lr_lambda(epoch):
            if epoch < self.hparams.warmup_epochs:
                return (epoch + 1) / self.hparams.warmup_epochs
            else:
                progress = (epoch - self.hparams.warmup_epochs) / (self.hparams.total_epochs - self.hparams.warmup_epochs)
                return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }