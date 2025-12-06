# little_classifiers.py (FIXED VERSION)
"""
Train 39 completely separate binary models, one for each phoneme.
Each model gets custom class balancing based on phoneme frequency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import F1Score, AUROC, Accuracy
import numpy as np
from pathlib import Path
import yaml
import json


class SinglePhonemeDetector(nn.Module):  # Changed to nn.Module, not L.LightningModule
    """
    Independent binary classifier for a single phoneme.
    Optimized architecture based on phoneme frequency.
    """
    
    def __init__(self,
                 target_phoneme: int,
                 phoneme_count: int,
                 total_samples: int,
                 time_points: int = 125,
                 n_channels: int = 306,
                 n_classes: int = 39,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.01,
                 label_smoothing: float = 0.0):
        super().__init__()
        
        self.target_phoneme = target_phoneme
        self.phoneme_count = phoneme_count
        self.total_samples = total_samples
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Calculate natural frequency
        self.natural_freq = phoneme_count / total_samples
        
        # Determine model size based on data availability
        if phoneme_count < 50:  # Very rare phonemes
            hidden_dim = 64
            n_layers = 2
            dropout = 0.5
        elif phoneme_count < 200:  # Rare phonemes
            hidden_dim = 128
            n_layers = 3
            dropout = 0.4
        elif phoneme_count < 500:  # Medium frequency
            hidden_dim = 192
            n_layers = 3
            dropout = 0.3
        else:  # Common phonemes
            hidden_dim = 256
            n_layers = 4
            dropout = 0.2
        
        # Calculate optimal class balance ratio
        if phoneme_count < 50:
            self.pos_weight = 10.0
            self.balance_ratio = min(0.3, phoneme_count / 100)
        elif phoneme_count < 200:
            self.pos_weight = 5.0
            self.balance_ratio = min(0.2, phoneme_count / 500)
        else:
            self.pos_weight = 2.0
            self.balance_ratio = min(0.1, phoneme_count / 1000)
        
        print(f"Phoneme {target_phoneme}: {phoneme_count} samples, "
              f"hidden_dim={hidden_dim}, balance_ratio={self.balance_ratio:.3f}, "
              f"pos_weight={self.pos_weight:.1f}")
        
        # Build adaptive architecture
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )
        
        # Adaptive depth
        conv_layers = []
        for i in range(n_layers):
            conv_layers.extend([
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        self.temporal_processor = nn.Sequential(*conv_layers)
        
        # Global pooling
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Loss and metrics
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.pos_weight]))
        self.train_f1 = F1Score(task="binary")
        self.val_f1 = F1Score(task="binary")
        self.val_auroc = AUROC(task="binary")
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.temporal_processor(x)
        max_pool = self.global_max_pool(x).squeeze(-1)
        avg_pool = self.global_avg_pool(x).squeeze(-1)
        x = torch.cat([max_pool, avg_pool], dim=1)
        logits = self.classifier(x)
        return logits.squeeze(-1)
    
    def prepare_batch(self, x, y):
        """Convert to binary classification problem."""
        y_binary = (y == self.target_phoneme).float()
        return x, y_binary
    
    def balanced_sampling(self, x, y_binary):
        """Custom balanced sampling based on phoneme frequency."""
        pos_idx = (y_binary == 1).nonzero(as_tuple=True)[0]
        neg_idx = (y_binary == 0).nonzero(as_tuple=True)[0]
        
        n_pos = len(pos_idx)
        n_neg = len(neg_idx)
        
        if n_pos == 0:
            return x, y_binary
        
        if self.phoneme_count < 50:
            n_neg_needed = min(n_neg, int(n_pos / self.balance_ratio) - n_pos)
            neg_idx = neg_idx[torch.randperm(n_neg)[:n_neg_needed]]
            all_idx = torch.cat([pos_idx, neg_idx])
        else:
            if n_pos / (n_pos + n_neg) < self.balance_ratio:
                n_pos_needed = int(n_neg * self.balance_ratio / (1 - self.balance_ratio))
                if n_pos_needed > n_pos:
                    oversample_idx = torch.randint(0, n_pos, (n_pos_needed - n_pos,))
                    pos_idx = torch.cat([pos_idx, pos_idx[oversample_idx]])
            else:
                n_neg_needed = int(n_pos * (1 - self.balance_ratio) / self.balance_ratio)
                neg_idx = neg_idx[torch.randperm(n_neg)[:n_neg_needed]]
            
            all_idx = torch.cat([pos_idx, neg_idx])
        
        all_idx = all_idx[torch.randperm(len(all_idx))]
        return x[all_idx], y_binary[all_idx]


class PhonemeModelManager(L.LightningModule):
    """
    Manager class that trains individual phoneme models.
    This is the Lightning module that handles training.
    """
    
    def __init__(self,
                 time_points: int = 125,
                 n_channels: int = 306,
                 n_classes: int = 39,
                 current_phoneme: int = 0,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.01,
                 label_smoothing: float = 0.0,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        # Phoneme counts
        self.phoneme_counts = {
            0: 216, 1: 443, 2: 1584, 3: 231, 4: 114, 5: 360, 6: 268, 7: 93,
            8: 772, 9: 504, 10: 477, 11: 429, 12: 231, 13: 282, 14: 119,
            15: 428, 16: 1052, 17: 570, 18: 63, 19: 430, 20: 566, 21: 518,
            22: 1128, 23: 154, 24: 226, 25: 14, 26: 276, 27: 634, 28: 743,
            29: 113, 30: 1143, 31: 110, 32: 96, 33: 236, 34: 326, 35: 428,
            36: 151, 37: 456, 38: 7
        }
        
        self.total_samples = sum(self.phoneme_counts.values())
        self.current_phoneme = current_phoneme
        
        # Create the model for the specified phoneme
        self.model = SinglePhonemeDetector(
            target_phoneme=current_phoneme,
            phoneme_count=self.phoneme_counts[current_phoneme],
            total_samples=self.total_samples,
            time_points=time_points,
            n_channels=n_channels,
            n_classes=n_classes,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            label_smoothing=label_smoothing
        )
        
        # Metrics (managed by Lightning)
        self.train_f1 = F1Score(task="binary")
        self.val_f1 = F1Score(task="binary")
        self.val_auroc = AUROC(task="binary")
        
        # Overall metrics
        self.f1_macro = F1Score(num_classes=n_classes, average='macro', task="multiclass")
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y_binary = self.model.prepare_batch(x, y)
        
        # Balance batch
        if self.training:
            x, y_binary = self.model.balanced_sampling(x, y_binary)
        
        # Forward pass
        logits = self.model(x)
        loss = self.model.criterion(logits, y_binary)
        
        # Metrics
        preds = torch.sigmoid(logits) > 0.5
        f1 = self.train_f1(preds, y_binary.int())
        
        # Log from the Lightning module
        self.log(f'train_loss', loss, prog_bar=True)
        self.log(f'train_f1', f1, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y_binary = self.model.prepare_batch(x, y)
        
        # No balancing in validation
        logits = self.model(x)
        loss = self.model.criterion(logits, y_binary)
        
        # Metrics
        probs = torch.sigmoid(logits)
        preds = probs > 0.5
        
        f1 = self.val_f1(preds, y_binary.int())
        
        # Log from the Lightning module
        self.log(f'val_loss', loss, prog_bar=True)
        self.log(f'val_f1', f1, prog_bar=True)
        
        # Only calculate AUROC if we have both classes
        if y_binary.sum() > 0 and y_binary.sum() < len(y_binary):
            auroc = self.val_auroc(probs, y_binary.int())
            self.log(f'val_auroc', auroc)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        # Adjust learning rate based on data availability
        phoneme_count = self.phoneme_counts[self.current_phoneme]
        lr_scale = min(1.0, phoneme_count / 500)
        adjusted_lr = self.hparams.learning_rate * lr_scale
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=adjusted_lr,
            weight_decay=self.hparams.weight_decay
        )
        
        # More epochs for rare phonemes
        if phoneme_count < 100:
            T_0 = 20
        else:
            T_0 = 10
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=2,
            eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }
    
    def load_ensemble(self, model_dir):
        """Load all 39 models for ensemble inference."""
        self.ensemble_models = []
        model_dir = Path(model_dir)
        
        for i in range(39):
            possible_paths = [
                model_dir / f"little_classifiers_082325v0_phoneme_{i}" / "checkpoints" / "best.ckpt",
                model_dir / f"little_classifiers_082325v0_phoneme_{i}" / "checkpoints" / "last.ckpt",
            ]
            
            loaded = False
            for checkpoint_path in possible_paths:
                if checkpoint_path.exists():
                    try:
                        # Load the full Lightning module
                        manager = PhonemeModelManager.load_from_checkpoint(
                            checkpoint_path,
                            current_phoneme=i
                        )
                        manager.eval()
                        self.ensemble_models.append(manager.model)  # Extract the SinglePhonemeDetector
                        loaded = True
                        print(f"✓ Loaded model for phoneme {i}")
                        break
                    except Exception as e:
                        print(f"Error loading phoneme {i}: {e}")
            
            if not loaded:
                print(f"✗ No model found for phoneme {i}")
                # Create dummy model
                model = SinglePhonemeDetector(
                    target_phoneme=i,
                    phoneme_count=self.phoneme_counts[i],
                    total_samples=self.total_samples
                )
                self.ensemble_models.append(model)
        
        print(f"Loaded {len(self.ensemble_models)} models for ensemble inference")
