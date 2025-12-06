"""
Single-Stage MEG Model for Phoneme Classification
Balanced pre-training to handle class imbalance with focal loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import F1Score, Accuracy  # Add Accuracy import
from collections import defaultdict

# ============================================
# Balanced Pre-training Module
# ============================================

class BalancedPhonemePretrainer(nn.Module):
    """ # Ihor: avoid linear weights - exponential/temperature - so super rare phonemes won't get large weights
    Pre-training module to address class imbalance.
    Uses focal loss and class reweighting based on phoneme performance.
    """
    
    def __init__(self, vocab_size: int = 39, hidden_dim: int = 16):
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
        
        # Create class weights (higher weight for poor performers)
        self.class_weights = torch.zeros(vocab_size)
        for i, score in self.phoneme_performance.items():
            # Inverse weight with smoothing
            self.class_weights[i] = 1.0 / (score + 0.1)
        
        # Normalize weights
        self.class_weights = self.class_weights / self.class_weights.mean()
        
    def focal_loss(self, logits: torch.Tensor, targets: torch.Tensor, 
                   gamma: float = 1.0, alpha: torch.Tensor = None):
        """
        Focal loss to focus on hard-to-classify phonemes.
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** gamma * ce_loss
        
        if alpha is not None:
            focal_loss = alpha[targets] * focal_loss
        
        return focal_loss.mean()

# ============================================
# MEG Conformer Layer
# ============================================

class MEGConformerLayer(nn.Module):
    """Conformer layer adapted for MEG data."""
    
    def __init__(self, dim: int, num_heads: int = 1, ff_dim: int = None, 
                 kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        ff_dim = ff_dim or 2 * dim
        
        # Depthwise separable convolution
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim),
            nn.BatchNorm1d(dim),
            nn.Conv1d(dim, dim, 1),
            nn.ReLU()
        )
        
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ln3 = nn.LayerNorm(dim)
        
        # deberta unique: in other positional models, positional info independant, while in deberta positional info is more dependant on context
        # Replace multihead attention with deberta attention (see link)
        # Ihor will share deberta custom, just need some modifications
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    # Ihor: issue with post-layer norm you increase gradient loss; bad for cases with little data; you can try pre-layer norm, when you normalize only
    # inputs that go to operation (ex. attention, convolution, fully connected - see Discord pic); but issue is you can lose info during normalization
    # Also look into mixed-layer norm
    def forward(self, x):
        # x: (B, T, D)
        # Convolution module
        res = x
        x_conv = x.transpose(1, 2)  # (B, D, T)
        x_conv = self.conv(x_conv).transpose(1, 2)  # (B, T, D)
        x = self.ln1(x_conv + res)
        
        # Self-attention module
        res = x
        attn_out, _ = self.attention(x, x, x)
        x = self.ln2(self.dropout(attn_out) + res)
        
        # Feed-forward module
        res = x
        x = self.ffn(x)
        x = self.ln3(x + res)
        
        return x

# ============================================
# Main Single-Stage MEG Model
# ============================================

class SingleStageMEGClassifier(L.LightningModule):
    """
    Single-stage phoneme classification for MEG data.
    Supports cross entropy (default) and focal loss with class reweighting.
    """
    
    def __init__(self,
                 meg_channels: int = 306,
                 time_points: int = 125,
                 vocab_size: int = 39,
                 hidden_dim: int = 64,
                 num_conformers: int = 1,
                 learning_rate: float = 1e-4,
                 use_conformer: bool = True,
                 loss_type: str = "cross_entropy",  # "cross_entropy" or "focal"
                 focal_gamma: float = 1.0,
                 dropout_rate: float = 0.0,
                 label_smoothing: float = 0.0,
                 weight_decay: float = 0.0,
                 classifier_lr_multiplier: float = 1.0,
                 warmup_epochs: int = 0,
                 total_epochs: int = 100,
                 metric_type: str = "f1_macro"):
        super().__init__()
        self.save_hyperparameters()
        
        self.metric_type = metric_type

        # Balanced pre-trainer
        self.pretrainer = BalancedPhonemePretrainer(vocab_size, hidden_dim)
        
        # MEG encoder
        if use_conformer:
            # Initial projection
            self.input_projection = nn.Sequential(
                nn.Conv1d(meg_channels, hidden_dim, kernel_size=7, padding=3),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ) # Ihor: super basic; as with images, CNN work better not with batch norms; minimize amount of normalizations; try replace relu with signmoid linear unit
              # If you have a lot of convolutions, it's important to have residual connections; knowledgator -> global layers before convolutions; 
            
            # Conformer layers
            self.meg_encoder = nn.ModuleList([
                MEGConformerLayer(hidden_dim, 4, hidden_dim*2, dropout=dropout_rate) 
                for _ in range(num_conformers)
            ])
            
            self.encoder_output_dim = hidden_dim
        else:
            # LSTM encoder
            self.input_projection = None
            self.meg_encoder = nn.LSTM(
                meg_channels, hidden_dim, num_conformers,
                batch_first=True, bidirectional=True, dropout=dropout_rate
            )
            self.encoder_output_dim = hidden_dim * 2
        
        self.use_conformer = use_conformer
        
        # Classification head with stronger regularization
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder_output_dim * time_points, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate + 0.1),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(32),
            nn.Linear(32, vocab_size)
        ) # Simplify, as this is the first layer gradient passes through, super deep classifier can be challenging to train
        
        # Loss parameters
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        
        # Metrics
        self.train_f1 = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
        self.val_f1 = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
        self.test_f1 = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
        self.phoneme_f1_scores = defaultdict(float)
        self.phoneme_counts = defaultdict(int)

        if metric_type == "balanced_acc":
            self.train_metric = Accuracy(num_classes=vocab_size, average='macro', task="multiclass")
            self.val_metric = Accuracy(num_classes=vocab_size, average='macro', task="multiclass")
            self.test_metric = Accuracy(num_classes=vocab_size, average='macro', task="multiclass")
            self.metric_name = "balanced_acc"
        else:  # f1_macro
            self.train_metric = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
            self.val_metric = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
            self.test_metric = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
            self.metric_name = "f1_macro"
        
        self.phoneme_f1_scores = defaultdict(float)
        self.phoneme_counts = defaultdict(int)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features using the chosen encoder.
        """
        B, C, T = x.shape
        
        if self.use_conformer:
            # Apply initial convolution
            features = self.input_projection(x)  # (B, hidden_dim, T)
            features = features.transpose(1, 2)  # (B, T, hidden_dim)
            
            # Apply conformer layers
            for conformer in self.meg_encoder:
                features = conformer(features)
        else:
            x = x.transpose(1, 2)  # (B, T, C)
            features, _ = self.meg_encoder(x)  # (B, T, hidden_dim*2)
        # Can add layer norm for final features, but if using post-layer norm don't need 
        return features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification.
        """
        B, C, T = x.shape
        features = self.extract_features(x)  # (B, T, D)
        features_flat = features.reshape(B, -1)  # (B, T*D)
        logits = self.classifier(features_flat)  # (B, vocab_size)
        return logits
    
    def compute_loss(self, logits, targets):
        """
        Compute loss based on configured loss type.
        Supports cross_entropy (default) and focal loss.
        """
        if self.loss_type == "focal":
            # Focal loss with class weights
            focal_loss = self.pretrainer.focal_loss(
                logits, targets, 
                gamma=self.focal_gamma,
                alpha=self.pretrainer.class_weights.to(logits.device)
            )
            
            # Add label smoothing if configured
            if self.label_smoothing > 0:
                smooth_loss = F.cross_entropy(
                    logits, targets, 
                    label_smoothing=self.label_smoothing
                )
                loss = 0.7 * focal_loss + 0.3 * smooth_loss
            else:
                loss = focal_loss
                
        else:  # cross_entropy (default)
            # Standard cross entropy loss
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
        logits = self(x)
        loss = self.compute_loss(logits, y)
        
        # Metrics
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = (preds == y).float().mean()
            metric_value = self.train_metric(logits, y)
            
            # Track per-phoneme performance
            for i in range(len(y)):
                phoneme_id = y[i].item()
                self.phoneme_counts[phoneme_id] += 1
                if preds[i] == y[i]:
                    self.phoneme_f1_scores[phoneme_id] += 1
        
        self.log('train_loss', loss, prog_bar=True)
        self.log(f'train_{self.metric_name}', metric_value, prog_bar=True)
        self.log('train_acc', acc)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.compute_loss(logits, y)
        
        # Metrics
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()
        metric_value = self.val_metric(logits, y)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log(f'val_{self.metric_name}', metric_value, prog_bar=True)
        self.log('val_acc', acc)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.compute_loss(logits, y)
        
        # Metrics
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()
        metric_value = self.test_metric(logits, y)
        
        self.log('test_loss', loss)
        self.log(f'test_{self.metric_name}', metric_value)
        self.log('test_acc', acc)
        
        return loss

    def on_train_epoch_end(self):
        """
        Log per-phoneme performance statistics.
        """
        if self.current_epoch % 5 == 0:  # Log every 5 epochs
            print(f"\n{'='*50}")
            print(f"Epoch {self.current_epoch} - Per-Phoneme Performance:")
            
            # Calculate per-phoneme accuracy
            phoneme_accuracies = {}
            for phoneme_id in self.phoneme_counts:
                if self.phoneme_counts[phoneme_id] > 0:
                    accuracy = self.phoneme_f1_scores[phoneme_id] / self.phoneme_counts[phoneme_id]
                    phoneme_accuracies[phoneme_id] = accuracy
            
            # Find best and worst performing phonemes
            sorted_phonemes = sorted(phoneme_accuracies.items(), key=lambda x: x[1])
            
            print("Worst performing phonemes:")
            for pid, acc in sorted_phonemes[:5]:
                print(f"  Phoneme {pid}: {acc:.3f} (count: {self.phoneme_counts[pid]})")
            
            print("Best performing phonemes:")
            for pid, acc in sorted_phonemes[-5:]:
                print(f"  Phoneme {pid}: {acc:.3f} (count: {self.phoneme_counts[pid]})")
            
            print(f"{'='*50}\n")
            
            # Reset counters
            self.phoneme_f1_scores = defaultdict(float)
            self.phoneme_counts = defaultdict(int)
    
    def configure_optimizers(self):
        """
        Configure optimizer with different learning rates for different components.
        """
        params = []
        
        # Encoder parameters with base learning rate
        if self.use_conformer and self.input_projection is not None:
            params.append({
                'params': self.input_projection.parameters(), 
                'lr': self.hparams.learning_rate
            })
        
        params.append({
            'params': self.meg_encoder.parameters(), 
            'lr': self.hparams.learning_rate
        })
        
        # Classifier with higher learning rate
        params.append({
            'params': self.classifier.parameters(), 
            'lr': self.hparams.learning_rate * self.hparams.classifier_lr_multiplier
        })
        
        optimizer = torch.optim.AdamW(params, weight_decay=self.hparams.weight_decay)
        
        # Learning rate scheduling with warmup
        def lr_lambda(epoch):
            if epoch < self.hparams.warmup_epochs:
                # Linear warmup
                return (epoch + 1) / self.hparams.warmup_epochs
            else:
                # Cosine annealing
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