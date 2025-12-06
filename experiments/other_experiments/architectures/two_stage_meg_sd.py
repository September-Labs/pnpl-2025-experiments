"""
Single-Stage MEG Model for Phoneme Classification
Balanced pre-training to handle class imbalance with focal loss
Enhanced with spectral density features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import F1Score, Accuracy
from collections import defaultdict
import numpy as np
from scipy import signal
from typing import Optional, Tuple

# ============================================
# Spectral Feature Extraction Module
# ============================================

class SpectralDensityExtractor(nn.Module):
    """
    Extract power spectral density features from MEG signals.
    Computes PSDs across multiple frequency bands relevant for speech processing.
    """
    
    def __init__(self, 
                 sampling_rate: float = 250.0,  # Hz
                 freq_bands: Optional[list] = None,
                 nperseg: int = 32,
                 noverlap: Optional[int] = None,
                 use_learnable_filters: bool = False):
        super().__init__()
        
        self.sampling_rate = sampling_rate
        self.nperseg = nperseg
        self.noverlap = noverlap if noverlap is not None else nperseg // 2
        
        # Default frequency bands relevant for speech/phoneme processing
        if freq_bands is None:
            self.freq_bands = [
                (0.5, 4),    # Delta
                (4, 8),      # Theta
                (8, 13),     # Alpha
                (13, 30),    # Beta
                (30, 50),    # Low Gamma
                (50, 100),   # High Gamma
                (100, 125)   # Very High Gamma (up to Nyquist for 250Hz sampling)
            ]
        else:
            self.freq_bands = freq_bands
        
        self.n_bands = len(self.freq_bands)
        self.use_learnable_filters = use_learnable_filters
        
        # Optional learnable frequency band attention
        if use_learnable_filters:
            self.band_attention = nn.Parameter(torch.ones(self.n_bands))
            self.band_projection = nn.Linear(self.n_bands, self.n_bands * 2)
    
    def compute_psd_torch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute PSD using PyTorch operations for differentiability.
        Uses Welch's method approximation.
        
        Args:
            x: Input tensor of shape (batch, channels, time)
        
        Returns:
            PSD features of shape (batch, channels, n_bands)
        """
        B, C, T = x.shape
        
        # Initialize output tensor
        psd_features = torch.zeros(B, C, self.n_bands, device=x.device)
        
        # Compute STFT for spectral analysis
        # Window for STFT
        window = torch.hann_window(self.nperseg, device=x.device)
        
        for b in range(B):
            for c in range(C):
                # Compute STFT
                stft_result = torch.stft(
                    x[b, c],
                    n_fft=self.nperseg,
                    hop_length=self.nperseg - self.noverlap,
                    win_length=self.nperseg,
                    window=window,
                    return_complex=True,
                    center=True
                )
                
                # Compute power spectral density
                psd = torch.abs(stft_result) ** 2
                
                # Average across time windows
                psd_mean = psd.mean(dim=-1)
                
                # Frequency bins
                freqs = torch.fft.fftfreq(self.nperseg, 1/self.sampling_rate)[:psd_mean.shape[0]]
                freqs = freqs.to(x.device)
                
                # Extract power in each frequency band
                for band_idx, (low_freq, high_freq) in enumerate(self.freq_bands):
                    mask = (freqs >= low_freq) & (freqs <= high_freq)
                    if mask.sum() > 0:
                        # Average power in the frequency band
                        band_power = psd_mean[mask].mean()
                        psd_features[b, c, band_idx] = torch.log1p(band_power)  # Log transform for stability
        
        return psd_features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract spectral density features from MEG signals.
        
        Args:
            x: Input tensor of shape (batch, channels, time)
        
        Returns:
            Spectral features of shape (batch, channels, n_bands) or 
            (batch, channels, n_bands*2) if using learnable filters
        """
        # Compute PSD features
        psd_features = self.compute_psd_torch(x)
        
        # Apply learnable band attention if enabled
        if self.use_learnable_filters:
            # Apply attention weights to bands
            psd_features = psd_features * self.band_attention.view(1, 1, -1)
            
            # Project to higher dimension
            B, C, _ = psd_features.shape
            psd_features_flat = psd_features.reshape(B * C, -1)
            psd_features_proj = self.band_projection(psd_features_flat)
            psd_features = psd_features_proj.reshape(B, C, -1)
        
        return psd_features

# ============================================
# Multi-Modal Feature Fusion Module
# ============================================

class MultiModalFusion(nn.Module):
    """
    Fuse temporal and spectral features from MEG signals.
    """
    
    def __init__(self, 
                 temporal_dim: int,
                 spectral_dim: int,
                 output_dim: int,
                 fusion_type: str = "concat",  # "concat", "attention", "gated"
                 dropout: float = 0.1):
        super().__init__()
        
        self.fusion_type = fusion_type
        self.temporal_dim = temporal_dim
        self.spectral_dim = spectral_dim
        self.output_dim = output_dim
        
        if fusion_type == "concat":
            # Simple concatenation with projection
            self.fusion_proj = nn.Sequential(
                nn.Linear(temporal_dim + spectral_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            
        elif fusion_type == "attention":
            # Cross-attention based fusion
            self.temporal_proj = nn.Linear(temporal_dim, output_dim)
            self.spectral_proj = nn.Linear(spectral_dim, output_dim)
            
            self.cross_attention = nn.MultiheadAttention(
                output_dim, num_heads=4, dropout=dropout, batch_first=True
            )
            
            self.fusion_proj = nn.Sequential(
                nn.LayerNorm(output_dim),
                nn.Linear(output_dim, output_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            
        elif fusion_type == "gated":
            # Gated fusion mechanism
            self.temporal_proj = nn.Linear(temporal_dim, output_dim)
            self.spectral_proj = nn.Linear(spectral_dim, output_dim)
            
            # Gates for controlling information flow
            self.temporal_gate = nn.Sequential(
                nn.Linear(temporal_dim + spectral_dim, output_dim),
                nn.Sigmoid()
            )
            self.spectral_gate = nn.Sequential(
                nn.Linear(temporal_dim + spectral_dim, output_dim),
                nn.Sigmoid()
            )
            
            self.fusion_proj = nn.Sequential(
                nn.LayerNorm(output_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
    
    def forward(self, temporal_features: torch.Tensor, 
                spectral_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse temporal and spectral features.
        
        Args:
            temporal_features: Shape (batch, time, temporal_dim) or (batch, temporal_dim)
            spectral_features: Shape (batch, spectral_dim)
        
        Returns:
            Fused features of shape (batch, output_dim)
        """
        if self.fusion_type == "concat":
            # Ensure both features have same batch dimension
            if len(temporal_features.shape) == 3:
                # Pool temporal features across time
                temporal_features = temporal_features.mean(dim=1)
            
            # Concatenate and project
            fused = torch.cat([temporal_features, spectral_features], dim=-1)
            output = self.fusion_proj(fused)
            
        elif self.fusion_type == "attention":
            # Project features
            if len(temporal_features.shape) == 2:
                temporal_features = temporal_features.unsqueeze(1)
            
            temporal_proj = self.temporal_proj(temporal_features)
            spectral_proj = self.spectral_proj(spectral_features.unsqueeze(1))
            
            # Cross-attention (spectral attending to temporal)
            attended, _ = self.cross_attention(
                spectral_proj, temporal_proj, temporal_proj
            )
            
            # Combine and project
            output = self.fusion_proj(attended.squeeze(1))
            
        elif self.fusion_type == "gated":
            # Pool temporal features if necessary
            if len(temporal_features.shape) == 3:
                temporal_features = temporal_features.mean(dim=1)
            
            # Project features
            temporal_proj = self.temporal_proj(temporal_features)
            spectral_proj = self.spectral_proj(spectral_features)
            
            # Compute gates
            concat_features = torch.cat([temporal_features, spectral_features], dim=-1)
            temporal_gate = self.temporal_gate(concat_features)
            spectral_gate = self.spectral_gate(concat_features)
            
            # Apply gates and combine
            gated_temporal = temporal_gate * temporal_proj
            gated_spectral = spectral_gate * spectral_proj
            output = self.fusion_proj(gated_temporal + gated_spectral)
        
        return output

# ============================================
# Balanced Pre-training Module (unchanged)
# ============================================

class BalancedPhonemePretrainer(nn.Module):
    """
    Pre-training module to address class imbalance.
    Uses focal loss and class reweighting based on phoneme performance.
    """
    
    def __init__(self, vocab_size: int = 39, hidden_dim: int = 16):
        super().__init__()

        phoneme_counts = {
            0: 216, 1: 443, 2: 1584, 3: 231, 4: 114, 5: 360, 6: 268, 7: 93,
            8: 772, 9: 504, 10: 477, 11: 429, 12: 231, 13: 282, 14: 119, 15: 428,
            16: 1052, 17: 570, 18: 63, 19: 430, 20: 566, 21: 518, 22: 1128, 23: 154,
            24: 226, 25: 14, 26: 276, 27: 634, 28: 743, 29: 113, 30: 1143, 31: 110,
            32: 96, 33: 236, 34: 326, 35: 428, 36: 151, 37: 456, 38: 7
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
# MEG Conformer Layer (unchanged)
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
        
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
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
# Enhanced Single-Stage MEG Model with Spectral Features
# ============================================

class SingleStageMEGClassifier(L.LightningModule):
    """
    Single-stage phoneme classification for MEG data.
    Enhanced with spectral density features and multi-modal fusion.
    """
    
    def __init__(self,
                 meg_channels: int = 306,
                 time_points: int = 125,
                 vocab_size: int = 39,
                 hidden_dim: int = 64,
                 num_conformers: int = 1,
                 learning_rate: float = 1e-4,
                 use_conformer: bool = True,
                 use_spectral: bool = True,
                 spectral_bands: Optional[list] = None,
                 fusion_type: str = "concat",
                 sampling_rate: float = 250.0,
                 loss_type: str = "cross_entropy",
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
        self.use_spectral = use_spectral

        # Balanced pre-trainer
        self.pretrainer = BalancedPhonemePretrainer(vocab_size, hidden_dim)
        
        # Spectral feature extractor
        if use_spectral:
            self.spectral_extractor = SpectralDensityExtractor(
                sampling_rate=sampling_rate,
                freq_bands=spectral_bands,
                use_learnable_filters=True
            )
            
            # Spectral feature dimension (channels * n_bands * 2 with learnable filters)
            spectral_feature_dim = meg_channels * self.spectral_extractor.n_bands * 2
        
        # MEG temporal encoder
        if use_conformer:
            # Initial projection
            self.input_projection = nn.Sequential(
                nn.Conv1d(meg_channels, hidden_dim, kernel_size=7, padding=3),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )
            
            # Conformer layers
            self.meg_encoder = nn.ModuleList([
                MEGConformerLayer(hidden_dim, 4, hidden_dim*2, dropout=dropout_rate) 
                for _ in range(num_conformers)
            ])
            
            temporal_feature_dim = hidden_dim * time_points
        else:
            # LSTM encoder
            self.input_projection = None
            self.meg_encoder = nn.LSTM(
                meg_channels, hidden_dim, num_conformers,
                batch_first=True, bidirectional=True, dropout=dropout_rate
            )
            temporal_feature_dim = hidden_dim * 2 * time_points
        
        self.use_conformer = use_conformer
        
        # Multi-modal fusion
        if use_spectral:
            self.fusion = MultiModalFusion(
                temporal_dim=temporal_feature_dim,
                spectral_dim=spectral_feature_dim,
                output_dim=128,
                fusion_type=fusion_type,
                dropout=dropout_rate
            )
            classifier_input_dim = 128
        else:
            classifier_input_dim = temporal_feature_dim
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate + 0.1),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(32),
            nn.Linear(32, vocab_size)
        )
        
        # Loss parameters
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        
        # Metrics
        if metric_type == "balanced_acc":
            self.train_metric = Accuracy(num_classes=vocab_size, average='macro', task="multiclass")
            self.val_metric = Accuracy(num_classes=vocab_size, average='macro', task="multiclass")
            self.test_metric = Accuracy(num_classes=vocab_size, average='macro', task="multiclass")
            self.metric_name = "balanced_acc"
        else:
            self.train_metric = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
            self.val_metric = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
            self.test_metric = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
            self.metric_name = "f1_macro"
        
        self.phoneme_f1_scores = defaultdict(float)
        self.phoneme_counts = defaultdict(int)
    
    def extract_temporal_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract temporal features using the chosen encoder.
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
        
        return features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification with spectral features.
        """
        B, C, T = x.shape
        
        # Extract temporal features
        temporal_features = self.extract_temporal_features(x)  # (B, T, D)
        temporal_features_flat = temporal_features.reshape(B, -1)  # (B, T*D)
        
        if self.use_spectral:
            # Extract spectral features
            spectral_features = self.spectral_extractor(x)  # (B, C, n_bands*2)
            spectral_features_flat = spectral_features.reshape(B, -1)  # (B, C*n_bands*2)
            
            # Fuse temporal and spectral features
            fused_features = self.fusion(temporal_features_flat, spectral_features_flat)
            
            # Classification
            logits = self.classifier(fused_features)
        else:
            # Classification using only temporal features
            logits = self.classifier(temporal_features_flat)
        
        return logits
    
    def compute_loss(self, logits, targets):
        """
        Compute loss based on configured loss type.
        """
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
                
        else:
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
        if self.current_epoch % 5 == 0:
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
        
        # Spectral extractor parameters
        if self.use_spectral:
            params.append({
                'params': self.spectral_extractor.parameters(),
                'lr': self.hparams.learning_rate * 0.5  # Lower LR for spectral features
            })
        
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
        
        # Fusion module parameters
        if self.use_spectral:
            params.append({
                'params': self.fusion.parameters(),
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
