"""
LCS-CTC Adapted for MEG-based Phoneme Classification with Zipf Weighting
Integrates the LCS-CTC framework with LibriBrain MEG data
Includes Zipf distribution learning for phoneme priors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np
from typing import Optional, Dict, List, Tuple
from torchmetrics import F1Score
from collections import defaultdict

# ============================================
# Zipf Distribution Learner
# ============================================

class ZipfWeightLearner(nn.Module):
    """
    Learns Zipf distribution of phonemes and their MEG signatures.
    Uses exponential moving average for online learning.
    """
    
    def __init__(self, vocab_size: int, meg_dim: int, alpha: float = 0.99):
        super().__init__()
        self.vocab_size = vocab_size
        self.alpha = alpha  # EMA decay factor
        
        # Track phoneme frequencies (initialized uniformly)
        self.register_buffer('phoneme_counts', torch.ones(vocab_size))
        self.register_buffer('total_count', torch.tensor(vocab_size, dtype=torch.float32))
        
        # Learn prototypical MEG patterns per phoneme
        self.register_buffer('meg_prototypes', torch.zeros(vocab_size, meg_dim))
        self.register_buffer('prototype_counts', torch.ones(vocab_size))
        
        # Learnable Zipf parameters
        self.zipf_s = nn.Parameter(torch.tensor(1.0))  # Zipf exponent
        self.temperature = nn.Parameter(torch.tensor(1.0))  # Temperature for softmax
        
        # Phoneme-MEG attention module (outputs single weight per phoneme)
        self.phoneme_meg_attention = nn.Sequential(
            nn.Linear(meg_dim * 2, meg_dim),
            nn.ReLU(),
            nn.Linear(meg_dim, 1),  # Single attention weight per phoneme
            nn.Sigmoid()
        )
    
    def update_statistics(self, phonemes: torch.Tensor, meg_features: torch.Tensor):
        """
        Update phoneme frequency and MEG prototype statistics.
        
        Args:
            phonemes: (B,) phoneme labels
            meg_features: (B, meg_dim) aggregated MEG features
        """
        with torch.no_grad():
            # Update phoneme counts with EMA
            for phoneme in phonemes:
                self.phoneme_counts[phoneme] = self.alpha * self.phoneme_counts[phoneme] + (1 - self.alpha)
                self.total_count = self.alpha * self.total_count + (1 - self.alpha)
            
            # Update MEG prototypes with EMA
            for phoneme, meg_feat in zip(phonemes, meg_features):
                old_prototype = self.meg_prototypes[phoneme]
                self.meg_prototypes[phoneme] = (
                    self.alpha * old_prototype + (1 - self.alpha) * meg_feat
                )
                self.prototype_counts[phoneme] += 1
    
    def get_zipf_weights(self) -> torch.Tensor:
        """
        Compute Zipf-based prior probabilities for phonemes.
        
        Returns:
            (vocab_size,) tensor of prior probabilities
        """
        # Normalize counts to get frequencies
        frequencies = self.phoneme_counts / self.total_count
        
        # Sort by frequency to get ranks
        sorted_freqs, sorted_indices = torch.sort(frequencies, descending=True)
        ranks = torch.zeros_like(frequencies)
        ranks[sorted_indices] = torch.arange(1, self.vocab_size + 1, dtype=torch.float32, device=frequencies.device)
        
        # Apply Zipf's law: P(rank) ∝ 1 / rank^s
        zipf_weights = 1.0 / torch.pow(ranks, self.zipf_s)
        zipf_weights = zipf_weights / zipf_weights.sum()
        
        return zipf_weights
    
    def compute_meg_similarity(self, meg_features: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity between input MEG and learned prototypes.
        
        Args:
            meg_features: (B, meg_dim) MEG features
            
        Returns:
            (B, vocab_size) similarity scores
        """
        # Normalize prototypes
        norm_prototypes = F.normalize(self.meg_prototypes, p=2, dim=1)
        norm_meg = F.normalize(meg_features, p=2, dim=1)
        
        # Compute cosine similarity
        similarity = torch.matmul(norm_meg, norm_prototypes.T)  # (B, vocab_size)
        
        # Apply temperature scaling
        similarity = similarity / self.temperature
        
        return similarity
    
    def forward(self, meg_features: torch.Tensor, training: bool = False) -> torch.Tensor:
        """
        Compute Zipf-weighted adjustments for predictions.
        
        Args:
            meg_features: (B, meg_dim) aggregated MEG features
            training: Whether in training mode
            
        Returns:
            (B, vocab_size) weight adjustments
        """
        B = meg_features.size(0)
        
        # Get Zipf prior probabilities
        zipf_priors = self.get_zipf_weights().unsqueeze(0).expand(B, -1)  # (B, vocab_size)
        
        # Get MEG-based similarity scores
        meg_similarity = self.compute_meg_similarity(meg_features)  # (B, vocab_size)
        
        # Combine Zipf priors with MEG similarity
        # Use attention mechanism to balance the two
        combined_features = torch.cat([
            meg_features.unsqueeze(1).expand(-1, self.vocab_size, -1),
            self.meg_prototypes.unsqueeze(0).expand(B, -1, -1)
        ], dim=-1)  # (B, vocab_size, meg_dim * 2)
        
        attention_weights = self.phoneme_meg_attention(combined_features.reshape(B * self.vocab_size, -1))  # (B*vocab_size, 1)
        attention_weights = attention_weights.squeeze(-1).reshape(B, self.vocab_size)  # (B, vocab_size)
        
        # Weighted combination
        weights = zipf_priors * (1 + meg_similarity) * attention_weights
        weights = F.softmax(weights, dim=-1)
        
        return weights

# ============================================
# MEG-adapted Cost Matrix Learner
# ============================================

class MEGCostMatrixLearner(nn.Module):
    """Learn frame-phoneme cost matrix for MEG-phoneme alignment."""
    
    def __init__(self, vocab_size=39, meg_channels=306, hidden_dim=256, projection_dim=32): 
        super().__init__()
        # MEG encoder (replacing wav2vec2)
        self.meg_encoder = nn.Sequential(
            nn.Conv1d(meg_channels, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        self.temporal_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.linear = nn.Linear(hidden_dim, projection_dim)
        self.label_embedding = nn.Embedding(vocab_size, projection_dim)
    
    def forward(self, meg_data, text_labels):
        # meg_data: (B, channels, time_points)
        # Encode MEG features
        meg_features = self.meg_encoder(meg_data)  # (B, hidden_dim, T)
        meg_features = meg_features.transpose(1, 2)  # (B, T, hidden_dim)
        
        # Apply temporal attention
        meg_features, _ = self.temporal_attention(meg_features, meg_features, meg_features)
        meg_features = self.linear(meg_features)  # (B, T, projection_dim)
        
        # Encode text labels
        text_embeddings = self.label_embedding(text_labels)  # (B, L, projection_dim)
        
        # Compute cost matrix
        cost_matrix = -torch.matmul(text_embeddings, meg_features.transpose(1, 2))
        cost_matrix = F.softmax(cost_matrix, dim=1)  # (B, L, T)
        
        return cost_matrix

# ============================================
# MEG-adapted Conformer Layer
# ============================================

class MEGConformerLayer(nn.Module):
    """Conformer layer adapted for MEG data."""
    
    def __init__(self, dim: int, num_heads: int = 4, ff_dim: int = None, 
                 kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        ff_dim = ff_dim or 2 * dim
        
        # Depthwise separable convolution for efficiency
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
# MEG LCS-CTC Model for LibriBrain with Zipf
# ============================================

class MEGLCSCTC(L.LightningModule):
    """Main LCS-CTC model adapted for MEG phoneme classification with Zipf weighting."""
    
    def __init__(self, 
                 meg_channels=306,
                 time_points=125,
                 vocab_size=39,
                 hidden_dim=256,
                 num_conformers=4,
                 learning_rate=1e-4,
                 use_alignment=True,
                 zipf_weights=False,
                 zipf_alpha=0.99,
                 zipf_boost_factor=0.3):
        super().__init__()
        self.save_hyperparameters()
        
        # MEG feature extraction
        self.meg_encoder = nn.Sequential(
            nn.Conv1d(meg_channels, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Temporal modeling with Conformers
        self.conformers = nn.ModuleList([
            MEGConformerLayer(hidden_dim, 4, hidden_dim*2) 
            for _ in range(num_conformers)
        ])
        
        # CTC components
        self.ctc_projection = nn.Linear(hidden_dim, vocab_size + 1)  # +1 for blank
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        
        # Standard classification head (for non-CTC path)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * time_points, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, vocab_size)
        )
        
        # Cost matrix learner (optional)
        self.use_alignment = use_alignment
        if use_alignment:
            self.cost_learner = MEGCostMatrixLearner(
                vocab_size, meg_channels, hidden_dim
            )
        
        # Zipf weight learner (optional)
        self.use_zipf = zipf_weights
        if zipf_weights:
            self.zipf_learner = ZipfWeightLearner(
                vocab_size, 
                hidden_dim,
                alpha=zipf_alpha
            )
            self.zipf_boost_factor = zipf_boost_factor
            
            # Additional projection for MEG feature aggregation
            self.meg_aggregator = nn.Sequential(
                nn.Linear(hidden_dim * time_points, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
        
        self.criterion = nn.CrossEntropyLoss()
        self.f1_macro = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
        
    def forward(self, x, use_ctc=False):
        # x: (B, channels, time_points)
        B, C, T = x.shape
        
        # Encode MEG features
        features = self.meg_encoder(x)  # (B, hidden_dim, T)
        features = features.transpose(1, 2)  # (B, T, hidden_dim)
        
        # Apply conformers
        for conformer in self.conformers:
            features = conformer(features)  # (B, T, hidden_dim)
        
        if use_ctc:
            # CTC path
            logits = self.ctc_projection(features)  # (B, T, vocab_size+1)
            return logits
        else:
            # Standard classification path
            features_flat = features.reshape(B, -1)  # (B, hidden_dim * T)
            logits = self.classifier(features_flat)  # (B, vocab_size)
            
            # Apply Zipf weighting if enabled and not training
            if self.use_zipf and not self.training:
                # Aggregate MEG features for Zipf learner
                meg_agg = self.meg_aggregator(features_flat)  # (B, hidden_dim)
                
                # Get Zipf-based adjustments
                zipf_adjustments = self.zipf_learner(meg_agg, training=False)  # (B, vocab_size)
                
                # Apply adjustments to logits
                # Convert logits to probabilities, apply adjustment, then back to log space
                probs = F.softmax(logits, dim=-1)
                adjusted_probs = (1 - self.zipf_boost_factor) * probs + self.zipf_boost_factor * zipf_adjustments
                logits = torch.log(adjusted_probs + 1e-10)  # Add small epsilon for numerical stability
            
            return logits
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # Standard classification
        y_hat = self(x, use_ctc=False)
        loss = self.criterion(y_hat, y)
        
        # Update Zipf statistics if enabled
        if self.use_zipf:
            with torch.no_grad():
                # Get aggregated MEG features
                B, C, T = x.shape
                features = self.meg_encoder(x)
                features = features.transpose(1, 2)
                for conformer in self.conformers:
                    features = conformer(features)
                features_flat = features.reshape(B, -1)
                meg_agg = self.meg_aggregator(features_flat)
                
                # Update statistics
                self.zipf_learner.update_statistics(y, meg_agg)
        
        # Optional: Add CTC loss component
        if self.use_alignment and batch_idx % 10 == 0:  # Use CTC every 10 batches
            ctc_logits = self(x, use_ctc=True)
            # For CTC, we need sequence format
            log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)
            input_lengths = torch.full((x.size(0),), ctc_logits.size(1), dtype=torch.long)
            target_lengths = torch.ones(x.size(0), dtype=torch.long)
            
            # Create extended targets for CTC (single phoneme per sequence)
            ctc_targets = y.unsqueeze(1)
            
            ctc_loss = self.ctc_loss(log_probs, ctc_targets, input_lengths, target_lengths)
            loss = 0.7 * loss + 0.3 * ctc_loss
            
            self.log('train_ctc_loss', ctc_loss, prog_bar=False)
        
        f1_macro = self.f1_macro(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_f1_macro', f1_macro)
        
        # Log Zipf statistics if enabled
        if self.use_zipf and batch_idx % 100 == 0:
            zipf_weights = self.zipf_learner.get_zipf_weights()
            self.log('zipf_entropy', -torch.sum(zipf_weights * torch.log(zipf_weights + 1e-10)))
            self.log('zipf_s', self.zipf_learner.zipf_s)
            self.log('zipf_temperature', self.zipf_learner.temperature)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x, use_ctc=False)
        loss = self.criterion(y_hat, y)
        f1_macro = self.f1_macro(y_hat, y)
        
        self.log('val_loss', loss)
        self.log('val_f1_macro', f1_macro, prog_bar=True)
        
        # Additional evaluation with and without Zipf if enabled
        if self.use_zipf:
            # Get predictions without Zipf boost (for comparison)
            self.use_zipf = False
            y_hat_no_zipf = self(x, use_ctc=False)
            self.use_zipf = True
            
            f1_no_zipf = self.f1_macro(y_hat_no_zipf, y)
            self.log('val_f1_no_zipf', f1_no_zipf)
            self.log('val_f1_zipf_gain', f1_macro - f1_no_zipf)
        
        return loss
    
    def configure_optimizers(self):
        # Separate parameter groups for different learning rates
        params = [
            {'params': self.meg_encoder.parameters(), 'lr': self.hparams.learning_rate},
            {'params': self.conformers.parameters(), 'lr': self.hparams.learning_rate},
            {'params': self.classifier.parameters(), 'lr': self.hparams.learning_rate},
        ]
        
        if self.use_alignment:
            params.append({'params': self.cost_learner.parameters(), 'lr': self.hparams.learning_rate})
        
        if self.use_zipf:
            # Zipf learner gets a different learning rate
            params.append({'params': self.zipf_learner.parameters(), 'lr': self.hparams.learning_rate * 0.1})
            params.append({'params': self.meg_aggregator.parameters(), 'lr': self.hparams.learning_rate})
        
        optimizer = torch.optim.AdamW(params, weight_decay=0.01)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=50,
            eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }

# ============================================
# Simplified Alignment Module for MEG
# ============================================

class MEGPhonemeAligner:
    """Simplified alignment for MEG-phoneme matching."""
    
    def __init__(self, threshold=0.5):
        self.threshold = threshold
    
    def align(self, meg_features, phoneme_probs):
        """
        Simple alignment based on probability peaks.
        
        Args:
            meg_features: MEG features (B, T, D)
            phoneme_probs: Phoneme probabilities (B, vocab_size, T)
        
        Returns:
            Alignment mask
        """
        B, V, T = phoneme_probs.shape
        mask = torch.zeros_like(phoneme_probs)
        
        # Find peaks in probability distribution
        for b in range(B):
            for t in range(T):
                max_prob, max_idx = phoneme_probs[b, :, t].max(dim=0)
                if max_prob > self.threshold:
                    mask[b, max_idx, t] = 1.0
        
        return mask

# ============================================
# Training Integration with Zipf
# ============================================

def create_meg_lcs_ctc_model(dataset_info, use_zipf=False):
    """
    Create LCS-CTC model configured for LibriBrain dataset with optional Zipf weighting.
    
    Args:
        dataset_info: Dictionary with dataset information
            - meg_channels: Number of MEG channels (306)
            - time_points: Number of time points per sample (125 for 0.5s at 250Hz)
            - num_phonemes: Number of phoneme classes (39)
        use_zipf: Whether to enable Zipf weighting
    """
    model = MEGLCSCTC(
        meg_channels=dataset_info.get('meg_channels', 306),
        time_points=dataset_info.get('time_points', 125),
        vocab_size=dataset_info.get('num_phonemes', 39),
        hidden_dim=256,
        num_conformers=4,
        learning_rate=1e-4,
        use_alignment=True,
        zipf_weights=use_zipf,
        zipf_alpha=0.99,  # EMA decay for statistics
        zipf_boost_factor=0.3  # How much to boost predictions
    )
    
    return model

# ============================================
# Usage Example with Zipf Weighting
# ============================================

def integrate_with_libribrain(train_dataset, val_dataset, use_zipf=True):
    """
    Example integration with LibriBrain competition code including Zipf weighting.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset  
        use_zipf: Whether to enable Zipf-based boosting
    """
    from torch.utils.data import DataLoader
    import lightning as L
    
    # Dataset info
    dataset_info = {
        'meg_channels': 306,
        'time_points': 125,  # 0.5 seconds at 250Hz
        'num_phonemes': 39
    }
    
    # Create model with Zipf weighting
    model = create_meg_lcs_ctc_model(dataset_info, use_zipf=use_zipf)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Create trainer with callbacks for monitoring
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
    
    callbacks = [
        ModelCheckpoint(
            monitor='val_f1_macro',
            mode='max',
            save_top_k=3,
            filename='meg-lcs-ctc-zipf-{epoch:02d}-{val_f1_macro:.3f}'
        ),
        EarlyStopping(
            monitor='val_f1_macro',
            mode='max',
            patience=5
        )
    ]
    
    trainer = L.Trainer(
        devices="auto",
        max_epochs=20,
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,  # Effective batch size of 32
        callbacks=callbacks,
        enable_progress_bar=True
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    return model

# ============================================
# Analysis Tools for Zipf Distribution
# ============================================

def analyze_zipf_distribution(model: MEGLCSCTC):
    """
    Analyze the learned Zipf distribution and MEG prototypes.
    
    Args:
        model: Trained MEGLCSCTC model with Zipf weighting
        
    Returns:
        Dictionary with analysis results
    """
    if not model.use_zipf:
        raise ValueError("Model does not have Zipf weighting enabled")
    
    zipf_learner = model.zipf_learner
    
    # Get Zipf weights
    zipf_weights = zipf_learner.get_zipf_weights().cpu().numpy()
    
    # Get phoneme frequencies
    frequencies = (zipf_learner.phoneme_counts / zipf_learner.total_count).cpu().numpy()
    
    # Sort by frequency
    sorted_indices = np.argsort(frequencies)[::-1]
    sorted_freqs = frequencies[sorted_indices]
    sorted_zipf = zipf_weights[sorted_indices]
    
    # Calculate Zipf exponent
    zipf_s = zipf_learner.zipf_s.item()
    
    # Analyze MEG prototypes
    prototypes = zipf_learner.meg_prototypes.cpu().numpy()
    prototype_norms = np.linalg.norm(prototypes, axis=1)
    
    analysis = {
        'zipf_weights': zipf_weights,
        'phoneme_frequencies': frequencies,
        'sorted_phoneme_indices': sorted_indices,
        'sorted_frequencies': sorted_freqs,
        'sorted_zipf_weights': sorted_zipf,
        'zipf_exponent': zipf_s,
        'meg_prototype_norms': prototype_norms,
        'temperature': zipf_learner.temperature.item()
    }
    
    return analysis