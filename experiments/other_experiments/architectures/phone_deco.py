"""
Phonetic Feature Decomposition Model for MEG Phoneme Classification
Predicts articulatory features instead of raw phonemes, then maps to phonemes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np
from typing import Optional, Tuple, Dict
import torchmetrics
from torchmetrics import F1Score

class PhoneticFeatureModel(L.LightningModule):
    """
    Decompose phoneme prediction into articulatory features.
    More neurobiologically plausible and handles rare phonemes better.
    """
    
    def __init__(self,
                 meg_channels: int = 306,
                 time_points: int = 125,
                 hidden_dim: int = 256,
                 num_lstm_layers: int = 2,
                 learning_rate: float = 1e-4,
                 dropout_rate: float = 0.2):
        super().__init__()
        self.save_hyperparameters()
        
        # Build phonetic feature matrix (ground truth features for each phoneme)
        self.register_buffer('phoneme_features', self._build_phoneme_features())
        
        # Shared encoder - processes all phonemes
        self.temporal_encoder = nn.LSTM(
            meg_channels, hidden_dim, 
            num_lstm_layers, 
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate
        )
        
        # Feature dimension after bidirectional LSTM
        feature_dim = hidden_dim * 2
        
        # Temporal attention to focus on relevant time points
        self.temporal_attention = nn.MultiheadAttention(
            feature_dim, num_heads=8, batch_first=True
        )
        
        # Feature-specific decoders (like different cortical regions)
        # Consonant features
        self.manner_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 7)  # stop/fric/affric/nasal/liquid/glide/vowel
        )
        
        self.place_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 8)  # bilabial/labiodental/dental/alveolar/postalveolar/velar/glottal/NA
        )
        
        self.voicing_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 2)  # voiced/voiceless
        )
        
        # Vowel features
        self.height_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 4)  # high/mid-high/mid-low/low
        )
        
        self.backness_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 3)  # front/central/back
        )
        
        self.rounding_head = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # rounded/unrounded
        )
        
        # Vowel/Consonant classifier
        self.vowel_cons_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # vowel/consonant
        )
        
        # Metrics
        self.train_f1 = F1Score(num_classes=39, average='macro', task="multiclass")
        self.val_f1 = F1Score(num_classes=39, average='macro', task="multiclass")
        
        # Feature-level metrics for debugging
        self.manner_acc = torchmetrics.Accuracy(num_classes=7, task="multiclass")
        self.voicing_acc = torchmetrics.Accuracy(num_classes=2, task="multiclass")
        
    def _build_phoneme_features(self):
        """
        Create the feature matrix for all 39 phonemes.
        Returns a tensor of shape (39, total_features).
        """
        # Initialize feature matrix
        # Features: [manner(7), place(8), voicing(2), height(4), backness(3), rounding(2), is_vowel(2)]
        # Total: 28 features
        features = torch.zeros(39, 28)
        
        # Define phoneme features based on articulatory phonetics
        # Format: (manner, place, voicing, height, backness, rounding, is_vowel)
        # -1 means not applicable
        
        phoneme_definitions = {
            # Vowels (is_vowel=1)
            0:  (6, 7, -1, 3, 2, 0, 1),  # AA - low back unrounded
            1:  (6, 7, -1, 3, 0, 0, 1),  # AE - low front unrounded
            2:  (6, 7, -1, 2, 1, 0, 1),  # AH - mid central unrounded
            3:  (6, 7, -1, 2, 2, 1, 1),  # AO - mid back rounded
            4:  (6, 7, -1, 3, 2, 1, 1),  # AW - low back rounded
            5:  (6, 7, -1, 3, 0, 0, 1),  # AY - low front unrounded
            10: (6, 7, -1, 2, 0, 0, 1),  # EH - mid front unrounded
            11: (6, 7, -1, 2, 1, 0, 1),  # ER - mid central unrounded
            12: (6, 7, -1, 2, 0, 0, 1),  # EY - mid front unrounded
            16: (6, 7, -1, 0, 0, 0, 1),  # IH - high front unrounded
            17: (6, 7, -1, 0, 0, 0, 1),  # IY - high front unrounded
            24: (6, 7, -1, 2, 2, 1, 1),  # OW - mid back rounded
            25: (6, 7, -1, 2, 2, 1, 1),  # OY - mid back rounded
            32: (6, 7, -1, 0, 2, 0, 1),  # UH - high back unrounded
            33: (6, 7, -1, 0, 2, 1, 1),  # UW - high back rounded
            
            # Stops (manner=0)
            6:  (0, 0, 1, -1, -1, -1, 0),  # B - voiced bilabial stop
            26: (0, 0, 0, -1, -1, -1, 0),  # P - voiceless bilabial stop
            8:  (0, 3, 1, -1, -1, -1, 0),  # D - voiced alveolar stop
            30: (0, 3, 0, -1, -1, -1, 0),  # T - voiceless alveolar stop
            14: (0, 5, 1, -1, -1, -1, 0),  # G - voiced velar stop
            19: (0, 5, 0, -1, -1, -1, 0),  # K - voiceless velar stop
            
            # Fricatives (manner=1)
            13: (1, 1, 0, -1, -1, -1, 0),  # F - voiceless labiodental fricative
            34: (1, 1, 1, -1, -1, -1, 0),  # V - voiced labiodental fricative
            31: (1, 2, 0, -1, -1, -1, 0),  # TH - voiceless dental fricative
            9:  (1, 2, 1, -1, -1, -1, 0),  # DH - voiced dental fricative
            28: (1, 3, 0, -1, -1, -1, 0),  # S - voiceless alveolar fricative
            37: (1, 3, 1, -1, -1, -1, 0),  # Z - voiced alveolar fricative
            29: (1, 4, 0, -1, -1, -1, 0),  # SH - voiceless postalveolar fricative
            38: (1, 4, 1, -1, -1, -1, 0),  # ZH - voiced postalveolar fricative
            15: (1, 6, 0, -1, -1, -1, 0),  # HH - voiceless glottal fricative
            
            # Affricates (manner=2)
            7:  (2, 4, 0, -1, -1, -1, 0),  # CH - voiceless postalveolar affricate
            18: (2, 4, 1, -1, -1, -1, 0),  # JH - voiced postalveolar affricate
            
            # Nasals (manner=3)
            21: (3, 0, 1, -1, -1, -1, 0),  # M - bilabial nasal
            22: (3, 3, 1, -1, -1, -1, 0),  # N - alveolar nasal
            23: (3, 5, 1, -1, -1, -1, 0),  # NG - velar nasal
            
            # Liquids (manner=4)
            20: (4, 3, 1, -1, -1, -1, 0),  # L - alveolar lateral
            27: (4, 3, 1, -1, -1, -1, 0),  # R - alveolar approximant
            
            # Glides (manner=5)
            35: (5, 0, 1, -1, -1, -1, 0),  # W - bilabial glide
            36: (5, 4, 1, -1, -1, -1, 0),  # Y - palatal glide
        }
        
        # Convert to one-hot encoding
        for phoneme_id, (manner, place, voicing, height, backness, rounding, is_vowel) in phoneme_definitions.items():
            offset = 0
            
            # Manner (7 classes)
            if manner >= 0:
                features[phoneme_id, offset + manner] = 1
            offset += 7
            
            # Place (8 classes)
            if place >= 0:
                features[phoneme_id, offset + place] = 1
            offset += 8
            
            # Voicing (2 classes)
            if voicing >= 0:
                features[phoneme_id, offset + voicing] = 1
            offset += 2
            
            # Height (4 classes) - vowels only
            if height >= 0:
                features[phoneme_id, offset + height] = 1
            offset += 4
            
            # Backness (3 classes) - vowels only
            if backness >= 0:
                features[phoneme_id, offset + backness] = 1
            offset += 3
            
            # Rounding (2 classes) - vowels only
            if rounding >= 0:
                features[phoneme_id, offset + rounding] = 1
            offset += 2
            
            # Is vowel (2 classes)
            features[phoneme_id, offset + is_vowel] = 1
        
        return features
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract temporal features from MEG data.
        """
        B, C, T = x.shape  # (batch, channels, time)
        
        # Transpose for LSTM (batch, time, channels)
        x = x.transpose(1, 2)
        
        # Encode temporal dynamics
        encoded, _ = self.temporal_encoder(x)  # (B, T, hidden*2)
        
        # Apply temporal attention
        attended, _ = self.temporal_attention(encoded, encoded, encoded)
        
        # Pool over time (could also use attention weights)
        pooled = attended.mean(dim=1)  # (B, hidden*2)
        
        return pooled
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: MEG → features → phonemes.
        """
        # Extract features
        features = self.extract_features(x)
        
        # Predict articulatory features
        manner_logits = self.manner_head(features)
        place_logits = self.place_head(features)
        voicing_logits = self.voicing_head(features)
        height_logits = self.height_head(features)
        backness_logits = self.backness_head(features)
        rounding_logits = self.rounding_head(features)
        vowel_cons_logits = self.vowel_cons_head(features)
        
        # Convert to probabilities
        manner_probs = F.softmax(manner_logits, dim=-1)
        place_probs = F.softmax(place_logits, dim=-1)
        voicing_probs = F.softmax(voicing_logits, dim=-1)
        height_probs = F.softmax(height_logits, dim=-1)
        backness_probs = F.softmax(backness_logits, dim=-1)
        rounding_probs = F.softmax(rounding_logits, dim=-1)
        vowel_cons_probs = F.softmax(vowel_cons_logits, dim=-1)
        
        # Concatenate all feature probabilities
        all_features = torch.cat([
            manner_probs, place_probs, voicing_probs,
            height_probs, backness_probs, rounding_probs,
            vowel_cons_probs
        ], dim=-1)  # (B, 28)
        
        # Compute similarity to each phoneme's feature vector
        # Using cosine similarity for better gradient flow
        phoneme_scores = F.cosine_similarity(
            all_features.unsqueeze(1),  # (B, 1, 28)
            self.phoneme_features.unsqueeze(0),  # (1, 39, 28)
            dim=-1
        )  # (B, 39)
        
        # Scale and convert to logits
        phoneme_logits = phoneme_scores * 10  # Temperature scaling
        
        return phoneme_logits
    
    def compute_feature_loss(self, features, targets):
        """
        Compute loss directly on features for better gradient flow.
        """
        # Get ground truth features for target phonemes
        target_features = self.phoneme_features[targets]  # (B, 28)
        
        # Split features back into components
        manner_true = target_features[:, :7].argmax(dim=1)
        place_true = target_features[:, 7:15].argmax(dim=1)
        voicing_true = target_features[:, 15:17].argmax(dim=1)
        
        # Return individual feature losses (can weight differently)
        return manner_true, place_true, voicing_true
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # Get features
        features = self.extract_features(x)
        
        # Get feature predictions
        manner_logits = self.manner_head(features)
        place_logits = self.place_head(features)
        voicing_logits = self.voicing_head(features)
        
        # Get ground truth features
        manner_true, place_true, voicing_true = self.compute_feature_loss(features, y)
        
        # Feature-level losses (weighted by importance)
        manner_loss = F.cross_entropy(manner_logits, manner_true) * 1.0
        place_loss = F.cross_entropy(place_logits, place_true) * 1.0
        voicing_loss = F.cross_entropy(voicing_logits, voicing_true) * 0.5
        
        # Get phoneme predictions for overall loss
        phoneme_logits = self(x)
        phoneme_loss = F.cross_entropy(phoneme_logits, y)
        
        # Combined loss
        total_loss = phoneme_loss + 0.3 * (manner_loss + place_loss + voicing_loss)
        
        # Metrics
        self.train_f1(phoneme_logits, y)
        
        # Log everything
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('train_phoneme_loss', phoneme_loss)
        self.log('train_manner_loss', manner_loss)
        self.log('train_place_loss', place_loss)
        self.log('train_voicing_loss', voicing_loss)
        self.log('train_f1', self.train_f1, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        # Get predictions
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Metrics
        self.val_f1(logits, y)
        
        self.log('val_loss', loss)
        self.log('val_f1_macro', self.val_f1, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        # Different learning rates for different components
        params = [
            {'params': self.temporal_encoder.parameters(), 'lr': self.hparams.learning_rate},
            {'params': self.temporal_attention.parameters(), 'lr': self.hparams.learning_rate},
            # Feature heads learn faster
            {'params': self.manner_head.parameters(), 'lr': self.hparams.learning_rate * 2},
            {'params': self.place_head.parameters(), 'lr': self.hparams.learning_rate * 2},
            {'params': self.voicing_head.parameters(), 'lr': self.hparams.learning_rate * 2},
        ]
        
        optimizer = torch.optim.AdamW(params, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50, eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }