"""
Phonetic Feature Decomposition Model for MEG Phoneme Classification
Tests the hypothesis that 8 articulatory features > 39 direct classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import F1Score, ConfusionMatrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
from pathlib import Path
from datetime import datetime
import wandb

# ============================================
# PHONETIC FEATURE DEFINITIONS
# ============================================

class PhoneticFeatureSystem:
    """Complete phonetic feature system for English phonemes"""
    
    # Feature dimensions
    MANNER_DIM = 7
    PLACE_DIM = 8  
    VOICING_DIM = 2
    HEIGHT_DIM = 4
    BACKNESS_DIM = 3
    ROUNDING_DIM = 2
    
    # Phoneme to index mapping
    PHONEME_NAMES = {
        0: "AA", 1: "AE", 2: "AH", 3: "AO", 4: "AW", 5: "AY",
        6: "B", 7: "CH", 8: "D", 9: "DH", 10: "EH", 11: "ER",
        12: "EY", 13: "F", 14: "G", 15: "HH", 16: "IH", 17: "IY",
        18: "JH", 19: "K", 20: "L", 21: "M", 22: "N", 23: "NG",
        24: "OW", 25: "OY", 26: "P", 27: "R", 28: "S", 29: "SH",
        30: "T", 31: "TH", 32: "UH", 33: "UW", 34: "V", 35: "W",
        36: "Y", 37: "Z", 38: "ZH"
    }
    
    # Complete feature mapping
    # Format: (manner, place, voicing, height, backness, rounding)
    # -1 means feature not applicable (e.g., height for consonants)
    PHONEME_FEATURES = {
        # STOPS (manner=0)
        6:  (0, 0, 1, -1, -1, -1),  # B - bilabial voiced
        8:  (0, 3, 1, -1, -1, -1),  # D - alveolar voiced
        14: (0, 6, 1, -1, -1, -1),  # G - velar voiced
        26: (0, 0, 0, -1, -1, -1),  # P - bilabial voiceless
        30: (0, 3, 0, -1, -1, -1),  # T - alveolar voiceless
        19: (0, 6, 0, -1, -1, -1),  # K - velar voiceless
        
        # FRICATIVES (manner=1)
        13: (1, 1, 0, -1, -1, -1),  # F - labiodental voiceless
        34: (1, 1, 1, -1, -1, -1),  # V - labiodental voiced
        31: (1, 2, 0, -1, -1, -1),  # TH - dental voiceless (think)
        9:  (1, 2, 1, -1, -1, -1),  # DH - dental voiced (this)
        28: (1, 3, 0, -1, -1, -1),  # S - alveolar voiceless
        37: (1, 3, 1, -1, -1, -1),  # Z - alveolar voiced
        29: (1, 5, 0, -1, -1, -1),  # SH - postalveolar voiceless
        38: (1, 5, 1, -1, -1, -1),  # ZH - postalveolar voiced
        15: (1, 7, 0, -1, -1, -1),  # HH - glottal voiceless
        
        # AFFRICATES (manner=2)
        7:  (2, 5, 0, -1, -1, -1),  # CH - postalveolar voiceless
        18: (2, 5, 1, -1, -1, -1),  # JH - postalveolar voiced
        
        # NASALS (manner=3)
        21: (3, 0, 1, -1, -1, -1),  # M - bilabial
        22: (3, 3, 1, -1, -1, -1),  # N - alveolar
        23: (3, 6, 1, -1, -1, -1),  # NG - velar
        
        # LIQUIDS (manner=4)
        20: (4, 3, 1, -1, -1, -1),  # L - lateral alveolar
        27: (4, 4, 1, -1, -1, -1),  # R - retroflex
        
        # GLIDES (manner=5)
        35: (5, 0, 1, -1, -1, -1),  # W - labio-velar
        36: (5, 4, 1, -1, -1, -1),  # Y - palatal
        
        # VOWELS (manner=6)
        # Format: (6, -1, -1, height, backness, rounding)
        0:  (6, -1, -1, 3, 2, 0),  # AA - low back unrounded (bot)
        1:  (6, -1, -1, 3, 0, 0),  # AE - low front unrounded (bat)
        2:  (6, -1, -1, 2, 1, 0),  # AH - mid central (but)
        3:  (6, -1, -1, 2, 2, 1),  # AO - mid back rounded (bought)
        4:  (6, -1, -1, 3, 2, 1),  # AW - low back rounded (bout)
        5:  (6, -1, -1, 2, 0, 0),  # AY - mid front (bite)
        10: (6, -1, -1, 2, 0, 0),  # EH - mid-low front (bet)
        11: (6, -1, -1, 2, 1, 0),  # ER - mid central r-colored
        12: (6, -1, -1, 1, 0, 0),  # EY - mid-high front (bait)
        16: (6, -1, -1, 1, 0, 0),  # IH - high-mid front (bit)
        17: (6, -1, -1, 0, 0, 0),  # IY - high front (beat)
        24: (6, -1, -1, 1, 2, 1),  # OW - mid back rounded (boat)
        25: (6, -1, -1, 2, 2, 1),  # OY - mid back rounded (boy)
        32: (6, -1, -1, 1, 2, 1),  # UH - high-mid back rounded (book)
        33: (6, -1, -1, 0, 2, 1),  # UW - high back rounded (boot)
    }
    
    @classmethod
    def get_feature_vector(cls, phoneme_id):
        """Convert phoneme ID to feature vector"""
        features = cls.PHONEME_FEATURES[phoneme_id]
        vector = torch.zeros(cls.MANNER_DIM + cls.PLACE_DIM + cls.VOICING_DIM + 
                            cls.HEIGHT_DIM + cls.BACKNESS_DIM + cls.ROUNDING_DIM)
        
        idx = 0
        # Manner (one-hot)
        if features[0] >= 0:
            vector[features[0]] = 1
        idx += cls.MANNER_DIM
        
        # Place (one-hot)
        if features[1] >= 0:
            vector[idx + features[1]] = 1
        idx += cls.PLACE_DIM
        
        # Voicing (one-hot)
        if features[2] >= 0:
            vector[idx + features[2]] = 1
        idx += cls.VOICING_DIM
        
        # Height (one-hot)
        if features[3] >= 0:
            vector[idx + features[3]] = 1
        idx += cls.HEIGHT_DIM
        
        # Backness (one-hot)
        if features[4] >= 0:
            vector[idx + features[4]] = 1
        idx += cls.BACKNESS_DIM
        
        # Rounding (one-hot)
        if features[5] >= 0:
            vector[idx + features[5]] = 1
            
        return vector
    
    @classmethod
    def get_similarity_matrix(cls):
        """Create phoneme similarity matrix based on shared features"""
        n_phonemes = 39
        similarity = torch.zeros(n_phonemes, n_phonemes)
        
        for i in range(n_phonemes):
            for j in range(n_phonemes):
                if i in cls.PHONEME_FEATURES and j in cls.PHONEME_FEATURES:
                    feat_i = cls.PHONEME_FEATURES[i]
                    feat_j = cls.PHONEME_FEATURES[j]
                    
                    # Count shared features
                    shared = 0
                    total = 0
                    for fi, fj in zip(feat_i, feat_j):
                        if fi >= 0 and fj >= 0:
                            total += 1
                            if fi == fj:
                                shared += 1
                    
                    if total > 0:
                        similarity[i, j] = shared / total
                        
        return similarity

# ============================================
# EXPERIMENTAL MODEL WITH EXTENSIVE LOGGING
# ============================================

class PhoneticFeatureExperiment(L.LightningModule):
    """
    Experimental model to test phonetic feature decomposition hypothesis
    Includes extensive logging, visualization, and analysis
    """
    
    def __init__(self,
                 meg_channels: int = 306,
                 time_points: int = 125,
                 vocab_size: int = 39,
                 hidden_dim: int = 256,
                 num_conformers: int = 4,
                 learning_rate: float = 1e-4,
                 use_features: bool = True,
                 log_every_n_steps: int = 10,
                 save_confusion_every_n_epochs: int = 5,
                 dropout_rate: float = 0.2,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.save_hyperparameters()
        
        self.use_features = use_features
        self.feature_system = PhoneticFeatureSystem()
        
        # Build similarity matrix for analysis
        self.register_buffer('similarity_matrix', 
                            self.feature_system.get_similarity_matrix())
        
        # MEG Encoder (shared regardless of approach)
        self.meg_encoder = nn.Sequential(
            nn.Conv1d(meg_channels, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Temporal modeling with LSTM
        self.temporal_encoder = nn.LSTM(
            hidden_dim, hidden_dim, 
            num_layers=num_conformers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_conformers > 1 else 0
        )
        
        encoder_output_dim = hidden_dim * 2  # bidirectional
        
        if use_features:
            # Feature-based approach: Multiple heads
            self.manner_head = nn.Linear(encoder_output_dim * time_points, 
                                        PhoneticFeatureSystem.MANNER_DIM)
            self.place_head = nn.Linear(encoder_output_dim * time_points, 
                                       PhoneticFeatureSystem.PLACE_DIM)
            self.voicing_head = nn.Linear(encoder_output_dim * time_points, 
                                         PhoneticFeatureSystem.VOICING_DIM)
            self.height_head = nn.Linear(encoder_output_dim * time_points, 
                                        PhoneticFeatureSystem.HEIGHT_DIM)
            self.backness_head = nn.Linear(encoder_output_dim * time_points, 
                                          PhoneticFeatureSystem.BACKNESS_DIM)
            self.rounding_head = nn.Linear(encoder_output_dim * time_points, 
                                         PhoneticFeatureSystem.ROUNDING_DIM)
            
            # Constraint layer to combine features into phonemes
            self.phoneme_decoder = nn.Linear(
                PhoneticFeatureSystem.MANNER_DIM + PhoneticFeatureSystem.PLACE_DIM + 
                PhoneticFeatureSystem.VOICING_DIM + PhoneticFeatureSystem.HEIGHT_DIM + 
                PhoneticFeatureSystem.BACKNESS_DIM + PhoneticFeatureSystem.ROUNDING_DIM,
                vocab_size
            )
            
            # Initialize constraint layer with known mappings
            self._initialize_constraints()
            
        else:
            # Direct approach: Single classifier
            self.direct_classifier = nn.Sequential(
                nn.Linear(encoder_output_dim * time_points, 512),
                nn.ReLU(),
                nn.Dropout(dropout_rate + 0.1),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, vocab_size)
            )
        
        # Metrics
        self.train_f1 = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
        self.val_f1 = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
        self.confusion_matrix = ConfusionMatrix(num_classes=vocab_size, task="multiclass")
        
        # Tracking for analysis
        self.training_history = defaultdict(list)
        self.feature_activations = defaultdict(list)
        self.phoneme_confusions = defaultdict(lambda: defaultdict(int))
        
    def _initialize_constraints(self):
        """Initialize phoneme decoder with linguistic constraints"""
        with torch.no_grad():
            # Start with very negative weights (impossible combinations)
            self.phoneme_decoder.weight.fill_(-10.0)
            
            # Set positive weights for valid feature combinations
            for phoneme_id, features in PhoneticFeatureSystem.PHONEME_FEATURES.items():
                feature_vec = PhoneticFeatureSystem.get_feature_vector(phoneme_id)
                self.phoneme_decoder.weight[phoneme_id] = feature_vec * 5.0
    
    def extract_features(self, x):
        """Extract features from MEG data"""
        B, C, T = x.shape
        
        # Spatial encoding
        x_encoded = self.meg_encoder(x)  # (B, hidden_dim, T)
        
        # Temporal encoding
        x_encoded = x_encoded.transpose(1, 2)  # (B, T, hidden_dim)
        x_temporal, _ = self.temporal_encoder(x_encoded)  # (B, T, hidden_dim*2)
        
        return x_temporal
    
    def forward(self, x):
        """Forward pass"""
        features = self.extract_features(x)  # (B, T, D)
        B, T, D = features.shape
        features_flat = features.reshape(B, -1)  # (B, T*D)
        
        if self.use_features:
            # Predict individual features
            manner = self.manner_head(features_flat)
            place = self.place_head(features_flat)
            voicing = self.voicing_head(features_flat)
            height = self.height_head(features_flat)
            backness = self.backness_head(features_flat)
            rounding = self.rounding_head(features_flat)
            
            # Combine features
            all_features = torch.cat([manner, place, voicing, height, backness, rounding], dim=1)
            
            # Decode to phonemes
            phoneme_logits = self.phoneme_decoder(all_features)
            
            # Store feature activations for analysis
            if self.training:
                self.feature_activations['manner'].append(manner.detach().cpu())
                self.feature_activations['place'].append(place.detach().cpu())
                self.feature_activations['voicing'].append(voicing.detach().cpu())
            
            return phoneme_logits, {
                'manner': manner,
                'place': place,
                'voicing': voicing,
                'height': height,
                'backness': backness,
                'rounding': rounding
            }
        else:
            # Direct classification
            phoneme_logits = self.direct_classifier(features_flat)
            return phoneme_logits, None
    
    def compute_feature_losses(self, features, targets):
        """Compute auxiliary losses for individual features"""
        feature_losses = {}
        B = targets.shape[0]
        device = targets.device
        
        # Create target tensors for each feature
        manner_targets = torch.full((B,), -1, dtype=torch.long, device=device)
        place_targets = torch.full((B,), -1, dtype=torch.long, device=device)
        voicing_targets = torch.full((B,), -1, dtype=torch.long, device=device)
        height_targets = torch.full((B,), -1, dtype=torch.long, device=device)
        backness_targets = torch.full((B,), -1, dtype=torch.long, device=device)
        rounding_targets = torch.full((B,), -1, dtype=torch.long, device=device)
        
        # Fill in targets based on phoneme features
        for i, phoneme_id in enumerate(targets):
            phoneme_id = phoneme_id.item()
            if phoneme_id in PhoneticFeatureSystem.PHONEME_FEATURES:
                feat = PhoneticFeatureSystem.PHONEME_FEATURES[phoneme_id]
                if feat[0] >= 0: manner_targets[i] = feat[0]
                if feat[1] >= 0: place_targets[i] = feat[1]
                if feat[2] >= 0: voicing_targets[i] = feat[2]
                if feat[3] >= 0: height_targets[i] = feat[3]
                if feat[4] >= 0: backness_targets[i] = feat[4]
                if feat[5] >= 0: rounding_targets[i] = feat[5]
        
        # Compute losses for features with valid targets
        if 'manner' in features:
            valid_manner = manner_targets >= 0
            if valid_manner.any():
                feature_losses['manner'] = F.cross_entropy(
                    features['manner'][valid_manner],
                    manner_targets[valid_manner]
                )
        
        if 'place' in features:
            valid_place = place_targets >= 0
            if valid_place.any():
                feature_losses['place'] = F.cross_entropy(
                    features['place'][valid_place],
                    place_targets[valid_place]
                )
        
        if 'voicing' in features:
            valid_voicing = voicing_targets >= 0
            if valid_voicing.any():
                feature_losses['voicing'] = F.cross_entropy(
                    features['voicing'][valid_voicing],
                    voicing_targets[valid_voicing]
                )
        
        if 'height' in features:
            valid_height = height_targets >= 0
            if valid_height.any():
                feature_losses['height'] = F.cross_entropy(
                    features['height'][valid_height],
                    height_targets[valid_height]
                )
        
        if 'backness' in features:
            valid_backness = backness_targets >= 0
            if valid_backness.any():
                feature_losses['backness'] = F.cross_entropy(
                    features['backness'][valid_backness],
                    backness_targets[valid_backness]
                )
        
        if 'rounding' in features:
            valid_rounding = rounding_targets >= 0
            if valid_rounding.any():
                feature_losses['rounding'] = F.cross_entropy(
                    features['rounding'][valid_rounding],
                    rounding_targets[valid_rounding]
                )
        
        return feature_losses
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, features = self(x)
        
        # Main loss
        main_loss = F.cross_entropy(logits, y, label_smoothing=self.hparams.label_smoothing)
        
        # Feature auxiliary losses (if using features)
        total_loss = main_loss
        if features is not None:
            feature_losses = self.compute_feature_losses(features, y)
            for feat_name, feat_loss in feature_losses.items():
                total_loss = total_loss + 0.5 * feat_loss
                self.log(f'train/feature_loss_{feat_name}', feat_loss)
        
        # Metrics
        f1 = self.train_f1(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        
        # Extensive logging
        self.log('train/loss', main_loss, prog_bar=True)
        self.log('train/total_loss', total_loss)
        self.log('train/f1', f1)
        self.log('train/acc', acc)
        
        # Track predictions for confusion analysis
        if batch_idx % self.hparams.log_every_n_steps == 0:
            preds = logits.argmax(dim=-1)
            for pred, true in zip(preds, y):
                pred_name = PhoneticFeatureSystem.PHONEME_NAMES[pred.item()]
                true_name = PhoneticFeatureSystem.PHONEME_NAMES[true.item()]
                self.phoneme_confusions[true_name][pred_name] += 1
        
        # Log feature statistics
        if features is not None and batch_idx % 50 == 0:
            for feat_name, feat_tensor in features.items():
                if feat_tensor is not None:
                    probs = F.softmax(feat_tensor, dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
                    self.log(f'train/feature_entropy_{feat_name}', entropy)
                    
                    # Log distribution
                    pred_dist = feat_tensor.argmax(dim=-1)
                    for i in range(feat_tensor.shape[1]):
                        count = (pred_dist == i).sum().float()
                        self.log(f'train/{feat_name}_class_{i}_count', count)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, features = self(x)
        
        loss = F.cross_entropy(logits, y)
        f1 = self.val_f1(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        
        self.log('val_loss', loss)
        self.log('val_f1_macro', f1, prog_bar=True)
        self.log('val/acc', acc)
        
        # Update confusion matrix
        self.confusion_matrix.update(logits, y)
        
        return loss
    
    def on_validation_epoch_end(self):
        """Generate and save confusion matrix and analysis"""
        # Get confusion matrix
        cm = self.confusion_matrix.compute().cpu().numpy()
        self.confusion_matrix.reset()
        
        # Save confusion matrix plot
        if self.current_epoch % self.hparams.save_confusion_every_n_epochs == 0:
            self.save_confusion_matrix(cm)
            self.analyze_feature_clustering(cm)
    
    def save_confusion_matrix(self, cm):
        """Save confusion matrix with phoneme labels"""
        plt.figure(figsize=(20, 18))
        
        # Normalize by row (true labels)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create labels
        labels = [PhoneticFeatureSystem.PHONEME_NAMES[i] for i in range(39)]
        
        # Plot
        sns.heatmap(cm_normalized, annot=False, cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix - Epoch {self.current_epoch} ({"Features" if self.use_features else "Direct"})')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        
        # Save
        save_path = Path(f'confusion_matrices_features/epoch_{self.current_epoch}_{"features" if self.use_features else "direct"}.png')
        save_path.parent.mkdir(exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        
        # Log to wandb
        if wandb.run is not None:
            wandb.log({f'confusion_matrix_epoch_{self.current_epoch}': wandb.Image(str(save_path))})
        
        print(f"\n{'='*60}")
        print(f"Confusion matrix saved to {save_path}")
        print(f"{'='*60}\n")
    
    def analyze_feature_clustering(self, cm):
        """Analyze if confusions follow phonetic feature patterns"""
        print(f"\n{'='*60}")
        print(f"FEATURE CLUSTERING ANALYSIS - Epoch {self.current_epoch}")
        print(f"{'='*60}")
        
        # Expected confusion pairs based on shared features
        expected_confusions = {
            'ZH-SH': (38, 29),  # Same place+manner, diff voicing
            'CH-JH': (7, 18),   # Same place+manner, diff voicing
            'S-Z': (28, 37),    # Same place+manner, diff voicing
            'F-V': (13, 34),    # Same place+manner, diff voicing
            'P-B': (26, 6),     # Same place+manner, diff voicing
            'T-D': (30, 8),     # Same place+manner, diff voicing
            'K-G': (19, 14),    # Same place+manner, diff voicing
        }
        
        analysis_results = {}
        for pair_name, (i, j) in expected_confusions.items():
            # Bidirectional confusion rate
            confusion_rate = (cm[i, j] + cm[j, i]) / (cm[i, i] + cm[j, j] + cm[i, j] + cm[j, i] + 1e-10)
            
            # Compare to random confusion rate
            total_confusions = cm.sum() - np.diag(cm).sum()
            random_rate = total_confusions / (cm.sum() + 1e-10)
            
            ratio = confusion_rate / (random_rate + 1e-10)
            
            analysis_results[pair_name] = {
                'confusion_rate': confusion_rate,
                'random_rate': random_rate,
                'ratio': ratio,
                'follows_theory': ratio > 2.0  # Confusion 2x more likely than random
            }
            
            print(f"{pair_name}: Confusion={confusion_rate:.3f}, Random={random_rate:.3f}, Ratio={ratio:.2f} {'✓' if ratio > 2.0 else '✗'}")
        
        # Log to wandb
        if wandb.run is not None:
            wandb.log({f'feature_clustering_epoch_{self.current_epoch}': analysis_results})
        
        # Calculate overall feature clustering score
        theory_score = sum(1 for r in analysis_results.values() if r['follows_theory']) / len(analysis_results)
        print(f"\nFeature Theory Score: {theory_score:.1%} of expected confusions observed")
        print(f"{'='*60}\n")
        
        return analysis_results
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits, features = self(x)
        
        loss = F.cross_entropy(logits, y)
        f1 = self.val_f1(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        
        self.log('test/loss', loss)
        self.log('test/f1', f1)
        self.log('test/acc', acc)
        
        # Per-phoneme F1
        preds = logits.argmax(dim=-1)
        for phoneme_id in range(39):
            mask = y == phoneme_id
            if mask.sum() > 0:
                phoneme_f1 = ((preds[mask] == phoneme_id).float().mean())
                phoneme_name = PhoneticFeatureSystem.PHONEME_NAMES[phoneme_id]
                self.log(f'test/f1_{phoneme_name}', phoneme_f1)
        
        return loss
    
    def configure_optimizers(self):
        # Different learning rates for different components
        params = []
        
        # Encoder parameters
        params.append({'params': self.meg_encoder.parameters(), 
                      'lr': self.hparams.learning_rate})
        params.append({'params': self.temporal_encoder.parameters(), 
                      'lr': self.hparams.learning_rate})
        
        if self.use_features:
            # Feature heads - higher learning rate
            params.extend([
                {'params': self.manner_head.parameters(), 'lr': self.hparams.learning_rate * 2},
                {'params': self.place_head.parameters(), 'lr': self.hparams.learning_rate * 2},
                {'params': self.voicing_head.parameters(), 'lr': self.hparams.learning_rate * 2},
                {'params': self.height_head.parameters(), 'lr': self.hparams.learning_rate * 2},
                {'params': self.backness_head.parameters(), 'lr': self.hparams.learning_rate * 2},
                {'params': self.rounding_head.parameters(), 'lr': self.hparams.learning_rate * 2},
            ])
            # Constraint layer - lower learning rate
            params.append({'params': self.phoneme_decoder.parameters(), 
                         'lr': self.hparams.learning_rate * 0.1})
        else:
            params.append({'params': self.direct_classifier.parameters(), 
                         'lr': self.hparams.learning_rate})
        
        optimizer = torch.optim.AdamW(params, weight_decay=0.01)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }
    
    def on_train_end(self):
        """Generate final report"""
        self.generate_final_report()
    
    def generate_final_report(self):
        """Generate comprehensive analysis report"""
        report_dir = Path(f'experiment_reports/{datetime.now().strftime("%Y%m%d_%H%M%S")}_{"features" if self.use_features else "direct"}')
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report = {
            'model_type': 'Phonetic Features' if self.use_features else 'Direct Classification',
            'hyperparameters': dict(self.hparams),
            'final_metrics': {
                'val_f1': self.trainer.callback_metrics.get('val_f1_macro', 0).item(),
                'val_acc': self.trainer.callback_metrics.get('val/acc', 0).item(),
            },
            'confusion_analysis': self.phoneme_confusions,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save JSON report
        with open(report_dir / 'report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate human-readable report
        with open(report_dir / 'summary.txt', 'w') as f:
            f.write("="*60 + "\n")
            f.write("PHONETIC FEATURE EXPERIMENT FINAL REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Model Type: {report['model_type']}\n")
            f.write(f"Final Validation F1: {report['final_metrics']['val_f1']:.4f}\n")
            f.write(f"Final Validation Accuracy: {report['final_metrics']['val_acc']:.4f}\n\n")
            
            f.write("Top Confusions:\n")
            for true_phoneme, confusions in self.phoneme_confusions.items():
                if confusions:
                    top_confusion = max(confusions.items(), key=lambda x: x[1])
                    f.write(f"  {true_phoneme} → {top_confusion[0]} ({top_confusion[1]} times)\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write("HYPOTHESIS TEST RESULT:\n")
            if self.use_features:
                f.write("IF this model beats the direct baseline by >5% F1,\n")
                f.write("THEN phonetic features are the key to breaking the 0.4 ceiling.\n")
            else:
                f.write("This is the BASELINE for comparison.\n")
            f.write("="*60 + "\n")
        
        print(f"\n{'='*60}")
        print(f"Final report saved to {report_dir}")
        print(f"{'='*60}\n")
        
        return report_dir