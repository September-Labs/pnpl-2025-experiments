"""
Two-Step MEG-based Phoneme Classification with Sequential Training
First trains broad category classifier, then trains expert models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import lightning as L
except ImportError:
    import pytorch_lightning as L
import numpy as np
from typing import Optional, Dict, List, Tuple
from torchmetrics import F1Score, Accuracy, ConfusionMatrix
from collections import defaultdict, Counter
import warnings

# ============================================
# Phoneme Category Definitions (ARPAbet)
# ============================================

# Define phoneme categories based on linguistic features
PHONEME_CATEGORIES = {
    'vowels': [
        'aa', 'ae', 'ah', 'ao', 'aw', 'ax-h', 'ax', 'axr', 'ay', 
        'eh', 'el', 'em', 'en', 'eng', 'er', 'ey', 
        'ih', 'ix', 'iy', 'ow', 'oy', 'uh', 'uw', 'ux'
    ],
    'stops': [
        'b', 'bcl', 'd', 'dcl', 'g', 'gcl', 'k', 'kcl', 'p', 'pcl', 't', 'tcl', 'q'
    ],
    'fricatives': [
        'ch', 'dh', 'f', 'hh', 'hv', 'jh', 's', 'sh', 'th', 'v', 'z', 'zh'
    ],
    'nasals': [
        'm', 'n', 'ng', 'nx'
    ],
    'liquids': [
        'l', 'r', 'dx', 'w', 'y'
    ]
}

# Create reverse mapping from phoneme to category
PHONEME_TO_CATEGORY = {}
for category, phonemes in PHONEME_CATEGORIES.items():
    for phoneme in phonemes:
        PHONEME_TO_CATEGORY[phoneme] = category

# Category to index mapping
CATEGORY_TO_IDX = {cat: idx for idx, cat in enumerate(PHONEME_CATEGORIES.keys())}
IDX_TO_CATEGORY = {idx: cat for cat, idx in CATEGORY_TO_IDX.items()}

# ============================================
# Shared MEG Feature Extractor
# ============================================

class MEGFeatureExtractor(nn.Module):
    """Shared MEG feature extractor for both stages."""
    
    def __init__(self, meg_channels=306, hidden_dim=256, dropout=0.1):
        super().__init__()
        
        # Multi-scale convolutional feature extraction
        self.conv1 = nn.Conv1d(meg_channels, hidden_dim//2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(meg_channels, hidden_dim//2, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(meg_channels, hidden_dim//2, kernel_size=7, padding=3)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim//2)
        self.bn2 = nn.BatchNorm1d(hidden_dim//2)
        self.bn3 = nn.BatchNorm1d(hidden_dim//2)
        
        # Combine multi-scale features
        self.combine_conv = nn.Conv1d((hidden_dim//2)*3, hidden_dim, kernel_size=1)
        self.bn_combine = nn.BatchNorm1d(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (B, channels, time_points)
        
        # Multi-scale feature extraction
        feat1 = F.relu(self.bn1(self.conv1(x)))
        feat2 = F.relu(self.bn2(self.conv2(x)))
        feat3 = F.relu(self.bn3(self.conv3(x)))
        
        # Concatenate multi-scale features
        multi_scale = torch.cat([feat1, feat2, feat3], dim=1)
        
        # Combine features
        combined = F.relu(self.bn_combine(self.combine_conv(multi_scale)))
        combined = self.dropout(combined)
        
        return combined  # (B, hidden_dim, T)

# ============================================
# Temporal Modeling Module
# ============================================

class TemporalEncoder(nn.Module):
    """Temporal modeling with LSTM/GRU for sequence processing."""
    
    def __init__(self, hidden_dim=256, num_layers=2, dropout=0.1):
        super().__init__()
        
        self.lstm = nn.LSTM(
            hidden_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Project back to original dimension
        self.projection = nn.Linear(hidden_dim * 2, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        # x: (B, T, hidden_dim)
        lstm_out, _ = self.lstm(x)
        out = self.projection(lstm_out)
        out = self.layer_norm(out)
        return out

# ============================================
# Broad Category Classifier
# ============================================

class BroadCategoryClassifier(nn.Module):
    """First stage: classify into 5 broad phoneme categories."""
    
    def __init__(self, hidden_dim=256, num_categories=5, dropout=0.2):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            hidden_dim, 
            num_heads=8, 
            dropout=dropout,
            batch_first=True
        )
        
        # Enhanced classifier with residual connection
        self.pre_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Concatenate with residual
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, num_categories)
        )
        
        # For CTC
        self.ctc_projection = nn.Linear(hidden_dim, num_categories + 1)  # +1 for blank
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier/He initialization for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, features, use_ctc=False):
        # features: (B, T, hidden_dim)
        
        # Self-attention for global context
        attn_out, _ = self.attention(features, features, features)
        
        if use_ctc:
            # CTC output for each time step
            logits = self.ctc_projection(attn_out)  # (B, T, num_categories+1)
            return logits
        else:
            # Global pooling for single prediction
            pooled = torch.mean(attn_out, dim=1)  # (B, hidden_dim)
            
            # Enhanced classification with residual
            pre_class = self.pre_classifier(pooled)
            combined = torch.cat([pooled, pre_class], dim=-1)  # Residual connection
            logits = self.classifier(combined)  # (B, num_categories)
            return logits

# ============================================
# Expert Models for Each Category
# ============================================

class ExpertModel(nn.Module):
    """Expert model for fine-grained classification within a category."""
    
    def __init__(self, hidden_dim=256, num_phonemes=10, category_name="", dropout=0.2):
        super().__init__()
        self.category_name = category_name
        
        # Category-specific attention
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Category-specific temporal modeling
        self.temporal = nn.GRU(
            hidden_dim,
            hidden_dim//2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Phoneme classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, num_phonemes)
        )
        
        # For CTC
        self.ctc_projection = nn.Linear(hidden_dim, num_phonemes + 1)
        
    def forward(self, features, use_ctc=False):
        # Apply attention
        attn_out, _ = self.attention(features, features, features)
        
        # Apply temporal modeling
        temporal_out, _ = self.temporal(attn_out)
        
        if use_ctc:
            logits = self.ctc_projection(temporal_out)
            return logits
        else:
            # Global pooling
            pooled = torch.mean(temporal_out, dim=1)
            logits = self.classifier(pooled)
            return logits

# ============================================
# Sequential Two-Step MEG Phoneme Classifier
# ============================================

class MEGSequentialTwoStep(L.LightningModule):
    """Sequential two-step phoneme classification: train broad categories first, then expert models."""
    
    def __init__(self,
                 meg_channels=306,
                 time_points=125,
                 vocab_size=39,
                 hidden_dim=256,
                 num_temporal_layers=2,
                 learning_rate=1e-4,
                 use_ctc=True,
                 ctc_weight=0.3,
                 training_stage=1,  # 1 for category, 2 for experts
                 freeze_backbone_stage2=True,  # Freeze feature extractor in stage 2
                 stage1_checkpoint_path=None,  # Path to load Stage 1 weights
                 use_class_weights=True):  # Auto-calculate class weights for imbalanced data
        super().__init__()
        self.save_hyperparameters()
        
        # Build phoneme to index mapping for the 39 classes
        self.phoneme_to_idx = {}
        self.idx_to_phoneme = []
        
        # Feature extraction (shared)
        self.feature_extractor = MEGFeatureExtractor(meg_channels, hidden_dim)
        self.temporal_encoder = TemporalEncoder(hidden_dim, num_temporal_layers)
        
        # Broad category classifier
        self.category_classifier = BroadCategoryClassifier(
            hidden_dim, 
            num_categories=len(PHONEME_CATEGORIES)
        )
        
        # Expert models for each category
        self.experts = nn.ModuleDict()
        self.category_phoneme_mapping = {}
        
        # Initialize expert models and mappings
        self._initialize_experts(hidden_dim)

        self.verify_phoneme_mapping()
        
        # Loss functions
        # Calculate class weights for Stage 1 (based on typical phoneme distribution)
        if training_stage == 1 and use_class_weights:
            # These weights help balance the classes
            # Higher weights for rarer categories (fricatives, nasals, liquids)
            category_weights = torch.tensor([
                0.6,   # vowels (slightly up)
                2.0,   # stops (up)
                0.8,   # fricatives (DOWN - they're working too well!)
                2.5,   # nasals (up)
                3.0    # liquids (keep high)
            ])
            self.ce_loss = nn.CrossEntropyLoss(weight=category_weights)
            print(f"Using weighted loss for categories (vowels, stops, fricatives, nasals, liquids):")
            print(f"  Weights: {category_weights.tolist()}")
        else:
            self.ce_loss = nn.CrossEntropyLoss()
        
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        
        # Metrics for categories (5-class problem)
        self.train_cat_accuracy = Accuracy(num_classes=5, task="multiclass")
        self.val_cat_accuracy = Accuracy(num_classes=5, task="multiclass")
        self.train_cat_f1 = F1Score(num_classes=5, average='macro', task="multiclass")
        self.val_cat_f1 = F1Score(num_classes=5, average='macro', task="multiclass")
        
        # Metrics for phonemes (39-class problem)
        self.train_f1 = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
        self.val_f1 = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
        
        # For tracking category confusion
        self.val_cat_confusion = ConfusionMatrix(num_classes=5, task="multiclass")
        
        # Data verification counters
        self.phoneme_counter = Counter()
        self.category_counter = Counter()
        self.batch_count = 0
        
        # Load Stage 1 checkpoint if provided (for Stage 2 training)
        if stage1_checkpoint_path and training_stage == 2:
            self._load_stage1_weights(stage1_checkpoint_path)
        
        print("\n" + "="*80)
        print(f"INITIALIZED SEQUENTIAL TWO-STEP MODEL - STAGE {training_stage}")
        print("="*80)
        print(f"Training Stage: {'CATEGORY CLASSIFICATION' if training_stage == 1 else 'EXPERT MODELS'}")
        print(f"Freeze backbone in stage 2: {freeze_backbone_stage2}")
        if stage1_checkpoint_path and training_stage == 2:
            print(f"Loaded Stage 1 weights from: {stage1_checkpoint_path}")
        print("\nCategory Distribution:")
        for cat, phonemes in PHONEME_CATEGORIES.items():
            print(f"  {cat}: {len(phonemes)} phonemes")
        print("="*80 + "\n")
    
    def verify_phoneme_mapping(self):
        """Verify all dataset phonemes are correctly categorized."""
        print("\nPHONEME TO CATEGORY MAPPING VERIFICATION:")
        print("="*50)
        
        issues = []
        for idx, phoneme in enumerate(self.idx_to_phoneme):
            if phoneme in PHONEME_TO_CATEGORY:
                category = PHONEME_TO_CATEGORY[phoneme]
                print(f"  {idx:2d}: '{phoneme:4s}' → {category}")
            else:
                issues.append((idx, phoneme))
                print(f"  {idx:2d}: '{phoneme:4s}' → UNCATEGORIZED! (defaulting to vowels)")
        
        if issues:
            print(f"\n⚠️  WARNING: {len(issues)} phonemes not properly categorized!")
            print("These need to be added to PHONEME_CATEGORIES:")
            for idx, phoneme in issues:
                print(f"  - '{phoneme}'")
        else:
            print("\n✓ All phonemes properly categorized!")
        
        print("="*50)

    def _initialize_experts(self, hidden_dim):
        """Initialize expert models for each phoneme category."""
        # Get the 39 phonemes used in the dataset
        try:
            from pnpl.datasets.libribrain2025.constants import PHONEMES
            used_phonemes = PHONEMES[:39]  # First 39 phonemes
        except ImportError:
            # Fallback to a default list if import fails
            print("WARNING: Could not import PHONEMES, using default list")
            used_phonemes = ['aa', 'ae', 'ah', 'ao', 'aw', 'ay', 'b', 'ch', 'd', 
                           'dh', 'dx', 'eh', 'el', 'en', 'er', 'ey', 'f', 'g', 
                           'hh', 'ih', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 
                           'ow', 'oy', 'p', 'r', 's', 'sh', 't', 'th', 'uh', 
                           'uw', 'v', 'w']
        
        # Create index mappings
        for idx, phoneme in enumerate(used_phonemes):
            self.phoneme_to_idx[phoneme] = idx
        self.idx_to_phoneme = used_phonemes
        
        # Group phonemes by category
        category_phonemes = defaultdict(list)
        uncategorized = []
        
        for phoneme in used_phonemes:
            if phoneme in PHONEME_TO_CATEGORY:
                category = PHONEME_TO_CATEGORY[phoneme]
                category_phonemes[category].append(phoneme)
            else:
                uncategorized.append(phoneme)
        
        if uncategorized:
            print(f"WARNING: {len(uncategorized)} phonemes not categorized: {uncategorized}")
            # Add uncategorized to most similar category (default to vowels)
            category_phonemes['vowels'].extend(uncategorized)
        
        # Create expert model for each category
        print("\nExpert Model Initialization:")
        for category, phonemes in category_phonemes.items():
            if phonemes:  # Only create expert if category has phonemes
                num_phonemes = len(phonemes)
                self.experts[category] = ExpertModel(
                    hidden_dim, 
                    num_phonemes, 
                    category
                )
                
                # Store mapping from category-specific index to global index
                self.category_phoneme_mapping[category] = {
                    'phonemes': phonemes,
                    'local_to_global': {i: self.phoneme_to_idx[p] for i, p in enumerate(phonemes)},
                    'global_to_local': {self.phoneme_to_idx[p]: i for i, p in enumerate(phonemes)}
                }
                
                print(f"  {category}: {num_phonemes} phonemes - {phonemes[:5]}{'...' if len(phonemes) > 5 else ''}")
    
    def _load_stage1_weights(self, checkpoint_path):
        """Load weights from Stage 1 checkpoint."""
        print(f"\nLoading Stage 1 weights from: {checkpoint_path}")
        
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        
        # Load weights for shared components
        # The state dict keys will have the module names
        loaded_components = []
        
        for key in state_dict.keys():
            if key.startswith('feature_extractor.'):
                self.feature_extractor.load_state_dict(
                    {k.replace('feature_extractor.', ''): v 
                     for k, v in state_dict.items() 
                     if k.startswith('feature_extractor.')},
                    strict=True
                )
                if 'feature_extractor' not in loaded_components:
                    loaded_components.append('feature_extractor')
                    
            elif key.startswith('temporal_encoder.'):
                self.temporal_encoder.load_state_dict(
                    {k.replace('temporal_encoder.', ''): v 
                     for k, v in state_dict.items() 
                     if k.startswith('temporal_encoder.')},
                    strict=True
                )
                if 'temporal_encoder' not in loaded_components:
                    loaded_components.append('temporal_encoder')
                    
            elif key.startswith('category_classifier.'):
                self.category_classifier.load_state_dict(
                    {k.replace('category_classifier.', ''): v 
                     for k, v in state_dict.items() 
                     if k.startswith('category_classifier.')},
                    strict=True
                )
                if 'category_classifier' not in loaded_components:
                    loaded_components.append('category_classifier')
        
        print(f"Successfully loaded Stage 1 components: {', '.join(loaded_components)}")
        
        # Get Stage 1 metrics if available
        if 'callbacks' in checkpoint:
            for callback_state in checkpoint['callbacks'].values():
                if 'best_model_score' in callback_state:
                    best_score = callback_state['best_model_score']
                    if not torch.isnan(best_score) and not torch.isinf(best_score):
                        print(f"Stage 1 best validation score: {best_score:.4f}")
                        break
    
    def forward(self, x, training_stage=None):
        """
        Forward pass through the two-step model.
        
        Args:
            x: MEG data (B, channels, time_points)
            training_stage: Override training stage (1 or 2)
            
        Returns:
            Depends on training stage
        """
        if training_stage is None:
            training_stage = self.hparams.training_stage
            
        B, C, T = x.shape
        
        # Extract features
        features = self.feature_extractor(x)  # (B, hidden_dim, T)
        features = features.transpose(1, 2)  # (B, T, hidden_dim)
        
        # Temporal encoding
        temporal_features = self.temporal_encoder(features)  # (B, T, hidden_dim)
        
        if training_stage == 1:
            # Stage 1: Only return category predictions
            category_logits = self.category_classifier(temporal_features, use_ctc=False)
            return category_logits
        
        else:  # training_stage == 2
            # Stage 2: Full phoneme classification
            
            # Get category predictions (frozen in stage 2)
            with torch.no_grad() if self.hparams.freeze_backbone_stage2 else torch.enable_grad():
                category_logits = self.category_classifier(temporal_features, use_ctc=False)
            
            category_probs = F.softmax(category_logits, dim=-1)
            
            # Expert models for each category
            phoneme_logits = torch.zeros(B, self.hparams.vocab_size, device=x.device)
            
            for cat_idx, category in enumerate(IDX_TO_CATEGORY.values()):
                if category not in self.experts:
                    continue
                    
                # Get expert predictions
                expert_logits = self.experts[category](temporal_features, use_ctc=False)
                
                # Weight by category probability
                cat_weight = category_probs[:, cat_idx].unsqueeze(1)  # (B, 1)
                
                # Map expert predictions to global phoneme indices
                mapping = self.category_phoneme_mapping[category]
                for local_idx, global_idx in mapping['local_to_global'].items():
                    phoneme_logits[:, global_idx] = cat_weight[:, 0] * expert_logits[:, local_idx]
            
            return phoneme_logits, category_logits
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # Data augmentation for Stage 1 (add noise to help generalization)
        if self.hparams.training_stage == 1 and self.training:
            noise_level = 0.05
            x = x + torch.randn_like(x) * noise_level
        
        # Data verification (first few batches)
        if self.batch_count < 5:
            self._verify_data(y, batch_idx)
        self.batch_count += 1
        
        if self.hparams.training_stage == 1:
            # STAGE 1: Train category classifier only
            category_logits = self(x, training_stage=1)
            category_labels = self._get_category_labels(y)
            
            loss = self.ce_loss(category_logits, category_labels)
            
            # Metrics
            cat_acc = self.train_cat_accuracy(category_logits, category_labels)
            cat_f1 = self.train_cat_f1(category_logits, category_labels)
            
            # Detailed logging every 50 batches
            if batch_idx % 50 == 0:
                self._print_category_performance(category_logits, category_labels, y, "Train")
            
            # Standard logging - use generic names for compatibility
            self.log('train_loss', loss, prog_bar=True)
            self.log('train_acc', cat_acc, prog_bar=True)
            self.log('train_f1_macro', cat_f1)
            
        else:  # self.hparams.training_stage == 2
            # STAGE 2: Train expert models
            phoneme_logits, category_logits = self(x, training_stage=2)
            
            # Phoneme loss (main objective)
            phoneme_loss = self.ce_loss(phoneme_logits, y)
            
            # Optional: auxiliary category loss (shouldn't change much if frozen)
            category_labels = self._get_category_labels(y)
            category_loss = self.ce_loss(category_logits, category_labels)
            
            # Combined loss
            loss = phoneme_loss + 0.1 * category_loss  # Small weight for category
            
            # Metrics
            f1 = self.train_f1(phoneme_logits, y)
            cat_acc = self.train_cat_accuracy(category_logits, category_labels)
            
            # Detailed logging every 50 batches
            if batch_idx % 50 == 0:
                self._print_expert_performance(phoneme_logits, y, category_logits, category_labels, "Train")
            
            # Standard logging - use generic names for compatibility
            self.log('train_loss', loss, prog_bar=True)
            self.log('train_phoneme_loss', phoneme_loss)
            self.log('train_category_loss', category_loss)
            self.log('train_f1_macro', f1, prog_bar=True)
            self.log('train_cat_acc', cat_acc)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        if self.hparams.training_stage == 1:
            # STAGE 1: Validate category classifier
            category_logits = self(x, training_stage=1)
            category_labels = self._get_category_labels(y)
            
            loss = self.ce_loss(category_logits, category_labels)
            
            # Metrics
            cat_acc = self.val_cat_accuracy(category_logits, category_labels)
            cat_f1 = self.val_cat_f1(category_logits, category_labels)
            self.val_cat_confusion.update(category_logits, category_labels)
            
            # Logging - use generic names for compatibility with checkpointing
            self.log('val_loss', loss, prog_bar=True)
            self.log('val_acc', cat_acc, prog_bar=True)
            self.log('val_f1_macro', cat_f1)  # This is what checkpoint monitors
            
        else:  # self.hparams.training_stage == 2
            # STAGE 2: Validate expert models
            phoneme_logits, category_logits = self(x, training_stage=2)
            
            phoneme_loss = self.ce_loss(phoneme_logits, y)
            category_labels = self._get_category_labels(y)
            category_loss = self.ce_loss(category_logits, category_labels)
            
            loss = phoneme_loss + 0.1 * category_loss
            
            # Metrics
            f1 = self.val_f1(phoneme_logits, y)
            cat_acc = self.val_cat_accuracy(category_logits, category_labels)
            
            # Logging - use generic names for compatibility
            self.log('val_loss', loss, prog_bar=True)
            self.log('val_phoneme_loss', phoneme_loss)
            self.log('val_f1_macro', f1, prog_bar=True)  # This is what checkpoint monitors
            self.log('val_cat_acc', cat_acc)
        
        return loss
    
    def on_validation_epoch_end(self):
        """Print detailed performance at end of validation epoch."""
        if self.hparams.training_stage == 1:
            # Print confusion matrix for categories
            confusion = self.val_cat_confusion.compute()
            self.val_cat_confusion.reset()
            
            print("\n" + "="*60)
            print(f"STAGE 1 - Epoch {self.current_epoch} Validation Results")
            print("="*60)
            print("\nCategory Confusion Matrix:")
            print("Pred→  vowel  stop  fric  nasal liquid")
            for i, true_cat in enumerate(['vowel', 'stop', 'fric', 'nasal', 'liquid']):
                row = confusion[i]
                print(f"{true_cat:6} ", end="")
                for j in range(5):
                    print(f"{int(row[j]):5} ", end="")
                print()
            
            # Per-category accuracy
            print("\nPer-Category Accuracy:")
            for i, cat in enumerate(IDX_TO_CATEGORY.values()):
                if confusion[i].sum() > 0:
                    acc = confusion[i, i] / confusion[i].sum()
                    print(f"  {cat}: {acc:.3f}")
            print("="*60 + "\n")
    
    def _verify_data(self, y, batch_idx):
        """Verify that phoneme labels are correct."""
        print(f"\n--- Batch {batch_idx} Data Verification ---")
        
        for idx in y[:5]:  # Check first 5 samples
            phoneme = self.idx_to_phoneme[idx.item()]
            if phoneme in PHONEME_TO_CATEGORY:
                category = PHONEME_TO_CATEGORY[phoneme]
            else:
                category = "UNKNOWN"
            
            self.phoneme_counter[phoneme] += 1
            self.category_counter[category] += 1
            
            print(f"  Label idx: {idx.item():2d} → Phoneme: '{phoneme:4s}' → Category: {category}")
        
        if batch_idx == 4:  # After 5 batches, print summary
            print(f"\n--- Data Distribution Summary (first 5 batches) ---")
            print("Category counts:")
            for cat, count in sorted(self.category_counter.items()):
                print(f"  {cat}: {count}")
            print(f"Unique phonemes seen: {len(self.phoneme_counter)}")
            print("Most common phonemes:", self.phoneme_counter.most_common(5))
            print("-" * 50 + "\n")
    
    def _print_category_performance(self, logits, labels, phoneme_labels, prefix):
        """Print detailed category classification performance."""
        preds = torch.argmax(logits, dim=1)
        correct = (preds == labels).float().mean()
        
        print(f"\n[{prefix}] Category Classification - Accuracy: {correct:.3f}")
        
        # Per-category accuracy
        category_correct = {}
        category_total = {}
        
        for pred, label in zip(preds, labels):
            cat_name = IDX_TO_CATEGORY[label.item()]
            if cat_name not in category_correct:
                category_correct[cat_name] = 0
                category_total[cat_name] = 0
            
            category_total[cat_name] += 1
            if pred == label:
                category_correct[cat_name] += 1
        
        # Print per-category accuracy
        print("Per-category accuracy:")
        for cat in CATEGORY_TO_IDX.keys():
            if cat in category_total and category_total[cat] > 0:
                acc = category_correct[cat] / category_total[cat]
                print(f"  {cat}: {acc:.3f} ({category_correct[cat]}/{category_total[cat]})")
        
        # Show some predictions
        for i in range(min(3, len(preds))):
            pred_cat = IDX_TO_CATEGORY[preds[i].item()]
            true_cat = IDX_TO_CATEGORY[labels[i].item()]
            phoneme = self.idx_to_phoneme[phoneme_labels[i].item()]
            
            symbol = "✓" if preds[i] == labels[i] else "✗"
            probs = F.softmax(logits[i], dim=0)
            top_prob = probs[preds[i]].item()
            
            print(f"  {symbol} Phoneme '{phoneme}' | True: {true_cat} | Pred: {pred_cat} (conf: {top_prob:.2f})")
    
    def _print_expert_performance(self, phoneme_logits, phoneme_labels, cat_logits, cat_labels, prefix):
        """Print detailed expert model performance."""
        phoneme_preds = torch.argmax(phoneme_logits, dim=1)
        cat_preds = torch.argmax(cat_logits, dim=1)
        
        phoneme_correct = (phoneme_preds == phoneme_labels).float().mean()
        cat_correct = (cat_preds == cat_labels).float().mean()
        
        print(f"\n[{prefix}] Stage 2 Performance")
        print(f"  Category Acc: {cat_correct:.3f} | Phoneme Acc: {phoneme_correct:.3f}")
        
        # Show some predictions
        for i in range(min(3, len(phoneme_preds))):
            true_phoneme = self.idx_to_phoneme[phoneme_labels[i].item()]
            pred_phoneme = self.idx_to_phoneme[phoneme_preds[i].item()]
            pred_cat = IDX_TO_CATEGORY[cat_preds[i].item()]
            
            symbol = "✓" if phoneme_preds[i] == phoneme_labels[i] else "✗"
            
            print(f"  {symbol} True: '{true_phoneme}' | Pred: '{pred_phoneme}' (via {pred_cat})")
    
    def _get_category_labels(self, phoneme_indices):
        """Convert phoneme indices to category indices."""
        category_labels = []
        
        for idx in phoneme_indices:
            phoneme = self.idx_to_phoneme[idx.item()]
            if phoneme in PHONEME_TO_CATEGORY:
                category = PHONEME_TO_CATEGORY[phoneme]
                category_idx = CATEGORY_TO_IDX[category]
            else:
                # Default to most common category (vowels)
                category_idx = CATEGORY_TO_IDX['vowels']
            category_labels.append(category_idx)
        
        return torch.tensor(category_labels, device=phoneme_indices.device)
    
    def configure_optimizers(self):
        # Different optimizer config based on stage
        if self.hparams.training_stage == 1:
            # Stage 1: Train everything with warm restart
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=self.hparams.learning_rate,
                weight_decay=0.01
            )
            
            print(f"Stage 1 Optimizer: Training all parameters")
            
            # Use cosine annealing with warm restarts for Stage 1
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10,  # Restart every 10 epochs
                T_mult=2,  # Double the restart period after each restart
                eta_min=1e-6
            )
            
        else:  # Stage 2
            # Stage 2: Optionally freeze backbone
            if self.hparams.freeze_backbone_stage2:
                # Freeze feature extractor and temporal encoder
                for param in self.feature_extractor.parameters():
                    param.requires_grad = False
                for param in self.temporal_encoder.parameters():
                    param.requires_grad = False
                for param in self.category_classifier.parameters():
                    param.requires_grad = False
                
                # Only train experts
                optimizer = torch.optim.AdamW(
                    self.experts.parameters(),
                    lr=self.hparams.learning_rate * 2,  # Higher LR for experts
                    weight_decay=0.01
                )
                
                print(f"Stage 2 Optimizer: Training only expert models (backbone frozen)")
                
            else:
                # Train everything but with different learning rates
                param_groups = [
                    {'params': self.feature_extractor.parameters(), 'lr': self.hparams.learning_rate * 0.1},
                    {'params': self.temporal_encoder.parameters(), 'lr': self.hparams.learning_rate * 0.1},
                    {'params': self.category_classifier.parameters(), 'lr': self.hparams.learning_rate * 0.5},
                    {'params': self.experts.parameters(), 'lr': self.hparams.learning_rate}
                ]
                
                optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
                
                print(f"Stage 2 Optimizer: Training all parameters with different LRs")
            
            # Standard cosine annealing for Stage 2
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=1e-6
            )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }

# ============================================
# Model Creation Function
# ============================================

def create_sequential_meg_model(dataset_info):
    """
    Create sequential two-step MEG phoneme classification model.
    
    Args:
        dataset_info: Dictionary with dataset information
            Should include training_stage and optionally stage1_checkpoint_path
    """
    model = MEGSequentialTwoStep(
        meg_channels=dataset_info.get('meg_channels', 306),
        time_points=dataset_info.get('time_points', 125),
        vocab_size=dataset_info.get('num_phonemes', 39),
        hidden_dim=dataset_info.get('hidden_dim', 256),
        num_temporal_layers=dataset_info.get('num_temporal_layers', 2),
        learning_rate=dataset_info.get('learning_rate', 1e-4),
        use_ctc=dataset_info.get('use_ctc', True),
        ctc_weight=dataset_info.get('ctc_weight', 0.3),
        training_stage=dataset_info.get('training_stage', 1),
        freeze_backbone_stage2=dataset_info.get('freeze_backbone_stage2', True),
        stage1_checkpoint_path=dataset_info.get('stage1_checkpoint_path', None)
    )
    
    return model

# ============================================
# Training Script Example
# ============================================

"""
Example usage with config files:

# Stage 1: Train category classifier
# config_stage1.yaml should have:
#   training_stage: 1
#   max_epochs: 20
#   save_dir: /path/to/stage1_model

python train.py --config config_stage1.yaml

# Stage 2: Train expert models  
# config_stage2.yaml should have:
#   training_stage: 2
#   stage1_checkpoint_path: /path/to/stage1_model/checkpoints/last.ckpt
#   freeze_backbone_stage2: true
#   max_epochs: 30
#   save_dir: /path/to/stage2_model

python train.py --config config_stage2.yaml

The model will automatically load Stage 1 weights when stage1_checkpoint_path is provided.
"""