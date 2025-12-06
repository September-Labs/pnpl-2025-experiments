"""
Multi-Scale MEG Phoneme Classification with MSDD-inspired architecture
Enhanced with masking strategies for struggling phonemes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import F1Score
from pathlib import Path
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import List, Tuple, Dict, Optional
from collections import Counter


# ============================================
# Focal Loss for Struggling Phonemes
# ============================================

class FocalLossWithMasking(nn.Module):
    """Focal loss that emphasizes hard-to-classify phonemes."""
    
    def __init__(self, 
                 num_classes: int = 39,
                 alpha: Optional[torch.Tensor] = None,
                 gamma: float = 2.0,
                 reduction: str = 'mean',
                 label_smoothing: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
        # Alpha is class weights (computed from training data)
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [B, num_classes] logits
            targets: [B] class indices
        """
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            ce_loss = F.cross_entropy(
                inputs, targets, 
                reduction='none',
                label_smoothing=self.label_smoothing
            )
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get probabilities
        p = F.softmax(inputs, dim=1)
        p_t = p.gather(1, targets.view(-1, 1)).squeeze(1)
        
        # Focal term: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ============================================
# Multi-Scale Dataset Loader
# ============================================

class MultiScaleGroupedDataset(Dataset):
    """Loads multiple pre-grouped datasets with different averaging levels."""
    
    def __init__(self, 
                 base_path: str,
                 scales: List[int] = [5, 10, 25, 50, 100],
                 partition: str = 'train',
                 load_to_memory: bool = True,
                 align_samples: bool = True):
        """
        Args:
            base_path: Directory containing preprocessed H5 files
            scales: List of grouping scales to load
            partition: 'train', 'validation', or 'test'
            load_to_memory: Whether to load all data into memory
            align_samples: Whether to ensure samples are aligned across scales
        """
        self.scales = sorted(scales)
        self.partition = partition
        self.datasets = {}
        self.lengths = {}
        self.align_samples = align_samples
        
        # Load each scale's dataset
        for scale in self.scales:
            scale_dir = Path(base_path).parent / f"preprocessed_data_tmax0_5_grouped{scale}"
            h5_path = scale_dir / f"{partition}_grouped.h5"
            
            if not h5_path.exists():
                raise FileNotFoundError(f"Dataset not found: {h5_path}")
            
            print(f"Loading {partition} dataset for scale {scale} from {h5_path}")
            
            if load_to_memory:
                with h5py.File(h5_path, 'r') as f:
                    self.datasets[scale] = {
                        'data': torch.tensor(f['data'][:], dtype=torch.float32),
                        'labels': torch.tensor(f['labels'][:], dtype=torch.long)
                    }
                    self.lengths[scale] = len(self.datasets[scale]['labels'])
            else:
                self.datasets[scale] = h5py.File(h5_path, 'r')
                self.lengths[scale] = len(self.datasets[scale]['labels'])
        
        # Use the scale with most samples as the base length
        # This ensures we can always find corresponding samples
        self.length = min(self.lengths.values())
        
        print(f"Multi-scale dataset initialized:")
        for scale, length in self.lengths.items():
            print(f"  Scale {scale}: {length} samples")
        print(f"  Using minimum length: {self.length}")
        
        # Store all labels for balanced sampling
        self.all_labels = []
        if load_to_memory:
            # Use labels from the first scale (they should be aligned)
            first_scale = self.scales[0]
            self.all_labels = self.datasets[first_scale]['labels'][:self.length].tolist()
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        """Returns data from all scales for the same index."""
        multi_scale_data = {}
        labels = None
        
        for scale in self.scales:
            if isinstance(self.datasets[scale], dict):
                # Data loaded in memory
                data = self.datasets[scale]['data'][idx]
                label = self.datasets[scale]['labels'][idx]
            else:
                # Data in H5 file
                data = torch.tensor(self.datasets[scale]['data'][idx], dtype=torch.float32)
                label = torch.tensor(self.datasets[scale]['labels'][idx], dtype=torch.long)
            
            multi_scale_data[scale] = data
            
            # Verify labels are consistent across scales (they should be for aligned data)
            if labels is None:
                labels = label
            elif self.align_samples and labels != label:
                print(f"Warning: Label mismatch at idx {idx}: {labels} vs {label}")
        
        return multi_scale_data, labels


# ============================================
# Scale Weighting Module (MSDD-inspired)
# ============================================

class ScaleWeightingModule(nn.Module):
    """Computes dynamic weights for each scale based on input features."""
    
    def __init__(self, hidden_dim: int, num_scales: int, weighting_type: str = 'cnn'):
        super().__init__()
        self.num_scales = num_scales
        self.weighting_type = weighting_type
        
        if weighting_type == 'cnn':
            # CNN-based weighting (like original MSDD)
            self.weight_conv = nn.Sequential(
                nn.Conv1d(hidden_dim * num_scales, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, num_scales, kernel_size=1),
                nn.Softmax(dim=1)
            )
        elif weighting_type == 'attention':
            # Attention-based weighting
            self.weight_attention = nn.MultiheadAttention(
                hidden_dim, num_heads=4, batch_first=True
            )
            self.weight_projection = nn.Linear(hidden_dim, num_scales)
        else:
            # Simple MLP weighting
            self.weight_mlp = nn.Sequential(
                nn.Linear(hidden_dim * num_scales, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_scales),
                nn.Softmax(dim=-1)
            )
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of tensors [B, hidden_dim, T] for each scale
        Returns:
            weights: Tensor [B, num_scales, T] with weights for each scale
        """
        B, H, T = features[0].shape
        
        if self.weighting_type == 'cnn':
            # Concatenate all scale features
            concat_features = torch.cat(features, dim=1)  # [B, H*num_scales, T]
            weights = self.weight_conv(concat_features)  # [B, num_scales, T]
            
        elif self.weighting_type == 'attention':
            # Stack features for attention
            stacked = torch.stack(features, dim=1).transpose(2, 3)  # [B, T, num_scales, H]
            stacked = stacked.reshape(B * T, self.num_scales, H)
            
            # Self-attention across scales
            attn_out, _ = self.weight_attention(stacked, stacked, stacked)
            attn_out = attn_out.mean(dim=-1)  # [B*T, num_scales]
            weights = F.softmax(self.weight_projection(attn_out), dim=-1)
            weights = weights.reshape(B, T, self.num_scales).transpose(1, 2)  # [B, num_scales, T]
            
        else:
            # MLP-based weighting
            concat_features = torch.cat(features, dim=1).transpose(1, 2)  # [B, T, H*num_scales]
            weights = self.weight_mlp(concat_features).transpose(1, 2)  # [B, num_scales, T]
        
        return weights


# ============================================
# Multi-Scale MEG Conformer Layer
# ============================================

class MultiScaleMEGConformer(nn.Module):
    """Conformer layer that processes multiple scales."""
    
    def __init__(self, dim: int, num_heads: int = 4, ff_dim: int = None,
                 kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        ff_dim = ff_dim or 2 * dim
        
        # Shared conformer components for all scales
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
        """Process single scale through conformer."""
        # x: (B, T, D)
        res = x
        x_conv = x.transpose(1, 2)  # (B, D, T)
        x_conv = self.conv(x_conv).transpose(1, 2)  # (B, T, D)
        x = self.ln1(x_conv + res)
        
        res = x
        attn_out, _ = self.attention(x, x, x)
        x = self.ln2(self.dropout(attn_out) + res)
        
        res = x
        x = self.ffn(x)
        x = self.ln3(x + res)
        
        return x


# ============================================
# Main Multi-Scale MSDD Model with Enhanced Masking
# ============================================

class MultiScaleMEGPhonemeClassifier(L.LightningModule):
    """Multi-scale MEG phoneme classifier with MSDD-inspired weighting and masking strategies."""
    
    # Define struggling phonemes based on evaluation results
    STRUGGLING_PHONEMES = {
        20: 'aa', 21: 'ae', 22: 'ao', 23: 'aw', 24: 'b',
        25: 'ch', 26: 'dh', 27: 'eh', 28: 'er', 29: 'g',
        30: 'jh', 31: 'm', 32: 'oy', 33: 'p', 34: 'sh',
        35: 'th', 36: 'uh', 37: 'uw', 38: 'y'
    }
    
    def __init__(self,
                 meg_channels: int = 306,
                 time_points: int = 125,
                 vocab_size: int = 39,
                 hidden_dim: int = 256,
                 num_conformers: int = 4,
                 scales: List[int] = [5, 10, 25, 50, 100],
                 weighting_type: str = 'cnn',
                 learning_rate: float = 1e-4,
                 share_encoders: bool = True,
                 label_smoothing: float = 0.0,
                 use_mixup: bool = True,
                 mixup_alpha: float = 0.2,
                 use_focal_loss: bool = True,
                 focal_gamma: float = 2.0,
                 struggling_weight: float = 5.0):
        super().__init__()
        self.save_hyperparameters()
        
        self.scales = sorted(scales)
        self.num_scales = len(scales)
        self.struggling_indices = list(self.STRUGGLING_PHONEMES.keys())
        
        # Scale-specific or shared MEG encoders
        if share_encoders:
            # Single shared encoder for all scales
            self.meg_encoder = nn.Sequential(
                nn.Conv1d(meg_channels, hidden_dim, kernel_size=7, padding=3),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )
            self.scale_encoders = {scale: self.meg_encoder for scale in scales}
        else:
            # Separate encoder for each scale (can learn scale-specific features)
            self.scale_encoders = nn.ModuleDict({
                str(scale): nn.Sequential(
                    nn.Conv1d(meg_channels, hidden_dim, kernel_size=7, padding=3),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU()
                ) for scale in scales
            })
        
        # Conformer layers (shared across scales)
        self.conformers = nn.ModuleList([
            MultiScaleMEGConformer(hidden_dim, 4, hidden_dim*2, dropout=0.2)
            for _ in range(num_conformers)
        ])
        
        # Scale weighting module (MSDD-inspired)
        self.scale_weighter = ScaleWeightingModule(
            hidden_dim, self.num_scales, weighting_type
        )
        
        # Scale embedding (optional - helps model distinguish scales)
        self.scale_embeddings = nn.Embedding(max(scales) + 1, hidden_dim)
        
        # Classification head with increased dropout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * time_points, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, vocab_size)
        )
        
        # Loss function selection
        if use_focal_loss:
            # Create class weights for focal loss
            class_weights = torch.ones(vocab_size)
            for idx in self.struggling_indices:
                class_weights[idx] = struggling_weight
            
            self.criterion = FocalLossWithMasking(
                num_classes=vocab_size,
                alpha=class_weights,
                gamma=focal_gamma,
                label_smoothing=label_smoothing
            )
        else:
            # Standard cross-entropy
            if label_smoothing > 0:
                self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            else:
                self.criterion = nn.CrossEntropyLoss()
        
        self.f1_macro = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
        
        # For logging scale weights
        self.last_scale_weights = None
        
        # Mixup parameters
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
    
    def forward(self, x):
        """
        Args:
            x: Dict of tensors {scale: [B, channels, time_points]} or single tensor
        Returns:
            logits: [B, vocab_size]
        """
        # Handle both multi-scale and single-scale inputs
        if isinstance(x, dict):
            # Multi-scale input (training)
            return self.forward_multiscale(x)
        else:
            # Single-scale input (inference on 100-averaged)
            return self.forward_single_scale(x)
    
    def forward_multiscale(self, x_dict):
        """Forward pass with multiple scales."""
        B = next(iter(x_dict.values())).shape[0]
        scale_features = []
        
        # Process each scale
        for scale in self.scales:
            if scale not in x_dict:
                continue
                
            x = x_dict[scale]  # [B, C, T]
            
            # Encode MEG features
            if isinstance(self.scale_encoders, nn.ModuleDict):
                features = self.scale_encoders[str(scale)](x)  # [B, hidden_dim, T]
            else:
                features = self.meg_encoder(x)
            
            # Add scale embedding
            scale_emb = self.scale_embeddings(torch.tensor(scale, device=x.device))
            scale_emb = scale_emb.unsqueeze(0).unsqueeze(-1).expand(B, -1, features.shape[-1])
            features = features + scale_emb
            
            scale_features.append(features)
        
        # Compute scale weights
        scale_weights = self.scale_weighter(scale_features)  # [B, num_scales, T]
        self.last_scale_weights = scale_weights.mean(dim=(0, 2)).detach()  # Average weights for logging
        
        # Apply conformers to each scale and combine
        processed_features = []
        for features in scale_features:
            features = features.transpose(1, 2)  # [B, T, hidden_dim]
            for conformer in self.conformers:
                features = conformer(features)
            processed_features.append(features)
        
        # Weighted combination of scales
        combined = torch.zeros_like(processed_features[0])
        for i, features in enumerate(processed_features):
            weight = scale_weights[:, i:i+1, :].transpose(1, 2)  # [B, T, 1]
            combined = combined + features * weight
        
        # Classification
        features_flat = combined.reshape(B, -1)
        logits = self.classifier(features_flat)
        
        return logits
    
    def forward_single_scale(self, x):
        """Forward pass with single scale (for inference)."""
        B, C, T = x.shape
        
        # Process as scale 100 (highest quality)
        features = self.meg_encoder(x) if hasattr(self, 'meg_encoder') else self.scale_encoders['100'](x)
        
        # Add scale embedding for 100
        scale_emb = self.scale_embeddings(torch.tensor(100, device=x.device))
        scale_emb = scale_emb.unsqueeze(0).unsqueeze(-1).expand(B, -1, T)
        features = features + scale_emb
        
        # Apply conformers
        features = features.transpose(1, 2)  # [B, T, hidden_dim]
        for conformer in self.conformers:
            features = conformer(features)
        
        # Classification
        features_flat = features.reshape(B, -1)
        logits = self.classifier(features_flat)
        
        return logits
    
    def mixup_data(self, x, y, alpha=1.0):
        """Mixup augmentation for multi-scale data."""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = y.size(0)
        index = torch.randperm(batch_size).to(y.device)
        
        if isinstance(x, dict):
            mixed_x = {}
            for scale in x.keys():
                mixed_x[scale] = lam * x[scale] + (1 - lam) * x[scale][index]
        else:
            mixed_x = lam * x + (1 - lam) * x[index]
        
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    
    def apply_phoneme_augmentation(self, x_dict, y):
        """Apply augmentation specifically to struggling phonemes."""
        device = y.device
        
        for i in range(len(y)):
            if y[i].item() in self.struggling_indices:
                # 50% chance to augment struggling phonemes
                if torch.rand(1).item() < 0.5:
                    aug_type = torch.randint(0, 4, (1,)).item()
                    
                    for scale in x_dict:
                        if aug_type == 0:
                            # Add noise
                            noise = torch.randn_like(x_dict[scale][i]) * 0.15
                            x_dict[scale][i] += noise
                        
                        elif aug_type == 1:
                            # Channel dropout - drop 10% of channels
                            channels_to_drop = torch.randperm(306, device=device)[:31]
                            x_dict[scale][i][channels_to_drop] = 0
                        
                        elif aug_type == 2:
                            # Temporal shift
                            shift = torch.randint(-5, 6, (1,)).item()
                            x_dict[scale][i] = torch.roll(x_dict[scale][i], shifts=shift, dims=-1)
                        
                        elif aug_type == 3:
                            # Temporal masking
                            T = x_dict[scale][i].shape[-1]
                            mask_len = T // 10
                            mask_start = torch.randint(0, T - mask_len, (1,)).item()
                            x_dict[scale][i][:, mask_start:mask_start+mask_len] = 0
        
        return x_dict
    
    def training_step(self, batch, batch_idx):
        x_dict, y = batch
        
        # Apply phoneme-specific augmentation for struggling phonemes
        if self.training:
            x_dict = self.apply_phoneme_augmentation(x_dict, y)
        
        # Apply mixup if enabled
        if self.use_mixup and self.training:
            x_dict, y_a, y_b, lam = self.mixup_data(x_dict, y, self.mixup_alpha)
            y_hat = self(x_dict)
            loss = lam * self.criterion(y_hat, y_a) + (1 - lam) * self.criterion(y_hat, y_b)
            
            # Log F1 for original labels only
            f1_macro = self.f1_macro(y_hat, y_a)
        else:
            y_hat = self(x_dict)
            loss = self.criterion(y_hat, y)
            f1_macro = self.f1_macro(y_hat, y)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_f1_macro', f1_macro)
        
        # Log scale weights
        if self.last_scale_weights is not None:
            for i, scale in enumerate(self.scales):
                self.log(f'scale_weight_{scale}', self.last_scale_weights[i])
        
        # Monitor struggling phoneme performance periodically
        if batch_idx % 50 == 0:
            with torch.no_grad():
                preds = torch.argmax(y_hat, dim=1)
                
                # Log accuracy for top 5 struggling phonemes
                for idx in self.struggling_indices[:5]:
                    mask = y == idx
                    if mask.any():
                        acc = (preds[mask] == idx).float().mean()
                        phoneme_name = self.STRUGGLING_PHONEMES.get(idx, f'phoneme_{idx}')
                        self.log(f'train_acc_{phoneme_name}', acc)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        # Validation uses single scale (100-averaged)
        x_dict, y = batch
        
        # Use only the highest scale for validation
        if isinstance(x_dict, dict) and 100 in x_dict:
            x = x_dict[100]
        else:
            x = x_dict
            
        y_hat = self.forward_single_scale(x)
        loss = self.criterion(y_hat, y)
        f1_macro = self.f1_macro(y_hat, y)
        
        self.log('val_loss', loss)
        self.log('val_f1_macro', f1_macro, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=0.01
        )
        
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
# Helper function to create balanced dataloader
# ============================================

def create_balanced_dataloader(dataset, config, batch_size=32, num_workers=0):
    """Create a balanced dataloader with weighted sampling for struggling phonemes."""
    
    # Get all labels
    all_labels = []
    if hasattr(dataset, 'all_labels') and dataset.all_labels:
        all_labels = dataset.all_labels
    else:
        # Extract labels from dataset
        print("Extracting labels for balanced sampling...")
        for i in range(len(dataset)):
            _, label = dataset[i]
            if torch.is_tensor(label):
                all_labels.append(label.item())
            else:
                all_labels.append(label)
    
    # Count phoneme frequencies
    phoneme_counts = Counter(all_labels)
    num_classes = len(phoneme_counts)
    
    # Create weights (inverse frequency)
    class_weights = torch.zeros(num_classes)
    for phoneme_id, count in phoneme_counts.items():
        class_weights[phoneme_id] = 1.0 / (count + 1)
    
    # Boost struggling phonemes
    struggling_indices = list(MultiScaleMEGPhonemeClassifier.STRUGGLING_PHONEMES.keys())
    for idx in struggling_indices:
        if idx < num_classes:
            class_weights[idx] *= 5.0  # 5x weight for struggling phonemes
    
    # Create sample weights
    sample_weights = []
    for label in all_labels:
        sample_weights.append(class_weights[label].item())
    
    # Create weighted sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )
    
    # Create dataloader with sampler
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,  # Replaces shuffle=True
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False
    )
    
    print(f"Created balanced dataloader with {len(dataset)} samples")
    print(f"Struggling phonemes boosted by 5x")
    
    return dataloader


# ============================================
# Custom Data Module for Multi-Scale Training with Balanced Sampling
# ============================================

class MultiScaleDataModule(L.LightningDataModule):
    """Lightning DataModule for multi-scale training with balanced sampling."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_config = config['data']
        self.training_config = config['training']
        self.use_balanced_sampling = config.get('use_balanced_sampling', True)
        
    def setup(self, stage=None):
        base_path = self.data_config['preprocessed_dir']
        scales = self.data_config.get('multiscale_groups', [5, 10, 25, 50, 100])
        
        if stage == 'fit' or stage is None:
            self.train_dataset = MultiScaleGroupedDataset(
                base_path=base_path,
                scales=scales,
                partition='train',
                load_to_memory=self.data_config.get('load_to_memory', True)
            )
            
            # For validation, we can use single scale (100) or multiscale
            self.val_dataset = MultiScaleGroupedDataset(
                base_path=base_path,
                scales=[100],  # Validation on highest quality only
                partition='validation',
                load_to_memory=self.data_config.get('load_to_memory', True)
            )
            
        if stage == 'test':
            self.test_dataset = MultiScaleGroupedDataset(
                base_path=base_path,
                scales=[100],  # Test on highest quality only
                partition='test',
                load_to_memory=self.data_config.get('load_to_memory', True)
            )
    
    def train_dataloader(self):
        if self.use_balanced_sampling:
            # Use balanced sampling for struggling phonemes
            return create_balanced_dataloader(
                self.train_dataset,
                self.config,
                batch_size=self.training_config['batch_size'],
                num_workers=self.training_config['num_workers']
            )
        else:
            # Standard dataloader
            return DataLoader(
                self.train_dataset,
                batch_size=self.training_config['batch_size'],
                shuffle=True,
                num_workers=self.training_config['num_workers'],
                pin_memory=True,
                persistent_workers=True if self.training_config['num_workers'] > 0 else False
            )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=False,
            num_workers=self.training_config['num_workers'],
            pin_memory=True,
            persistent_workers=True if self.training_config['num_workers'] > 0 else False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=False,
            num_workers=self.training_config['num_workers'],
            pin_memory=True
        )