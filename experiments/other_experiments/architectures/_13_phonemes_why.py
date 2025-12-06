"""
Struggling Phoneme Specialist Model with Multiple Parallel Strategies
Tests various approaches simultaneously for the 13 underperforming phonemes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from torchmetrics import F1Score, Accuracy
from collections import defaultdict
import torchaudio.transforms as T

# ============================================
# Phonetic Features Definition
# ============================================

PHONETIC_FEATURES = {
    # Manner of articulation (7 categories)
    'manner': {
        'stop': ['b', 'd', 'g', 'k', 'p', 't'],
        'fricative': ['dh', 'f', 's', 'sh', 'th', 'v', 'z', 'zh'],
        'affricate': ['ch', 'jh'],
        'nasal': ['m', 'n', 'ng'],
        'liquid': ['l', 'r'],
        'glide': ['w', 'y'],
        'vowel': ['aa', 'ae', 'ah', 'ao', 'aw', 'ay', 'eh', 'er', 'ey', 
                  'ih', 'iy', 'ow', 'oy', 'uh', 'uw']
    },
    # Place of articulation (8 categories)
    'place': {
        'bilabial': ['b', 'm', 'p', 'w'],
        'labiodental': ['f', 'v'],
        'dental': ['dh', 'th'],
        'alveolar': ['d', 'l', 'n', 's', 't', 'z'],
        'palatal': ['ch', 'jh', 'sh', 'y', 'zh'],
        'velar': ['g', 'k', 'ng'],
        'glottal': ['hh'],
        'vowel': ['aa', 'ae', 'ah', 'ao', 'aw', 'ay', 'eh', 'er', 'ey',
                  'ih', 'iy', 'ow', 'oy', 'uh', 'uw']
    },
    # Voicing (2 categories)
    'voicing': {
        'voiced': ['b', 'd', 'dh', 'g', 'jh', 'l', 'm', 'n', 'ng', 'r', 
                   'v', 'w', 'y', 'z', 'zh'] + 
                  ['aa', 'ae', 'ah', 'ao', 'aw', 'ay', 'eh', 'er', 'ey',
                   'ih', 'iy', 'ow', 'oy', 'uh', 'uw'],
        'unvoiced': ['ch', 'f', 'hh', 'k', 'p', 's', 'sh', 't', 'th']
    },
    # Vowel height (for vowels only)
    'height': {
        'high': ['iy', 'ih', 'uw', 'uh'],
        'mid': ['ey', 'eh', 'er', 'ow', 'ao'],
        'low': ['ae', 'ah', 'aa', 'aw', 'ay', 'oy'],
        'na': []  # Non-vowels
    },
    # Vowel backness (for vowels only)
    'backness': {
        'front': ['iy', 'ih', 'ey', 'eh', 'ae'],
        'central': ['ah', 'er'],
        'back': ['uw', 'uh', 'ow', 'ao', 'aa'],
        'diphthong': ['aw', 'ay', 'oy'],
        'na': []  # Non-vowels
    }
}

# ============================================
# Heavy Augmentation Module
# ============================================

class HeavyAugmentation(nn.Module):
    """Advanced augmentation techniques for rare phonemes"""
    
    def __init__(self, phoneme_counts, rare_threshold=50):
        super().__init__()
        self.phoneme_counts = phoneme_counts
        self.rare_threshold = rare_threshold
        
        # SpecAugment-style augmentation
        self.time_masking = T.TimeMasking(time_mask_param=10)
        self.freq_masking = T.FrequencyMasking(freq_mask_param=20)
        
    def forward(self, x, labels):
        """Apply augmentation based on phoneme rarity
        x: (B, channels, time)
        labels: (B,)
        """
        if not self.training:
            return x
            
        batch_size = x.size(0)
        augmented = x.clone()
        
        for i in range(batch_size):
            phoneme_idx = labels[i].item()
            count = self.phoneme_counts.get(str(phoneme_idx), 1000)
            
            # More aggressive augmentation for rarer phonemes
            if count < self.rare_threshold:
                augmentation_strength = 1.0 - (count / self.rare_threshold)
                
                # 1. Gaussian noise
                noise_std = 0.05 * augmentation_strength
                noise = torch.randn_like(augmented[i]) * noise_std
                augmented[i] += noise
                
                # 2. Channel dropout (simulate sensor failure)
                if torch.rand(1) < 0.3 * augmentation_strength:
                    n_dropout = int(30 * augmentation_strength)
                    dropout_channels = torch.randperm(x.size(1))[:n_dropout]
                    augmented[i, dropout_channels, :] = 0
                
                # 3. Time warping
                if torch.rand(1) < 0.5:
                    # Simple time shift
                    max_shift = int(10 * augmentation_strength)
                    shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
                    if shift > 0:
                        augmented[i, :, shift:] = augmented[i, :, :-shift]
                        augmented[i, :, :shift] = 0
                    elif shift < 0:
                        augmented[i, :, :shift] = augmented[i, :, -shift:]
                        augmented[i, :, shift:] = 0
                
                # 4. Mixup with same class samples (if available)
                if torch.rand(1) < 0.3 and batch_size > 1:
                    # Find another sample with same label
                    same_label_mask = labels == phoneme_idx
                    if same_label_mask.sum() > 1:
                        other_indices = torch.where(same_label_mask & (torch.arange(batch_size, device=x.device) != i))[0]
                        if len(other_indices) > 0:
                            mix_idx = other_indices[torch.randint(0, len(other_indices), (1,))].item()
                            alpha = 0.2 + 0.3 * augmentation_strength
                            augmented[i] = (1 - alpha) * augmented[i] + alpha * x[mix_idx]
                
                # 5. Amplitude scaling
                scale = 1.0 + (torch.rand(1).item() - 0.5) * 0.4 * augmentation_strength
                augmented[i] *= scale
        
        return augmented

# ============================================
# Sensor Selection Strategies
# ============================================

class MultiStrategySensorSelector(nn.Module):
    """Test multiple sensor selection strategies in parallel - FIXED VERSION"""
    
    def __init__(self, sensor_importance_path, region_importance_path, 
                 phoneme_summary_path, struggling_phonemes, phoneme_info_path=None):
        super().__init__()
        
        # Load all analysis data
        self.sensor_importance = pd.read_csv(sensor_importance_path, index_col=0)
        self.region_importance = pd.read_csv(region_importance_path, index_col=0)
        self.phoneme_summary = pd.read_csv(phoneme_summary_path)
        self.struggling_phonemes = struggling_phonemes
        
        # Load phoneme mapping
        if phoneme_info_path and Path(phoneme_info_path).exists():
            with open(phoneme_info_path, 'r') as f:
                phoneme_info = json.load(f)
                # Create name to index mapping
                self.phoneme_to_idx = {name: int(idx) for idx, name in phoneme_info['phoneme_names'].items()}
        else:
            # Default mapping based on alphabetical order
            phoneme_labels = ['aa', 'ae', 'ah', 'ao', 'aw', 'ay', 'b', 'ch', 'd', 'dh',
                            'eh', 'er', 'ey', 'f', 'g', 'hh', 'ih', 'iy', 'jh', 'k',
                            'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 'sh',
                            't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z', 'zh']
            self.phoneme_to_idx = {name: idx for idx, name in enumerate(phoneme_labels)}
        
        # Convert struggling phoneme names to indices
        self.struggling_indices = [self.phoneme_to_idx[name] for name in struggling_phonemes]
        
        # Precompute different sensor selection strategies
        self.strategies = {}
        self._compute_all_strategies()
        
    def _compute_all_strategies(self):
        """Compute all sensor selection strategies - FIXED"""
        # Get importance for struggling phonemes using integer indices
        struggling_importance = self.sensor_importance.iloc[self.struggling_indices]
        
        # Check if we have sensor columns (should be Sensor_0, Sensor_1, etc.)
        sensor_columns = [col for col in self.sensor_importance.columns if col.startswith('Sensor_')]
        
        # If we only have 102 sensors in the CSV but need 306, we'll handle that
        n_sensors_in_csv = len(sensor_columns)
        print(f"Found {n_sensors_in_csv} sensors in importance matrix")
        
        # Strategy 1: Average importance
        for k in [50, 100, 150, 200]:
            if k > n_sensors_in_csv:
                k = n_sensors_in_csv  # Adjust k if needed
                
            mean_importance = struggling_importance.mean(axis=0)
            top_sensors = mean_importance.nlargest(min(k, n_sensors_in_csv)).index
            sensor_indices = []
            for s in top_sensors:
                if isinstance(s, str) and 'Sensor_' in s:
                    sensor_indices.append(int(s.split('_')[1]))
                else:
                    sensor_indices.append(int(s))
            self.strategies[f'avg_{k}'] = torch.tensor(sensor_indices)
        
        # Strategy 2: Union of top sensors
        for k in [50, 100, 150]:
            if k > n_sensors_in_csv:
                k = n_sensors_in_csv
                
            selected = set()
            k_per_phoneme = min(k // len(self.struggling_indices) + 2, n_sensors_in_csv // len(self.struggling_indices))
            
            for idx in self.struggling_indices:
                phoneme_importance = self.sensor_importance.iloc[idx]
                top_sensors = phoneme_importance.nlargest(k_per_phoneme).index
                for s in top_sensors:
                    if isinstance(s, str) and 'Sensor_' in s:
                        selected.add(int(s.split('_')[1]))
                    else:
                        selected.add(int(s))
            
            self.strategies[f'union_{k}'] = torch.tensor(list(selected)[:min(k, len(selected))])
        
        # Strategy 3: Weighted by inverse F1
        f1_weights = {
            'ih': 2.5, 'v': 2.5, 'ey': 3.0, 'ah': 7.0,
            'aa': 10.0, 'ao': 10.0, 'er': 10.0, 'g': 10.0,
            'jh': 10.0, 'oy': 10.0, 'th': 10.0, 'uh': 10.0,
            'y': 10.0, 'zh': 10.0
        }
        
        weighted_importance = None
        for phoneme_name, idx in zip(self.struggling_phonemes, self.struggling_indices):
            weight = f1_weights.get(phoneme_name, 10.0)
            if weighted_importance is None:
                weighted_importance = self.sensor_importance.iloc[idx] * weight
            else:
                weighted_importance += self.sensor_importance.iloc[idx] * weight
        
        for k in [50, 100, 150]:
            if k > n_sensors_in_csv:
                k = n_sensors_in_csv
                
            top_sensors = weighted_importance.nlargest(min(k, n_sensors_in_csv)).index
            sensor_indices = []
            for s in top_sensors:
                if isinstance(s, str) and 'Sensor_' in s:
                    sensor_indices.append(int(s.split('_')[1]))
                else:
                    sensor_indices.append(int(s))
            self.strategies[f'weighted_{k}'] = torch.tensor(sensor_indices)
        
        # Strategy 4: Max importance (any phoneme with high importance)
        max_importance = struggling_importance.max(axis=0)
        for k in [75, 125]:
            if k > n_sensors_in_csv:
                k = n_sensors_in_csv
                
            top_sensors = max_importance.nlargest(min(k, n_sensors_in_csv)).index
            sensor_indices = []
            for s in top_sensors:
                if isinstance(s, str) and 'Sensor_' in s:
                    sensor_indices.append(int(s.split('_')[1]))
                else:
                    sensor_indices.append(int(s))
            self.strategies[f'max_{k}'] = torch.tensor(sensor_indices)
        
        # Print strategy summary
        print("\nSensor Selection Strategies Created:")
        for name, indices in self.strategies.items():
            print(f"  {name}: {len(indices)} sensors selected")
    
    def get_strategy_mask(self, strategy_name, device):
        """Get sensor mask for a specific strategy"""
        if strategy_name not in self.strategies:
            return None
        
        # Handle case where we have fewer sensors in analysis than in actual data
        # Assume actual MEG has 306 channels
        mask = torch.zeros(306, device=device)
        sensor_indices = self.strategies[strategy_name].to(device)
        
        # Only set mask for valid sensor indices
        valid_indices = sensor_indices[sensor_indices < 306]
        mask[valid_indices] = 1.0
        
        return mask

# ============================================
# Parallel Expert Models
# ============================================

class PhonemeExpert(nn.Module):
    """Single expert model for phoneme classification"""
    
    def __init__(self, input_channels, hidden_dim=128, n_classes=14):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim*2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_classes)
        )
    
    def forward(self, x):
        features = self.encoder(x).squeeze(-1)
        return self.classifier(features)

class MultiTaskExpert(nn.Module):
    """Expert with multi-task learning"""
    
    def __init__(self, input_channels, hidden_dim=128, n_classes=14):
        super().__init__()
        
        # Shared encoder
        self.shared_encoder = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim*2),
            nn.ReLU()
        )
        
        # Task-specific heads
        self.phoneme_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim*2, n_classes)
        )
        
        # Phonetic features head (manner, place, voicing)
        self.manner_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim*2, 7)  # 7 manner categories
        )
        
        self.place_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim*2, 8)  # 8 place categories
        )
        
        self.voicing_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim*2, 2)  # voiced/unvoiced
        )
    
    def forward(self, x):
        shared_features = self.shared_encoder(x)
        
        return {
            'phoneme': self.phoneme_head(shared_features),
            'manner': self.manner_head(shared_features),
            'place': self.place_head(shared_features),
            'voicing': self.voicing_head(shared_features)
        }

# ============================================
# Main Struggling Phoneme Specialist Model
# ============================================

class StrugglingPhonemeSpecialist(L.LightningModule):
    """
    Comprehensive model testing multiple strategies for struggling phonemes
    FIXED VERSION with proper device handling for metrics
    """
    
    def __init__(self,
                 meg_channels=306,
                 time_points=125,
                 vocab_size=39,
                 hidden_dim=256,
                 learning_rate=1e-4,
                 sensor_importance_path=None,
                 region_importance_path=None,
                 phoneme_summary_path=None,
                 phoneme_counts_path=None,
                 test_all_strategies=True,
                 use_multitask=True,
                 use_heavy_augmentation=True):
        super().__init__()
        self.save_hyperparameters()
        
        # Define struggling phonemes
        self.struggling_phonemes = ['aa', 'ao', 'er', 'g', 'jh', 'oy', 'th', 'uh', 'y', 'zh', 'ah', 'ey', 'v', 'ih']
        self.struggling_indices = [0, 3, 11, 14, 18, 25, 31, 32, 36, 38, 2, 12, 34, 16]
        
        # Create mapping for struggling phonemes (0-13)
        self.struggling_to_reduced = {idx: i for i, idx in enumerate(self.struggling_indices)}
        
        # Load phoneme counts for augmentation
        if phoneme_counts_path and Path(phoneme_counts_path).exists():
            with open(phoneme_counts_path, 'r') as f:
                data = json.load(f)
                self.phoneme_counts = data.get('phoneme_counts', {})
        else:
            self.phoneme_counts = {}
        
        # Heavy augmentation
        if use_heavy_augmentation:
            self.augmentation = HeavyAugmentation(self.phoneme_counts)
        else:
            self.augmentation = None
        
        # Sensor selection
        if sensor_importance_path and Path(sensor_importance_path).exists():
            try:
                self.sensor_selector = MultiStrategySensorSelector(
                    sensor_importance_path,
                    region_importance_path,
                    phoneme_summary_path,
                    self.struggling_phonemes,
                    phoneme_counts_path
                )
                print(f"Sensor selector initialized with {len(self.sensor_selector.strategies)} strategies")
            except Exception as e:
                print(f"Warning: Could not initialize sensor selector: {e}")
                print("Continuing without sensor selection strategies")
                self.sensor_selector = None
                test_all_strategies = False
        else:
            print("No sensor importance path provided, skipping sensor selection")
            self.sensor_selector = None
            test_all_strategies = False
        
        # Create multiple expert models for different strategies
        self.experts = nn.ModuleDict()
        
        if test_all_strategies and self.sensor_selector:
            # Test different sensor selection strategies
            for strategy_name in self.sensor_selector.strategies.keys():
                n_sensors = len(self.sensor_selector.strategies[strategy_name])
                if n_sensors > 0:
                    self.experts[f'expert_{strategy_name}'] = PhonemeExpert(
                        n_sensors, hidden_dim // 2, n_classes=14
                    )
                    print(f"Created expert_{strategy_name} with {n_sensors} sensors")
        
        # Full sensor models
        self.experts['expert_full'] = PhonemeExpert(meg_channels, hidden_dim, n_classes=14)
        print(f"Created expert_full with {meg_channels} sensors")
        
        if use_multitask:
            self.experts['expert_multitask'] = MultiTaskExpert(
                meg_channels, hidden_dim, n_classes=14
            )
            print(f"Created expert_multitask with multi-task learning")
        
        # Binary gate
        self.binary_gate = nn.Sequential(
            nn.Conv1d(meg_channels, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
        
        # Ensemble weights
        n_experts = len(self.experts)
        self.ensemble_weights = nn.Parameter(torch.ones(n_experts) / n_experts)
        print(f"Total experts created: {n_experts}")
        
        # Initialize metrics as None - will be created in setup()
        self.train_f1 = None
        self.val_f1 = None
        self.train_acc = None
        self.val_acc = None
        self.expert_metrics = None
        
        # Phonetic feature encoding
        self._create_phonetic_feature_labels()
    
    def setup(self, stage=None):
        """Setup metrics on the correct device"""
        # This is called after the model is moved to the correct device
        device = self.device
        
        # Initialize metrics on the correct device
        self.train_f1 = F1Score(num_classes=14, average='macro', task='multiclass').to(device)
        self.val_f1 = F1Score(num_classes=14, average='macro', task='multiclass').to(device)
        self.train_acc = Accuracy(num_classes=14, task='multiclass').to(device)
        self.val_acc = Accuracy(num_classes=14, task='multiclass').to(device)
        
        # Initialize expert metrics
        self.expert_metrics = {}
        for name in self.experts.keys():
            self.expert_metrics[name] = F1Score(num_classes=14, average='macro', task='multiclass').to(device)
        
        print(f"Metrics initialized on device: {device}")
    
    def _create_phonetic_feature_labels(self):
        """Create ground truth labels for phonetic features"""
        phoneme_labels = ['aa', 'ae', 'ah', 'ao', 'aw', 'ay', 'b', 'ch', 'd', 'dh', 
                         'eh', 'er', 'ey', 'f', 'g', 'hh', 'ih', 'iy', 'jh', 'k',
                         'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 'sh',
                         't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z', 'zh']
        
        # Create feature mappings
        self.phoneme_to_features = {}
        
        for idx, phoneme in enumerate(phoneme_labels):
            features = {}
            
            # Manner
            for manner_idx, (manner_name, phonemes) in enumerate(PHONETIC_FEATURES['manner'].items()):
                if phoneme in phonemes:
                    features['manner'] = manner_idx
                    break
            
            # Place
            for place_idx, (place_name, phonemes) in enumerate(PHONETIC_FEATURES['place'].items()):
                if phoneme in phonemes:
                    features['place'] = place_idx
                    break
            
            # Voicing
            features['voicing'] = 0 if phoneme in PHONETIC_FEATURES['voicing']['voiced'] else 1
            
            self.phoneme_to_features[idx] = features
    
    def forward(self, x, return_all_experts=False):
        """Forward pass through all experts"""
        B, C, T = x.shape
        
        # Binary gate prediction
        gate_logits = self.binary_gate(x)
        
        # Collect predictions from all experts
        expert_outputs = {}
        
        for expert_name, expert_model in self.experts.items():
            if 'multitask' in expert_name:
                outputs = expert_model(x)
                expert_outputs[expert_name] = outputs['phoneme']
            elif self.sensor_selector and expert_name.startswith('expert_') and expert_name != 'expert_full':
                # Apply sensor selection
                strategy = expert_name.replace('expert_', '')
                mask = self.sensor_selector.get_strategy_mask(strategy, x.device)
                if mask is not None:
                    x_masked = x * mask.unsqueeze(0).unsqueeze(-1)
                    selected_indices = torch.where(mask)[0]
                    x_selected = x_masked[:, selected_indices, :]
                    expert_outputs[expert_name] = expert_model(x_selected)
            else:
                expert_outputs[expert_name] = expert_model(x)
        
        if return_all_experts:
            return expert_outputs, gate_logits
        
        # Ensemble predictions
        ensemble_logits = torch.zeros(B, 14, device=x.device)
        weights = F.softmax(self.ensemble_weights, dim=0)
        
        for i, (expert_name, logits) in enumerate(expert_outputs.items()):
            ensemble_logits += weights[i] * logits
        
        return ensemble_logits, gate_logits
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # Ensure metrics are initialized
        if self.train_f1 is None:
            self.setup()
        
        # Filter to only struggling phonemes
        struggling_mask = torch.tensor([y_i.item() in self.struggling_indices for y_i in y], 
                                      device=self.device, dtype=torch.bool)
        
        if struggling_mask.sum() == 0:
            return None
        
        x_struggling = x[struggling_mask]
        y_struggling = y[struggling_mask]
        
        # Apply augmentation
        if self.augmentation:
            x_struggling = self.augmentation(x_struggling, y_struggling)
        
        # Convert to reduced labels (0-13)
        y_reduced = torch.tensor([self.struggling_to_reduced[y_i.item()] for y_i in y_struggling], 
                                 device=self.device, dtype=torch.long)
        
        # Get predictions from all experts
        expert_outputs, gate_logits_struggling = self.forward(x_struggling, return_all_experts=True)
        
        # Calculate losses
        total_loss = 0
        log_dict = {}
        
        # Expert losses
        for expert_name, logits in expert_outputs.items():
            if 'multitask' in expert_name:
                # Handle multi-task outputs
                outputs = self.experts[expert_name](x_struggling)
                
                # Phoneme loss
                phoneme_loss = F.cross_entropy(outputs['phoneme'], y_reduced)
                
                # Phonetic feature losses
                manner_labels = torch.tensor([self.phoneme_to_features[y_i.item()].get('manner', 0) 
                                             for y_i in y_struggling], 
                                            device=self.device, dtype=torch.long)
                place_labels = torch.tensor([self.phoneme_to_features[y_i.item()].get('place', 0) 
                                            for y_i in y_struggling], 
                                           device=self.device, dtype=torch.long)
                voicing_labels = torch.tensor([self.phoneme_to_features[y_i.item()].get('voicing', 0) 
                                              for y_i in y_struggling], 
                                             device=self.device, dtype=torch.long)
                
                manner_loss = F.cross_entropy(outputs['manner'], manner_labels)
                place_loss = F.cross_entropy(outputs['place'], place_labels)
                voicing_loss = F.cross_entropy(outputs['voicing'], voicing_labels)
                
                expert_loss = phoneme_loss + 0.3 * (manner_loss + place_loss + voicing_loss)
                log_dict[f'{expert_name}_loss'] = expert_loss.detach()
            else:
                expert_loss = F.cross_entropy(logits, y_reduced)
                log_dict[f'{expert_name}_loss'] = expert_loss.detach()
            
            total_loss += expert_loss
            
            # Track F1 for this expert
            with torch.no_grad():
                self.expert_metrics[expert_name].update(logits.detach(), y_reduced)
        
        # Binary gate loss (for all samples)
        is_struggling = torch.zeros(len(y), dtype=torch.long, device=self.device)
        is_struggling[struggling_mask] = 1
        gate_logits_all = self.binary_gate(x)
        gate_loss = F.cross_entropy(gate_logits_all, is_struggling)
        total_loss += gate_loss
        log_dict['gate_loss'] = gate_loss.detach()
        
        # Ensemble predictions
        ensemble_logits, _ = self.forward(x_struggling, return_all_experts=False)
        
        # Update metrics
        with torch.no_grad():
            self.train_f1.update(ensemble_logits.detach(), y_reduced)
            self.train_acc.update(ensemble_logits.detach(), y_reduced)
        
        log_dict['train_f1'] = self.train_f1.compute()
        log_dict['train_acc'] = self.train_acc.compute()
        log_dict['train_loss'] = total_loss.detach()
        
        self.log_dict(log_dict, prog_bar=True, on_step=True, on_epoch=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        # Ensure metrics are initialized
        if self.val_f1 is None:
            self.setup()
        
        # Filter to struggling phonemes
        struggling_mask = torch.tensor([y_i.item() in self.struggling_indices for y_i in y], 
                                      device=self.device, dtype=torch.bool)
        
        if struggling_mask.sum() == 0:
            return None
        
        x_struggling = x[struggling_mask]
        y_struggling = y[struggling_mask]
        y_reduced = torch.tensor([self.struggling_to_reduced[y_i.item()] for y_i in y_struggling], 
                                 device=self.device, dtype=torch.long)
        
        # Get predictions
        with torch.no_grad():
            expert_outputs, gate_logits = self.forward(x_struggling, return_all_experts=True)
            ensemble_logits, _ = self.forward(x_struggling, return_all_experts=False)
        
        # Calculate metrics for each expert
        results = {}
        for expert_name, logits in expert_outputs.items():
            f1 = self.expert_metrics[expert_name](logits.detach(), y_reduced)
            results[f'val_{expert_name}_f1'] = f1
        
        # Ensemble metrics
        self.val_f1.update(ensemble_logits.detach(), y_reduced)
        self.val_acc.update(ensemble_logits.detach(), y_reduced)
        
        results['val_f1_macro'] = self.val_f1.compute()
        results['val_acc'] = self.val_acc.compute()
        
        # Also log the raw specialist F1 for tracking
        results['val_specialist_f1'] = self.val_f1.compute()  # Track specialist performance
    
        self.log_dict(results, prog_bar=True, on_epoch=True)
    
    def on_train_epoch_end(self):
        """Reset training metrics at end of epoch"""
        if self.train_f1 is not None:
            self.train_f1.reset()
            self.train_acc.reset()
    
    def on_validation_epoch_end(self):
        """Print comparison of expert performances and reset metrics"""
        print("\n" + "="*60)
        print("EXPERT PERFORMANCE COMPARISON")
        print("="*60)
        
        expert_scores = {}
        for expert_name, metric in self.expert_metrics.items():
            score = metric.compute()
            expert_scores[expert_name] = score.item() if torch.is_tensor(score) else score
            metric.reset()
            print(f"{expert_name:30s}: F1 = {expert_scores[expert_name]:.4f}")
        
        # Find best expert
        if expert_scores:
            best_expert = max(expert_scores, key=expert_scores.get)
            print(f"\nBest Expert: {best_expert} (F1 = {expert_scores[best_expert]:.4f})")
        
        # Reset validation metrics
        if self.val_f1 is not None:
            self.val_f1.reset()
            self.val_acc.reset()
    
    def configure_optimizers(self):
        # Different learning rates for different components
        param_groups = [
            {'params': self.binary_gate.parameters(), 'lr': self.hparams.learning_rate},
            {'params': self.ensemble_weights, 'lr': self.hparams.learning_rate * 10}
        ]
        
        # Add expert parameters
        for expert_name, expert in self.experts.items():
            if 'multitask' in expert_name:
                param_groups.append({
                    'params': expert.parameters(),
                    'lr': self.hparams.learning_rate * 0.5
                })
            else:
                param_groups.append({
                    'params': expert.parameters(),
                    'lr': self.hparams.learning_rate
                })
        
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        # Different learning rates for different components
        param_groups = [
            {'params': self.binary_gate.parameters(), 'lr': self.hparams.learning_rate},
            {'params': self.ensemble_weights, 'lr': self.hparams.learning_rate * 10}
        ]
        
        # Add expert parameters with potentially different LRs
        for expert_name, expert in self.experts.items():
            if 'multitask' in expert_name:
                param_groups.append({
                    'params': expert.parameters(),
                    'lr': self.hparams.learning_rate * 0.5
                })
            else:
                param_groups.append({
                    'params': expert.parameters(),
                    'lr': self.hparams.learning_rate
                })
        
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}