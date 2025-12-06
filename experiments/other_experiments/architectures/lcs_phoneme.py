"""
Enhanced MEG Model - Compatible with existing train.py infrastructure
Place this file in: models/architectures/enhanced_meg.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from torchmetrics import F1Score
from collections import defaultdict


class SensorAwareAttention(nn.Module):
    """Attention mechanism that uses sensor importance for each phoneme."""
    
    def __init__(self, 
                 n_sensors: int = 306,
                 n_phonemes: int = 39,
                 hidden_dim: int = 256):
        super().__init__()
        self.n_sensors = n_sensors
        self.n_phonemes = n_phonemes
        self.hidden_dim = hidden_dim
        
        # Learnable sensor importance (will be initialized from analysis if available)
        self.sensor_importance = nn.Parameter(torch.ones(n_phonemes, n_sensors) / n_sensors)
        
        # Temperature for attention
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # Attention refinement
        self.attention_refiner = nn.Sequential(
            nn.Linear(n_sensors, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_sensors),
            nn.Sigmoid()
        )
        
        # Projection
        self.projection = nn.Conv1d(n_sensors, hidden_dim, kernel_size=1)
    
    def forward(self, x: torch.Tensor):
        """Apply sensor-aware attention."""
        B, C, T = x.shape
        
        # Compute attention weights
        x_mean = x.mean(dim=2)  # (B, C)
        refined_attention = self.attention_refiner(x_mean)  # (B, C)
        
        # Apply attention
        x_attended = x * refined_attention.unsqueeze(2)
        
        # Project to hidden dimension
        output = self.projection(x_attended)  # (B, hidden_dim, T)
        
        return output


class PhonemeGroupExpert(nn.Module):
    """Expert network for a specific group of phonemes."""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 output_dim: int = 39,
                 dropout: float = 0.2):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Temporal pooling
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        # Encode
        features = self.encoder(x)
        # Pool
        pooled = self.temporal_pool(features).squeeze(-1)
        # Classify
        output = self.classifier(pooled)
        return output


class EnhancedMEGModel(L.LightningModule):
    """Enhanced MEG model compatible with existing train.py."""
    
    def __init__(self,
                 # Standard parameters expected by train.py
                 meg_channels: int = 306,
                 time_points: int = 125,
                 vocab_size: int = 39,
                 learning_rate: float = 1e-4,
                 
                 # Enhanced model parameters
                 hidden_dim: int = 512,
                 n_experts: int = 4,
                 dropout: float = 0.2,
                 label_smoothing: float = 0.1,
                 
                 # Optional paths for loading analysis
                 analysis_json_path: Optional[str] = None,
                 phoneme_info_path: Optional[str] = None,
                 
                 # Training strategy
                 use_focal_loss: bool = True,
                 focal_gamma: float = 2.0,
                 rare_phoneme_boost: float = 2.0,
                 
                 **kwargs):  # Accept any additional parameters
        
        super().__init__()
        self.save_hyperparameters()
        
        # Load analysis results if available
        self.phoneme_frequencies = None
        self._load_external_data(analysis_json_path, phoneme_info_path)
        
        # Sensor-aware attention
        self.sensor_attention = SensorAwareAttention(
            n_sensors=meg_channels,
            n_phonemes=vocab_size,
            hidden_dim=hidden_dim
        )
        
        # Global encoder (similar to your original MEG encoder)
        self.global_encoder = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Multi-expert system
        self.experts = nn.ModuleList([
            PhonemeGroupExpert(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim // 2,
                output_dim=vocab_size,
                dropout=dropout
            ) for _ in range(n_experts)
        ])
        
        # Expert gating
        self.expert_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_experts),
            nn.Softmax(dim=-1)
        )
        
        # Global classifier
        self.global_classifier = nn.Sequential(
            nn.Linear(hidden_dim * time_points, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size)
        )
        
        # Ensemble weight
        self.ensemble_weight = nn.Parameter(torch.tensor(0.5))
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # Focal loss components
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        
        # Class weights for rare phonemes
        self.register_buffer('class_weights', self._compute_class_weights(rare_phoneme_boost))
        
        # Metrics
        self.f1_macro = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
        self.f1_per_class = F1Score(num_classes=vocab_size, average='none', task="multiclass")
    
    def _load_external_data(self, analysis_path: Optional[str], phoneme_path: Optional[str]):
        """Load analysis results and phoneme information if available."""
        # Try to load sensor importance from analysis
        if analysis_path and Path(analysis_path).exists():
            try:
                with open(analysis_path, 'r') as f:
                    analysis = json.load(f)
                
                # Extract sensor importance matrix
                if 'phoneme_sensor_importance' in analysis:
                    importance_matrix = torch.zeros(self.hparams.vocab_size, self.hparams.meg_channels)
                    
                    # This is simplified - you'd need to map phoneme names to indices properly
                    for i, (phoneme_name, data) in enumerate(analysis['phoneme_sensor_importance'].items()):
                        if i < self.hparams.vocab_size and 'all_sensors' in data:
                            importance_matrix[i] = torch.tensor(data['all_sensors'][:self.hparams.meg_channels])
                    
                    # Update the attention module's importance
                    self.sensor_attention.sensor_importance.data = importance_matrix
                    print(f"Loaded sensor importance from {analysis_path}")
            except Exception as e:
                print(f"Could not load analysis from {analysis_path}: {e}")
        
        # Try to load phoneme frequencies
        if phoneme_path and Path(phoneme_path).exists():
            try:
                with open(phoneme_path, 'r') as f:
                    phoneme_data = json.load(f)
                
                if 'phoneme_frequencies' in phoneme_data:
                    self.phoneme_frequencies = list(phoneme_data['phoneme_frequencies'].values())
                    print(f"Loaded phoneme frequencies from {phoneme_path}")
            except Exception as e:
                print(f"Could not load phoneme info from {phoneme_path}: {e}")
    
    def _compute_class_weights(self, boost_factor: float):
        """Compute class weights based on frequency."""
        if self.phoneme_frequencies is not None:
            frequencies = np.array(self.phoneme_frequencies)
            # Inverse frequency weighting
            weights = 1.0 / (frequencies + 0.001)
            weights = weights / weights.mean()
            
            # Boost rare phonemes
            median_freq = np.median(frequencies)
            rare_mask = frequencies < median_freq
            weights[rare_mask] *= boost_factor
            
            return torch.FloatTensor(weights)
        else:
            # Uniform weights if no frequency information
            return torch.ones(self.hparams.vocab_size)
    
    def focal_loss(self, inputs, targets):
        """Focal loss for handling class imbalance."""
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.focal_gamma * ce_loss
        
        # Apply class weights
        if self.class_weights is not None:
            focal_loss = self.class_weights[targets] * focal_loss
        
        return focal_loss.mean()
    
    def forward(self, x):
        """Forward pass compatible with existing infrastructure."""
        B, C, T = x.shape
        
        # Apply sensor-aware attention
        x_attended = self.sensor_attention(x)  # (B, hidden_dim, T)
        
        # Global encoding
        global_features = self.global_encoder(x_attended)  # (B, hidden_dim, T)
        
        # Global prediction
        global_features_flat = global_features.reshape(B, -1)
        global_pred = self.global_classifier(global_features_flat)
        
        # Expert predictions
        expert_outputs = []
        for expert in self.experts:
            expert_pred = expert(global_features)
            expert_outputs.append(expert_pred)
        
        # Stack expert outputs
        expert_preds = torch.stack(expert_outputs, dim=1)  # (B, n_experts, vocab_size)
        
        # Compute gating weights
        pooled_features = global_features.mean(dim=2)  # (B, hidden_dim)
        gate_weights = self.expert_gate(pooled_features)  # (B, n_experts)
        
        # Weighted sum of expert predictions
        expert_ensemble = torch.einsum('bev,be->bv', expert_preds, gate_weights)
        
        # Final ensemble
        ensemble_weight = torch.sigmoid(self.ensemble_weight)
        final_pred = ensemble_weight * global_pred + (1 - ensemble_weight) * expert_ensemble
        
        return final_pred
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        # Compute losses
        ce_loss = self.ce_loss(y_hat, y)
        
        if self.use_focal_loss:
            focal_loss = self.focal_loss(y_hat, y)
            loss = 0.6 * ce_loss + 0.4 * focal_loss
            self.log('train_focal_loss', focal_loss)
        else:
            loss = ce_loss
        
        # Metrics
        f1_macro = self.f1_macro(y_hat, y)
        
        # Logging
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_ce_loss', ce_loss)
        self.log('train_f1_macro', f1_macro, prog_bar=True)
        
        # Log rare phoneme performance periodically
        if batch_idx % 100 == 0 and self.phoneme_frequencies is not None:
            f1_per_class = self.f1_per_class(y_hat, y)
            frequencies = np.array(self.phoneme_frequencies)
            rare_indices = np.argsort(frequencies)[:10]
            rare_f1 = f1_per_class[rare_indices].mean()
            self.log('train_rare_f1', rare_f1)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        loss = self.ce_loss(y_hat, y)
        f1_macro = self.f1_macro(y_hat, y)
        
        self.log('val_loss', loss)
        self.log('val_f1_macro', f1_macro, prog_bar=True)
        
        # Compute rare phoneme performance
        if self.phoneme_frequencies is not None:
            f1_per_class = self.f1_per_class(y_hat, y)
            frequencies = np.array(self.phoneme_frequencies)
            rare_indices = np.argsort(frequencies)[:10]
            common_indices = np.argsort(frequencies)[-10:]
            
            rare_f1 = f1_per_class[rare_indices].mean()
            common_f1 = f1_per_class[common_indices].mean()
            
            self.log('val_rare_f1', rare_f1, prog_bar=True)
            self.log('val_common_f1', common_f1)
            self.log('val_f1_gap', common_f1 - rare_f1)
        
        return loss
    
    def configure_optimizers(self):
        # Different learning rates for different components
        params = [
            {'params': self.sensor_attention.parameters(), 'lr': self.hparams.learning_rate * 0.5},
            {'params': self.global_encoder.parameters(), 'lr': self.hparams.learning_rate},
            {'params': self.global_classifier.parameters(), 'lr': self.hparams.learning_rate},
            {'params': self.expert_gate.parameters(), 'lr': self.hparams.learning_rate},
        ]
        
        # Experts get higher learning rate
        for expert in self.experts:
            params.append({'params': expert.parameters(), 'lr': self.hparams.learning_rate * 1.5})
        
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