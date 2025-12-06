"""
Enhanced Temporal Triangulation MEG Model with Transition Modeling
Captures phoneme bleed-over and uses transition probabilities to boost predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import F1Score, Accuracy
from collections import defaultdict
from typing import Optional, Union, Tuple, Iterable, Dict
import math
import numpy as np
import json
from pathlib import Path

# Import base components from temporal_triangulation
from .temporal_triangulation import (
    DisentangledSelfAttention,
    BalancedPhonemePretrainer,
    MEGConformerLayer,
    prepare_attention_mask,
    make_log_bucket_position,
    build_relative_position,
    scaled_size_sqrt,
    build_rpos
)

class PhonemeTransitionModel(nn.Module):
    """
    Models phoneme transition probabilities and boundary effects.
    Uses learned transition patterns to refine predictions.
    """
    
    def __init__(self, vocab_size: int = 39, hidden_dim: int = 64):
        super().__init__()
        
        # Learned transition matrix (can be initialized from data statistics)
        self.transition_logits = nn.Parameter(torch.randn(vocab_size, vocab_size) * 0.1)
        
        # Boundary effect modeling - how much each phoneme affects boundaries
        self.boundary_influence = nn.Parameter(torch.ones(vocab_size) * 0.5)
        
        # Context refinement network
        self.context_refiner = nn.Sequential(
            nn.Linear(vocab_size * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, vocab_size)
        )
        
        # Initialize with known statistics if available
        self._load_transition_statistics()
    
    def _load_transition_statistics(self):
        """Load pre-computed transition statistics if available."""
        stats_path = Path("phoneme_transition_analysis/transition_matrix.npy")
        if stats_path.exists():
            print("Loading pre-computed transition matrix...")
            transition_matrix = np.load(stats_path)
            # Convert probabilities to logits
            transition_matrix = np.clip(transition_matrix, 1e-8, 1 - 1e-8)
            logits = np.log(transition_matrix / (1 - transition_matrix))
            self.transition_logits.data = torch.tensor(logits, dtype=torch.float32)
    
    def get_transition_probs(self):
        """Get transition probability matrix."""
        return F.softmax(self.transition_logits, dim=1)
    
    def apply_transition_correction(self, 
                                   first_logits: torch.Tensor,
                                   middle_logits: torch.Tensor, 
                                   last_logits: torch.Tensor) -> torch.Tensor:
        """
        Apply transition-based correction to predictions.
        
        The idea: if the first third suggests phoneme A and the middle suggests B,
        the transition probability A->B should influence our confidence in B.
        Similarly for B->C with middle and last thirds.
        """
        batch_size = first_logits.shape[0]
        
        # Get probability distributions
        first_probs = F.softmax(first_logits, dim=1)
        middle_probs = F.softmax(middle_logits, dim=1)
        last_probs = F.softmax(last_logits, dim=1)
        
        # Get transition probability matrix
        trans_probs = self.get_transition_probs()
        
        # Calculate transition-weighted middle predictions
        # P(middle|first) = sum over all first phonemes of P(first) * P(first->middle)
        first_to_middle = torch.matmul(first_probs, trans_probs)
        
        # Calculate transition-weighted last predictions  
        # P(last|middle) = sum over all middle phonemes of P(middle) * P(middle->last)
        middle_to_last = torch.matmul(middle_probs, trans_probs)
        
        # Combine with original predictions
        # The middle is most reliable, but we adjust based on transition coherence
        refined_middle = middle_probs * (1 + 0.2 * first_to_middle)
        refined_middle = refined_middle / refined_middle.sum(dim=1, keepdim=True)
        
        # Context-aware refinement using all three segments
        context = torch.cat([first_probs, middle_probs, last_probs], dim=1)
        context_adjustment = self.context_refiner(context)
        
        # Final prediction combines refined middle with context adjustment
        final_logits = torch.log(refined_middle + 1e-8) + 0.3 * context_adjustment
        
        return final_logits

class BoundaryAwareEncoder(nn.Module):
    """
    Encoder that explicitly models boundary regions differently.
    Understands that edges contain transition information.
    """
    
    def __init__(self, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        # Separate processing for boundary vs center regions
        self.boundary_encoder = nn.ModuleList([
            MEGConformerLayer(hidden_dim, 4, hidden_dim*2, dropout=dropout, norm_type="pre")
            for _ in range(num_layers)
        ])
        
        self.center_encoder = nn.ModuleList([
            MEGConformerLayer(hidden_dim, 4, hidden_dim*2, dropout=dropout, norm_type="pre")
            for _ in range(num_layers)
        ])
        
        # Attention weights for boundary vs center
        self.boundary_weight = nn.Parameter(torch.tensor(0.3))
        self.center_weight = nn.Parameter(torch.tensor(0.7))
        
    def forward(self, features: torch.Tensor, is_boundary: bool = False) -> torch.Tensor:
        """
        Process features with boundary awareness.
        
        Args:
            features: Input features (B, T, D)
            is_boundary: Whether this is a boundary region
        """
        if is_boundary:
            for layer in self.boundary_encoder:
                features = layer(features)
            weight = torch.sigmoid(self.boundary_weight)
        else:
            for layer in self.center_encoder:
                features = layer(features)
            weight = torch.sigmoid(self.center_weight)
        
        return features * weight

class EnhancedTemporalTriangulationMEGClassifier(L.LightningModule):
    """
    Enhanced temporal triangulation with explicit transition modeling and boundary awareness.
    """
    
    def __init__(self,
                 meg_channels: int = 306,
                 time_points: int = 125,
                 vocab_size: int = 39,
                 hidden_dim: int = 64,
                 num_conformers: int = 8,
                 learning_rate: float = 3e-4,
                 use_conformer: bool = True,
                 loss_type: str = "focal",
                 focal_gamma: float = 4.5,
                 dropout_rate: float = 0.02,
                 label_smoothing: float = 0.0,
                 weight_decay: float = 0.0,
                 classifier_lr_multiplier: float = 1.0,
                 warmup_epochs: int = 5,
                 total_epochs: int = 300,
                 temperature: float = 8.0,
                 norm_type: str = "pre",
                 metric_type: str = "f1_macro",
                 use_transition_model: bool = True,
                 boundary_size: int = 10,  # Time points at boundaries
                 use_swa: bool = False):
        super().__init__()
        self.save_hyperparameters()
        
        # Store parameters
        self.metric_type = metric_type
        self.use_transition_model = use_transition_model
        self.boundary_size = boundary_size
        self.use_swa = use_swa
        
        # Calculate temporal segments with explicit boundary regions
        self.time_points = time_points
        
        # Define boundary and center regions
        self.first_boundary = slice(0, boundary_size)
        self.last_boundary = slice(time_points - boundary_size, time_points)
        self.center_region = slice(boundary_size, time_points - boundary_size)
        
        # Three overlapping segments for triangulation
        segment_size = time_points // 3
        self.first_third = slice(0, segment_size + boundary_size)  # Include some overlap
        self.middle_third = slice(segment_size - boundary_size, 2 * segment_size + boundary_size)
        self.last_third = slice(2 * segment_size - boundary_size, time_points)
        
        # Store actual sizes for classifiers
        self.first_segment_size = segment_size + boundary_size
        self.middle_segment_size = segment_size + 2 * boundary_size
        self.last_segment_size = time_points - (2 * segment_size - boundary_size)
        
        # Balanced pre-trainer
        self.pretrainer = BalancedPhonemePretrainer(vocab_size, hidden_dim, temperature)
        
        # Transition model for phoneme co-occurrence patterns
        if use_transition_model:
            self.transition_model = PhonemeTransitionModel(vocab_size, hidden_dim)
        
        # Shared initial encoder
        if use_conformer:
            self.input_projection = nn.Sequential(
                nn.Conv1d(meg_channels, hidden_dim, kernel_size=5, padding=2),
                nn.SiLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
            )
            self.input_skip = nn.Conv1d(meg_channels, hidden_dim, kernel_size=1)
            
            # Shared encoder
            self.shared_encoder = nn.ModuleList([
                MEGConformerLayer(hidden_dim, 4, hidden_dim*2, dropout=dropout_rate, norm_type=norm_type)
                for _ in range(num_conformers // 3)
            ])
            
            # Boundary-aware encoders for each segment
            self.first_encoder = BoundaryAwareEncoder(hidden_dim, num_conformers // 3, dropout_rate)
            self.middle_encoder = BoundaryAwareEncoder(hidden_dim, num_conformers // 3, dropout_rate)
            self.last_encoder = BoundaryAwareEncoder(hidden_dim, num_conformers // 3, dropout_rate)
            
            self.encoder_output_dim = hidden_dim
        else:
            # LSTM alternative (simplified)
            self.input_projection = None
            self.input_skip = None
            self.shared_encoder = nn.LSTM(
                meg_channels, hidden_dim, num_conformers // 3,
                batch_first=True, bidirectional=True, dropout=dropout_rate
            )
            self.encoder_output_dim = hidden_dim * 2
        
        self.use_conformer = use_conformer
        
        # Feature normalization
        self.feature_norm = nn.LayerNorm(self.encoder_output_dim)
        
        # Classifiers for each segment
        self.first_classifier = nn.Sequential(
            nn.Linear(self.encoder_output_dim * self.first_segment_size, 128),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, vocab_size)
        )
        
        self.middle_classifier = nn.Sequential(
            nn.Linear(self.encoder_output_dim * self.middle_segment_size, 128),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, vocab_size)
        )
        
        self.last_classifier = nn.Sequential(
            nn.Linear(self.encoder_output_dim * self.last_segment_size, 128),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, vocab_size)
        )
        
        # Learnable combination weights with boundary awareness
        self.combination_network = nn.Sequential(
            nn.Linear(vocab_size * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, vocab_size)
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
        
        # Per-phoneme tracking
        self.phoneme_f1_scores = defaultdict(float)
        self.phoneme_counts = defaultdict(int)
    
    def extract_shared_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract shared features with boundary awareness."""
        B, C, T = x.shape
        
        if self.use_conformer:
            features_main = self.input_projection(x)
            features_skip = self.input_skip(x)
            features = features_main + features_skip
            features = features.transpose(1, 2)  # (B, T, hidden_dim)
            
            for conformer in self.shared_encoder:
                features = conformer(features)
        else:
            x = x.transpose(1, 2)
            features, _ = self.shared_encoder(x)
        
        return features
    
    def extract_temporal_features(self, shared_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract temporal features with boundary awareness."""
        # Split with overlapping segments
        first_features = shared_features[:, self.first_third, :]
        middle_features = shared_features[:, self.middle_third, :]
        last_features = shared_features[:, self.last_third, :]
        
        if self.use_conformer:
            # First segment has boundary at the start
            first_features = self.first_encoder(first_features, is_boundary=True)
            
            # Middle segment is mostly center
            middle_features = self.middle_encoder(middle_features, is_boundary=False)
            
            # Last segment has boundary at the end
            last_features = self.last_encoder(last_features, is_boundary=True)
            
            # Apply normalization
            first_features = self.feature_norm(first_features)
            middle_features = self.feature_norm(middle_features)
            last_features = self.feature_norm(last_features)
        
        return first_features, middle_features, last_features
    
    def forward(self, x: torch.Tensor, return_all: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward pass with transition-aware prediction refinement."""
        B = x.shape[0]
        
        # Extract shared features
        shared_features = self.extract_shared_features(x)
        
        # Extract temporal-specific features
        first_features, middle_features, last_features = self.extract_temporal_features(shared_features)
        
        # Flatten and classify
        first_flat = first_features.reshape(B, -1)
        middle_flat = middle_features.reshape(B, -1)
        last_flat = last_features.reshape(B, -1)
        
        first_logits = self.first_classifier(first_flat)
        middle_logits = self.middle_classifier(middle_flat)
        last_logits = self.last_classifier(last_flat)
        
        # Apply transition model if enabled
        if self.use_transition_model and hasattr(self, 'transition_model'):
            refined_logits = self.transition_model.apply_transition_correction(
                first_logits, middle_logits, last_logits
            )
        else:
            # Simple weighted combination
            refined_logits = 0.2 * first_logits + 0.6 * middle_logits + 0.2 * last_logits
        
        # Additional refinement through combination network
        all_probs = torch.cat([
            F.softmax(first_logits, dim=1),
            F.softmax(middle_logits, dim=1),
            F.softmax(last_logits, dim=1)
        ], dim=1)
        
        combination_adjustment = self.combination_network(all_probs)
        combined_logits = refined_logits + 0.2 * combination_adjustment
        
        # For inference, return only combined
        if not return_all or not self.training:
            return combined_logits
        
        return combined_logits, first_logits, middle_logits, last_logits
    
    def compute_loss(self, logits, targets):
        """Compute loss based on configured loss type."""
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
        
        combined_logits, first_logits, middle_logits, last_logits = self(x, return_all=True)
        
        # Compute losses
        combined_loss = self.compute_loss(combined_logits, y)
        first_loss = self.compute_loss(first_logits, y)
        middle_loss = self.compute_loss(middle_logits, y)
        last_loss = self.compute_loss(last_logits, y)
        
        # Weighted total loss
        total_loss = 0.5 * combined_loss + 0.15 * first_loss + 0.2 * middle_loss + 0.15 * last_loss
        
        # Add transition regularization if using transition model
        if self.use_transition_model and hasattr(self, 'transition_model'):
            # Encourage smooth transition probabilities
            trans_probs = self.transition_model.get_transition_probs()
            trans_reg = -0.01 * (trans_probs * torch.log(trans_probs + 1e-8)).sum()
            total_loss = total_loss + trans_reg
        
        with torch.no_grad():
            preds = combined_logits.argmax(dim=-1)
            acc = (preds == y).float().mean()
            metric_value = self.train_metric(combined_logits, y)
            
            for i in range(len(y)):
                phoneme_id = y[i].item()
                self.phoneme_counts[phoneme_id] += 1
                if preds[i] == y[i]:
                    self.phoneme_f1_scores[phoneme_id] += 1
        
        self.log('train_loss', total_loss, prog_bar=True)
        self.log(f'train_{self.metric_name}', metric_value, prog_bar=True)
        self.log('train_acc', acc)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        result = self(x, return_all=True)
        if isinstance(result, tuple):
            combined_logits, first_logits, middle_logits, last_logits = result
        else:
            combined_logits = result
            first_logits = middle_logits = last_logits = combined_logits
        
        combined_loss = self.compute_loss(combined_logits, y)
        
        preds = combined_logits.argmax(dim=-1)
        acc = (preds == y).float().mean()
        metric_value = self.val_metric(combined_logits, y)
        
        self.log('val_loss', combined_loss, prog_bar=True)
        self.log(f'val_{self.metric_name}', metric_value, prog_bar=True)
        self.log('val_acc', acc)
        
        return combined_loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        
        result = self(x, return_all=True)
        if isinstance(result, tuple):
            combined_logits, _, _, _ = result
        else:
            combined_logits = result
        
        loss = self.compute_loss(combined_logits, y)
        
        preds = combined_logits.argmax(dim=-1)
        acc = (preds == y).float().mean()
        metric_value = self.test_metric(combined_logits, y)
        
        self.log('test_loss', loss)
        self.log(f'test_{self.metric_name}', metric_value)
        self.log('test_acc', acc)
        
        return loss
    
    def on_train_epoch_end(self):
        """Log per-phoneme performance and transition learning progress."""
        if self.current_epoch % 5 == 0:
            print(f"\n{'='*50}")
            print(f"Epoch {self.current_epoch} - Performance Summary:")
            
            if self.use_transition_model and hasattr(self, 'transition_model'):
                # Show top transition patterns learned
                trans_probs = self.transition_model.get_transition_probs().detach().cpu().numpy()
                top_transitions = []
                for i in range(39):
                    for j in range(39):
                        if trans_probs[i, j] > 0.15:  # Significant transitions
                            top_transitions.append((i, j, trans_probs[i, j]))
                
                top_transitions.sort(key=lambda x: x[2], reverse=True)
                print("\nTop learned transitions:")
                for from_p, to_p, prob in top_transitions[:5]:
                    print(f"  {from_p} -> {to_p}: {prob:.3f}")
            
            # Reset tracking
            self.phoneme_f1_scores = defaultdict(float)
            self.phoneme_counts = defaultdict(int)
    
    def configure_optimizers(self):
        """Configure optimizer with separate learning rates."""
        params = []
        
        # Encoder parameters
        if self.use_conformer and self.input_projection is not None:
            params.append({
                'params': list(self.input_projection.parameters()) +
                         list(self.input_skip.parameters()),
                'lr': self.hparams.learning_rate
            })
        
        params.append({
            'params': list(self.shared_encoder.parameters()) +
                     list(self.first_encoder.parameters()) +
                     list(self.middle_encoder.parameters()) +
                     list(self.last_encoder.parameters()),
            'lr': self.hparams.learning_rate
        })
        
        # Classifier parameters
        params.append({
            'params': list(self.first_classifier.parameters()) +
                     list(self.middle_classifier.parameters()) +
                     list(self.last_classifier.parameters()),
            'lr': self.hparams.learning_rate * self.hparams.classifier_lr_multiplier
        })
        
        # Transition model parameters (slower learning)
        if self.use_transition_model and hasattr(self, 'transition_model'):
            params.append({
                'params': self.transition_model.parameters(),
                'lr': self.hparams.learning_rate * 0.5
            })
        
        # Combination network
        params.append({
            'params': self.combination_network.parameters(),
            'lr': self.hparams.learning_rate * 0.5
        })
        
        optimizer = torch.optim.AdamW(params, weight_decay=self.hparams.weight_decay)
        
        def lr_lambda(epoch):
            if epoch < self.hparams.warmup_epochs:
                return (epoch + 1) / self.hparams.warmup_epochs
            else:
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