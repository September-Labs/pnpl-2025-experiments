"""
Single-Stage MEG Model for Phoneme Classification

Enhanced with DeBERTa attention, class-balanced focal loss, supervised contrastive learning,
and IPA (International Phonetic Alphabet) feature prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import F1Score, Accuracy
from collections import defaultdict
from typing import Optional, Union, Tuple
import math

from .components import (
    DisentangledSelfAttention,
    MEGConformerLayer,
    normalize,
    focal_loss_mean_over_present_classes,
    supervised_nt_xent,
    get_ipa_feature_matrix
)


class BalancedPhonemePretrainer(nn.Module):
    """
    Pre-training module with temperature-based reweighting for handling
    imbalanced phoneme distributions.
    
    Args:
        vocab_size: Number of phoneme classes
        hidden_dim: Hidden dimension (not used in current implementation)
        temperature: Temperature for exponential weighting
    """
    
    def __init__(self, vocab_size: int = 39, hidden_dim: int = 16, temperature: float = 2.0):
        super().__init__()

        # Phoneme counts from training data (can be updated based on your dataset)
        phoneme_counts = {
            0: 216, 1: 443, 2: 1584, 3: 231, 4: 114, 5: 360, 6: 268, 7: 93,
            8: 772, 9: 504, 10: 477, 11: 429, 12: 231, 13: 282, 14: 119, 15: 428,
            16: 1052, 17: 570, 18: 63, 19: 430, 20: 566, 21: 518, 22: 1128, 23: 154,
            24: 226, 25: 14, 26: 276, 27: 634, 28: 743, 29: 113, 30: 1143, 31: 110,
            32: 96, 33: 236, 34: 326, 35: 428, 36: 151, 37: 456, 38: 7
        }

        total_count = sum(phoneme_counts.values())
        
        # Calculate temperature-based class weights
        self.class_weights = torch.zeros(vocab_size)
        for i, count in phoneme_counts.items():
            freq = count / total_count
            self.class_weights[i] = math.exp(-temperature * freq)
        
        # Normalize and clip weights
        self.class_weights = self.class_weights / self.class_weights.mean()
        self.class_weights = torch.clamp(self.class_weights, min=0.5, max=5.0)

class SingleStageMEGClassifier(L.LightningModule):
    """
    Single-stage phoneme classification model for MEG signals.
    
    Features:
    - DeBERTa-style disentangled attention
    - Class-balanced focal loss for imbalanced datasets
    - Supervised contrastive learning
    - IPA phonetic feature prediction as auxiliary task
    - Optional IPA similarity-based classification
    
    Args:
        meg_channels: Number of MEG sensor channels (default: 306)
        time_points: Number of time points in input
        vocab_size: Number of phoneme classes (default: 39)
        hidden_dim: Hidden dimension size
        num_conformers: Number of Conformer layers
        learning_rate: Base learning rate
        use_conformer: Whether to use Conformer or LSTM encoder
        loss_type: Type of loss function ('cross_entropy', 'focal', 'balanced_focal')
        focal_gamma: Focal loss gamma parameter
        dropout_rate: Dropout rate
        label_smoothing: Label smoothing factor
        weight_decay: Weight decay for regularization
        classifier_lr_multiplier: Learning rate multiplier for classifier
        warmup_epochs: Number of warmup epochs
        total_epochs: Total number of training epochs
        temperature: Temperature for class reweighting
        norm_type: Normalization type ('pre' or 'post')
        metric_type: Metric to optimize ('f1_macro' or 'balanced_acc')
        use_class_weights: Whether to use class weights in loss
        use_contrastive: Whether to use contrastive loss
        contrastive_weight: Weight for contrastive loss
        contrastive_temperature: Temperature for contrastive similarity
        embedding_dim: Dimension of embedding space
        projection_dim: Dimension of projection head output
        use_ipa_features: Whether to predict IPA features
        ipa_feature_weight: Weight for IPA feature loss
        ipa_hidden_dim: Hidden dimension for IPA predictor
        use_ipa_similarity_classification: Whether to use IPA similarity for classification
        ipa_similarity_metric: Similarity metric for IPA classification
    """
    
    def __init__(self,
                 meg_channels: int = 306,
                 time_points: int = 125,
                 vocab_size: int = 39,
                 hidden_dim: int = 128,
                 num_conformers: int = 4,
                 learning_rate: float = 1e-4,
                 use_conformer: bool = True,
                 loss_type: str = "balanced_focal",
                 focal_gamma: float = 2.0,
                 dropout_rate: float = 0.5,
                 label_smoothing: float = 0.12,
                 weight_decay: float = 0.001,
                 classifier_lr_multiplier: float = 1.5,
                 warmup_epochs: int = 5,
                 total_epochs: int = 200,
                 temperature: float = 1.0,
                 norm_type: str = "pre",
                 metric_type: str = "f1_macro",
                 use_class_weights: bool = False,
                 # Contrastive learning parameters
                 use_contrastive: bool = True,
                 contrastive_weight: float = 0.5,
                 contrastive_temperature: float = 0.03,
                 embedding_dim: int = 128,
                 projection_dim: int = 128,
                 # IPA feature prediction parameters
                 use_ipa_features: bool = True,
                 ipa_feature_weight: float = 0.2,
                 ipa_hidden_dim: int = 64,
                 # IPA similarity-based classification
                 use_ipa_similarity_classification: bool = False,
                 ipa_similarity_metric: str = "cosine",
                 # Additional parameters
                 use_rotary: bool = False,
                 enable_aug: bool = False,
                 use_tta: bool = True,
                 tta_num_augmentations: int = 10,
                 tta_temporal_shifts: list = None,
                 tta_channel_dropout: float = 0.05,
                 tta_noise_std: float = 0.01,
                 tta_weight_original: float = 2.0,
                 **kwargs):
        super().__init__()  
        self.save_hyperparameters()

        self.metric_type = metric_type
        print(f"Model initialized with metric_type: {self.metric_type}, loss_type: {loss_type}")
        print(f"Contrastive: {use_contrastive}, IPA features: {use_ipa_features}")
        
        # IPA similarity-based classification
        self.use_ipa_similarity_classification = use_ipa_similarity_classification
        self.ipa_similarity_metric = ipa_similarity_metric
        if use_ipa_similarity_classification and not use_ipa_features:
            raise ValueError("use_ipa_similarity_classification requires use_ipa_features=True")
        if use_ipa_similarity_classification:
            print(f"IPA similarity classification enabled with metric: {ipa_similarity_metric}")
        
        # Balanced pre-trainer for class weights
        self.pretrainer = BalancedPhonemePretrainer(vocab_size, hidden_dim, temperature)
        
        # IPA feature matrix
        self.use_ipa_features = use_ipa_features
        self.ipa_feature_weight = ipa_feature_weight
        if use_ipa_features:
            feature_matrix, feature_names = get_ipa_feature_matrix()
            self.register_buffer('ipa_feature_matrix', feature_matrix)  # [39, 14]
            self.num_ipa_features = feature_matrix.shape[1]
            self.feature_names = feature_names
            print(f"IPA features enabled: {self.num_ipa_features} features")
        
        # TTA parameters
        if tta_temporal_shifts is None:
            tta_temporal_shifts = [2, 4, 6]
        self.tta_temporal_shifts = tta_temporal_shifts
        
        # MEG encoder
        if use_conformer:
            self.input_projection = nn.Sequential(
                nn.Conv1d(meg_channels, hidden_dim, kernel_size=5, padding=2),
                nn.SiLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
            )
            
            self.input_skip = nn.Conv1d(meg_channels, hidden_dim, kernel_size=1)
            
            self.meg_encoder = nn.ModuleList([
                MEGConformerLayer(hidden_dim, 4, hidden_dim*2, 
                                dropout=dropout_rate, norm_type=norm_type) 
                for _ in range(num_conformers)
            ])
            
            self.encoder_output_dim = hidden_dim
        else:
            self.input_projection = None
            self.input_skip = None
            self.meg_encoder = nn.LSTM(
                meg_channels, hidden_dim, num_conformers,
                batch_first=True, bidirectional=True, dropout=dropout_rate
            )
            self.encoder_output_dim = hidden_dim * 2
        
        self.use_conformer = use_conformer
        
        # Feature aggregation
        self.feature_aggregator = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.encoder_output_dim * time_points, 2*embedding_dim),
            nn.SiLU(),
            nn.Linear(2*embedding_dim, embedding_dim),
            nn.Dropout(dropout_rate)
        )
        
        # Classification head
        self.classifier = nn.Linear(embedding_dim, vocab_size)
        
        # IPA feature prediction head
        if use_ipa_features:
            self.ipa_predictor = nn.Sequential(
                nn.Linear(embedding_dim, ipa_hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(ipa_hidden_dim, self.num_ipa_features)
            )
        
        # Contrastive learning
        self.use_contrastive = use_contrastive
        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature
        
        if use_contrastive:
            self.projection_head = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, projection_dim)
            )
        
        self.feature_norm = nn.LayerNorm(self.encoder_output_dim)
        
        # Loss parameters
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.use_class_weights = use_class_weights
        
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
        
        # IPA feature tracking
        if use_ipa_features:
            self.ipa_feature_accuracy = defaultdict(float)
            self.ipa_feature_counts = defaultdict(int)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features using the chosen encoder."""
        B, C, T = x.shape
        
        if self.use_conformer:
            features_main = self.input_projection(x)
            features_skip = self.input_skip(x)
            features = features_main + features_skip
            
            features = features.transpose(1, 2)
            
            for conformer in self.meg_encoder:
                features = conformer(features)
            
            features = self.feature_norm(features)
        else:
            x = x.transpose(1, 2)
            features, _ = self.meg_encoder(x)
        
        return features
    
    def _compute_ipa_similarity_logits(self, ipa_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute classification logits by comparing predicted IPA features
        to each phoneme's ground truth IPA feature vector.
        """
        B = ipa_logits.size(0)
        
        if self.ipa_similarity_metric == "cosine":
            ipa_probs = torch.sigmoid(ipa_logits)
            ipa_probs_norm = F.normalize(ipa_probs, p=2, dim=1)
            target_features_norm = F.normalize(self.ipa_feature_matrix.float(), p=2, dim=1)
            logits = torch.matmul(ipa_probs_norm, target_features_norm.T)
            logits = logits * 10.0
            
        elif self.ipa_similarity_metric == "dot":
            ipa_probs = torch.sigmoid(ipa_logits)
            logits = torch.matmul(ipa_probs, self.ipa_feature_matrix.T)
            
        elif self.ipa_similarity_metric == "hamming":
            ipa_preds = (torch.sigmoid(ipa_logits) > 0.5).float()
            ipa_preds_exp = ipa_preds.unsqueeze(1)
            target_exp = self.ipa_feature_matrix.unsqueeze(0)
            matches = (ipa_preds_exp == target_exp).float()
            logits = matches.sum(dim=2)
            logits = logits / self.num_ipa_features
            logits = logits * 10.0
            
        else:
            raise ValueError(f"Unknown similarity metric: {self.ipa_similarity_metric}")
        
        return logits
    
    def forward(self, x: torch.Tensor, return_all: bool = False) -> Union[torch.Tensor, Tuple]:
        """
        Forward pass for classification and optional auxiliary tasks.
        
        Args:
            x: Input MEG data [B, C, T]
            return_all: Whether to return all outputs
            
        Returns:
            logits if return_all=False, else (logits, proj_embeddings, ipa_logits)
        """
        B, C, T = x.shape
        with torch.no_grad():
            x = normalize(x)
        features = self.extract_features(x)
        
        # Flatten temporal features
        features_flat = features.reshape(B, -1)
        
        # Create embeddings
        embeddings = self.feature_aggregator(features_flat)
        
        # IPA feature predictions
        ipa_logits = None
        if self.use_ipa_features and (return_all or self.training or self.use_ipa_similarity_classification):
            ipa_logits = self.ipa_predictor(embeddings)
        
        # Classification logits
        if self.use_ipa_similarity_classification and self.use_ipa_features:
            logits = self._compute_ipa_similarity_logits(ipa_logits)
        else:
            logits = self.classifier(embeddings)
        
        outputs = [logits]
        
        # Contrastive embeddings
        if self.use_contrastive and (return_all or self.training):
            proj_embeddings = self.projection_head(embeddings)
            proj_embeddings = F.normalize(proj_embeddings, p=2, dim=1)
            outputs.append(proj_embeddings)
        else:
            outputs.append(None)
        
        # IPA feature predictions
        outputs.append(ipa_logits)
        
        if return_all or self.training:
            return tuple(outputs)
        
        return logits
    
    def compute_loss(self, logits, targets, embeddings=None, ipa_logits=None):
        """Compute combined loss (classification + optional contrastive + optional IPA features)."""
        # Classification loss
        if self.loss_type == "balanced_focal":
            alpha = self.pretrainer.class_weights if self.use_class_weights else None
            cls_loss = focal_loss_mean_over_present_classes(
                logits, targets,
                gamma=self.focal_gamma,
                alpha=alpha,
                label_smoothing=self.label_smoothing
            )
            
            if self.label_smoothing > 0:
                smooth_loss = F.cross_entropy(
                    logits, targets, 
                    label_smoothing=self.label_smoothing
                )
                cls_loss = 0.8 * cls_loss + 0.2 * smooth_loss
                
        elif self.loss_type == "focal":
            ce_loss = F.cross_entropy(logits, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = (1 - pt) ** self.focal_gamma * ce_loss
            
            if self.use_class_weights:
                weights = self.pretrainer.class_weights.to(logits.device)
                focal_loss = weights[targets] * focal_loss
            
            cls_loss = focal_loss.mean()
            
            if self.label_smoothing > 0:
                smooth_loss = F.cross_entropy(
                    logits, targets, 
                    label_smoothing=self.label_smoothing
                )
                cls_loss = 0.7 * cls_loss + 0.3 * smooth_loss
                
        else:
            if self.label_smoothing > 0:
                cls_loss = F.cross_entropy(
                    logits, targets, 
                    label_smoothing=self.label_smoothing
                )
            else:
                cls_loss = F.cross_entropy(logits, targets)
        
        total_loss = cls_loss
        loss_dict = {'cls_loss': cls_loss}
        
        # Contrastive loss
        if self.use_contrastive and embeddings is not None:
            contrastive_loss = supervised_nt_xent(
                embeddings, targets,
                temperature=self.contrastive_temperature
            )
            total_loss = total_loss + self.contrastive_weight * contrastive_loss
            loss_dict['contrastive_loss'] = contrastive_loss
        
        # IPA feature prediction loss
        if self.use_ipa_features and ipa_logits is not None:
            ipa_targets = self.ipa_feature_matrix[targets]
            ipa_loss = F.binary_cross_entropy_with_logits(
                ipa_logits, ipa_targets, reduction='mean'
            )
            total_loss = total_loss + self.ipa_feature_weight * ipa_loss
            loss_dict['ipa_loss'] = ipa_loss
            
            # Track per-feature accuracy
            with torch.no_grad():
                ipa_preds = (ipa_logits > 0).float()
                feature_correct = (ipa_preds == ipa_targets).float()
                
                for feat_idx in range(self.num_ipa_features):
                    self.ipa_feature_accuracy[feat_idx] += feature_correct[:, feat_idx].sum().item()
                    self.ipa_feature_counts[feat_idx] += targets.size(0)
        
        loss_dict['total_loss'] = total_loss
        return total_loss, loss_dict
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        logits, embeddings, ipa_logits = self(x, return_all=True)
        loss, loss_dict = self.compute_loss(logits, y, embeddings, ipa_logits)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_cls_loss', loss_dict['cls_loss'], prog_bar=False)
        if 'contrastive_loss' in loss_dict:
            self.log('train_contrastive_loss', loss_dict['contrastive_loss'], prog_bar=False)
        if 'ipa_loss' in loss_dict:
            self.log('train_ipa_loss', loss_dict['ipa_loss'], prog_bar=False)
        
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = (preds == y).float().mean()
            metric_value = self.train_metric(logits, y)
            
            for i in range(len(y)):
                phoneme_id = y[i].item()
                self.phoneme_counts[phoneme_id] += 1
                if preds[i] == y[i]:
                    self.phoneme_f1_scores[phoneme_id] += 1
        
        self.log(f'train_{self.metric_name}', metric_value, prog_bar=True)
        self.log('train_acc', acc)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        logits = self(x, return_all=False)
        loss, _ = self.compute_loss(logits, y)
        
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()
        metric_value = self.val_metric(logits, y)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log(f'val_{self.metric_name}', metric_value, prog_bar=True)
        self.log('val_acc', acc)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        
        logits = self(x, return_all=False)
        loss, _ = self.compute_loss(logits, y)
        
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()
        metric_value = self.test_metric(logits, y)
        
        self.log('test_loss', loss)
        self.log(f'test_{self.metric_name}', metric_value)
        self.log('test_acc', acc)
        
        return loss
    
    def on_train_epoch_end(self):
        """Log per-phoneme and per-feature performance statistics."""
        if self.current_epoch % 5 == 0:
            print(f"\n{'='*50}")
            print(f"Epoch {self.current_epoch} - Performance Summary:")
            print(f"Optimizing for: {self.metric_name} | Loss: {self.loss_type}")
            if self.use_contrastive:
                print(f"Contrastive: weight={self.contrastive_weight}")
            if self.use_ipa_features:
                print(f"IPA features: weight={self.ipa_feature_weight}")
            if self.use_ipa_similarity_classification:
                print(f"IPA similarity classification: {self.ipa_similarity_metric}")
            
            # Phoneme accuracies
            phoneme_accuracies = {}
            for phoneme_id in self.phoneme_counts:
                if self.phoneme_counts[phoneme_id] > 0:
                    accuracy = self.phoneme_f1_scores[phoneme_id] / self.phoneme_counts[phoneme_id]
                    phoneme_accuracies[phoneme_id] = accuracy
            
            if phoneme_accuracies:
                sorted_phonemes = sorted(phoneme_accuracies.items(), key=lambda x: x[1])
                
                print("\nWorst performing phonemes:")
                for pid, acc in sorted_phonemes[:5]:
                    print(f"  Phoneme {pid}: {acc:.3f} (n={self.phoneme_counts[pid]})")
                
                print("Best performing phonemes:")
                for pid, acc in sorted_phonemes[-5:]:
                    print(f"  Phoneme {pid}: {acc:.3f} (n={self.phoneme_counts[pid]})")
                
                balanced_acc = sum(phoneme_accuracies.values()) / len(phoneme_accuracies)
                print(f"Balanced accuracy: {balanced_acc:.3f}")
            
            # IPA feature accuracies
            if self.use_ipa_features and self.ipa_feature_counts:
                print("\nIPA Feature Prediction Accuracies:")
                feature_accs = []
                for feat_idx in range(self.num_ipa_features):
                    if self.ipa_feature_counts[feat_idx] > 0:
                        acc = self.ipa_feature_accuracy[feat_idx] / self.ipa_feature_counts[feat_idx]
                        feature_accs.append((self.feature_names[feat_idx], acc))
                
                feature_accs.sort(key=lambda x: x[1])
                for feat_name, acc in feature_accs[:5]:
                    print(f"  {feat_name}: {acc:.3f} (worst)")
                print("  ...")
                for feat_name, acc in feature_accs[-5:]:
                    print(f"  {feat_name}: {acc:.3f} (best)")
                
                avg_feature_acc = sum(a for _, a in feature_accs) / len(feature_accs)
                print(f"Average IPA feature accuracy: {avg_feature_acc:.3f}")
            
            print(f"{'='*50}\n")
            
            # Reset counters
            self.phoneme_f1_scores = defaultdict(float)
            self.phoneme_counts = defaultdict(int)
            if self.use_ipa_features:
                self.ipa_feature_accuracy = defaultdict(float)
                self.ipa_feature_counts = defaultdict(int)
    
    def configure_optimizers(self):
        """Configure optimizer with different learning rates."""
        params = []
        
        if self.use_conformer and self.input_projection is not None:
            params.append({
                'params': list(self.input_projection.parameters()) + 
                         list(self.input_skip.parameters()), 
                'lr': self.hparams.learning_rate
            })
        
        params.append({
            'params': self.meg_encoder.parameters(), 
            'lr': self.hparams.learning_rate
        })
        
        # Feature aggregator and classifier
        params.append({
            'params': list(self.feature_aggregator.parameters()) + 
                     list(self.classifier.parameters()), 
            'lr': self.hparams.learning_rate * self.hparams.classifier_lr_multiplier
        })
        
        # Projection head
        if self.use_contrastive and hasattr(self, 'projection_head'):
            params.append({
                'params': self.projection_head.parameters(),
                'lr': self.hparams.learning_rate * self.hparams.classifier_lr_multiplier
            })
        
        # IPA predictor
        if self.use_ipa_features and hasattr(self, 'ipa_predictor'):
            params.append({
                'params': self.ipa_predictor.parameters(),
                'lr': self.hparams.learning_rate * self.hparams.classifier_lr_multiplier
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
