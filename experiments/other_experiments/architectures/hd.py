"""
Hyperdimensional Computing model for MEG phoneme classification
Compatible with Lightning training framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
import numpy as np

# Hyperdimensional computing
import torchhd
from torchhd import embeddings
from torchhd import functional as hd_func


class HyperdimensionalModel(L.LightningModule):
    """Lightning module for hyperdimensional computing on MEG data"""
    
    def __init__(
        self,
        time_points: int,
        num_channels: int = 306,
        num_classes: int = 39,
        d: int = 10000,  # Hypervector dimension
        quantization_levels: int = 100,
        dropout: float = 0.2,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        use_prototypes: bool = False,
        label_smoothing: float = 0.0,
        metric_type: str = 'f1_macro',
        # Additional parameters that might come from config
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model parameters
        self.d = d
        self.num_channels = num_channels
        self.time_points = time_points
        self.num_classes = num_classes
        self.quantization_levels = quantization_levels
        self.use_prototypes = use_prototypes
        self.metric_type = metric_type
        self.validation_step_outputs = []
        self.validation_step_result = []
        self.test_step_result = []
        self.training_step_result = []

        # Training parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Loss function with optional label smoothing
        if label_smoothing > 0:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Initialize hyperdimensional components
        self._build_model()
        
    def _build_model(self):
        """Build the hyperdimensional model components"""
        
        # Channel encoders - one hypervector per channel
        self.channel_hvs = embeddings.Random(
            self.num_channels, self.d, dtype=torch.float32
        )
        
        # Time position encoders
        self.time_hvs = embeddings.Level(
            self.time_points, self.d, dtype=torch.float32
        )
        
        # Feature value encoder for continuous MEG values
        self.value_encoder = embeddings.Level(
            self.quantization_levels, self.d, dtype=torch.float32
        )
        
        # Learnable transformations
        self.input_projection = nn.Sequential(
            nn.Linear(self.d, self.d),
            nn.LayerNorm(self.d),
            nn.ReLU(inplace=True),
            nn.Dropout(self.hparams.dropout)
        )
        
        if self.use_prototypes:
            # Class prototypes for similarity-based classification
            self.class_prototypes = nn.Parameter(torch.randn(self.num_classes, self.d))
            nn.init.xavier_normal_(self.class_prototypes)
        else:
            # Standard classification head
            self.class_projection = nn.Sequential(
                nn.Linear(self.d, self.d // 2),
                nn.LayerNorm(self.d // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(self.hparams.dropout),
                nn.Linear(self.d // 2, self.d // 4),
                nn.LayerNorm(self.d // 4),
                nn.ReLU(inplace=True),
                nn.Linear(self.d // 4, self.num_classes)
            )
    
    def normalize_meg_values(self, x, channel_means=None, channel_stds=None):
        """
        Normalize MEG values using global statistics (compatible with standard approach)
        
        Args:
            x: Input tensor of shape (batch_size, num_channels, time_points)
            channel_means: Pre-computed channel means for standardization
            channel_stds: Pre-computed channel stds for standardization
        """
        if channel_means is not None and channel_stds is not None:
            # Use provided statistics (standard approach)
            x_normalized = (x - channel_means.unsqueeze(0).unsqueeze(2)) / (
                channel_stds.unsqueeze(0).unsqueeze(2) + 1e-8
            )
        else:
            # Fallback to per-sample normalization
            batch_size = x.shape[0]
            x_flat = x.view(batch_size, -1)
            mean = x_flat.mean(dim=1, keepdim=True).unsqueeze(2)
            std = x_flat.std(dim=1, keepdim=True).unsqueeze(2) + 1e-8
            x_normalized = (x - mean) / std
        
        # Apply sigmoid to get values in [0, 1] for quantization
        x_normalized = torch.sigmoid(x_normalized)
        return x_normalized
    
    def encode_sample_vectorized(self, x):
        """Vectorized encoding of MEG samples to hypervectors"""
        batch_size, num_channels, time_points = x.shape
        device = x.device
        
        # Normalize and quantize values
        normalized = self.normalize_meg_values(x)
        quantized = (normalized * (self.quantization_levels - 1)).long().clamp(
            0, self.quantization_levels - 1
        )
        
        # Get all embeddings
        channel_indices = torch.arange(num_channels, device=device)
        time_indices = torch.arange(time_points, device=device)
        
        channel_embeds = self.channel_hvs(channel_indices)  # (num_channels, d)
        time_embeds = self.time_hvs(time_indices)  # (time_points, d)
        
        # Initialize result
        batch_hv = torch.zeros(batch_size, self.d, device=device)
        
        # Process each sample in the batch
        for b in range(batch_size):
            # Get value embeddings for this sample
            flat_values = quantized[b].flatten()
            value_embeds = self.value_encoder(flat_values)
            value_embeds = value_embeds.view(num_channels, time_points, self.d)
            
            # Expand dimensions for broadcasting
            ch_exp = channel_embeds.unsqueeze(1)  # (num_channels, 1, d)
            t_exp = time_embeds.unsqueeze(0)  # (1, time_points, d)
            
            # Bind all three components (channel, time, value)
            bound = hd_func.bind(ch_exp, hd_func.bind(t_exp, value_embeds))
            
            # Bundle across all channels and time points
            bundled = bound.sum(dim=[0, 1])
            
            # Normalize the hypervector
            batch_hv[b] = F.normalize(bundled, p=2, dim=0)
        
        return batch_hv
    
    def forward(self, x):
        """Forward pass through the hyperdimensional model"""
        # Encode input to hypervector
        hv = self.encode_sample_vectorized(x)
        
        # Apply learned transformations
        hv = self.input_projection(hv)
        
        if self.use_prototypes:
            # Compute similarities to class prototypes
            similarities = F.cosine_similarity(
                hv.unsqueeze(1),  # (batch, 1, d)
                self.class_prototypes.unsqueeze(0),  # (1, num_classes, d)
                dim=2
            )
            # Scale similarities for better gradients
            logits = similarities * 10.0
        else:
            # Standard classification
            logits = self.class_projection(hv)
        
        return logits
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.training_step_result.append(loss)

        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def on_train_epoch_end(self):
        epoch_average = torch.stack(self.training_step_result).mean()
        self.log("training_epoch_average", epoch_average)
        self.training_step_result.clear()  # free memory
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.validation_step_outputs.append(loss)
        
        # Store predictions and targets for epoch-end metrics
        preds = torch.argmax(logits, dim=1)
        val_result = {
            'val_loss': loss,
            'preds': preds,
            'targets': y
        } 
        self.validation_step_result.append(val_result)

        return val_result  
    
    def on_validation_epoch_end(self):
        """Compute validation metrics at epoch end"""
        # Concatenate all predictions and targets
        outputs = self.validation_step_result
        all_preds = torch.cat([x['preds'] for x in outputs])
        all_targets = torch.cat([x['targets'] for x in outputs])
        
        # Convert to numpy for sklearn metrics
        preds_np = all_preds.cpu().numpy()
        targets_np = all_targets.cpu().numpy()
        
        # Calculate metrics
        acc = accuracy_score(targets_np, preds_np)
        f1_macro = f1_score(targets_np, preds_np, average='macro', zero_division=0)
        balanced_acc = balanced_accuracy_score(targets_np, targets_np)
        
        # Calculate average loss
        # avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_loss = torch.stack(self.validation_step_outputs).mean()

        # Log metrics
        self.log('val_loss', avg_loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_f1_macro', f1_macro, prog_bar=True)
        self.log('val_balanced_acc', balanced_acc, prog_bar=True)
        self.validation_step_outputs.clear()
    
    def test_step(self, batch, batch_idx):
        """Test step"""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.validation_step_outputs.append(loss)
        
        # Store predictions and targets for epoch-end metrics
        preds = torch.argmax(logits, dim=1)
        test_result = {
            'val_loss': loss,
            'preds': preds,
            'targets': y
        } 
        self.test_step_result.append(test_result)
        return test_result

    def on_test_epoch_end(self, outputs):
        """Compute test metrics at epoch end"""
        outputs = self.test_step_result
        all_preds = torch.cat([x['preds'] for x in outputs])
        all_targets = torch.cat([x['targets'] for x in outputs])
        
        preds_np = all_preds.cpu().numpy()
        targets_np = all_targets.cpu().numpy()
        
        # Calculate metrics
        acc = accuracy_score(targets_np, preds_np)
        f1_macro = f1_score(targets_np, preds_np, average='macro', zero_division=0)
        balanced_acc = balanced_accuracy_score(targets_np, preds_np)
        
        # avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_loss = torch.stack(self.test_step_outputs).mean()
        
        # Log metrics
        self.log('test_loss', avg_loss)
        self.log('test_acc', acc)
        self.log('test_f1_macro', f1_macro)
        self.log('test_balanced_acc', balanced_acc)
        
        print(f"\nTest Results:")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1-macro: {f1_macro:.4f}")
        print(f"Balanced Accuracy: {balanced_acc:.4f}")
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = CosineAnnealingLR(
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
