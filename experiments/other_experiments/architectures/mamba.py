"""
Mamba SSM Model for Phoneme Classification
Processing averaged MEG signals with selective state space models
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import F1Score, Accuracy
from collections import defaultdict
from typing import Optional, Union, Tuple
import math

# ============================================
# Mamba SSM Core Components
# ============================================

class MambaBlock(nn.Module):
    """
    Mamba SSM block with selective scan mechanism.
    Based on the Mamba architecture for efficient sequence modeling.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Union[int, str] = 'auto',
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = 'random',
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        use_fast_path: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == 'auto' else dt_rank
        self.use_fast_path = use_fast_path
        
        # Input projection
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        
        # Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        # Activation
        self.act = nn.SiLU()
        
        # SSM Parameters
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # Initialize dt projection
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        
        # Initialize dt bias
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        
        # S4D real initialization
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
    
    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (B, L, D)
        Returns:
            output: (B, L, D)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Input projection
        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)
        
        # Convolution
        x = x.transpose(1, 2)  # (B, D, L)
        x = self.conv1d(x)[:, :, :seq_len]
        x = x.transpose(1, 2)  # (B, L, D)
        
        # Activation
        x = self.act(x)
        
        # SSM
        y = self.ssm(x)
        
        # Gating
        z = self.act(z)
        output = y * z
        
        # Output projection
        output = self.out_proj(output)
        
        return output
    
    def ssm(self, x):
        """
        Selective scan mechanism.
        Args:
            x: (B, L, D)
        Returns:
            y: (B, L, D)
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute SSM parameters
        deltaBC = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        
        # Split into components
        delta, B, C = torch.split(
            deltaBC, 
            [self.dt_rank, self.d_state, self.d_state], 
            dim=-1
        )
        
        # Compute dt
        delta = F.softplus(self.dt_proj(delta))  # (B, L, D)
        
        # Get A
        A = -torch.exp(self.A_log)  # (D, N)
        
        # Discretize
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, L, D, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, D, N)
        
        # Selective scan
        states = []
        state = torch.zeros(
            batch_size, self.d_inner, self.d_state, 
            device=x.device, dtype=x.dtype
        )
        
        for i in range(seq_len):
            state = deltaA[:, i] * state + deltaB[:, i] * x[:, i, :, None]
            states.append(state)
        
        states = torch.stack(states, dim=1)  # (B, L, D, N)
        
        # Compute output
        y = torch.einsum('bldn,bln->bld', states, C)
        y = y + self.D * x
        
        return y

class MambaLayer(nn.Module):
    """
    Mamba layer with residual connection and normalization.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        norm_type: str = "pre",
    ):
        super().__init__()
        self.norm_type = norm_type
        
        self.norm = nn.LayerNorm(d_model)
        self.mamba = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x):
        """
        Args:
            x: (B, L, D)
        Returns:
            output: (B, L, D)
        """
        if self.norm_type == "pre":
            # Pre-norm
            residual = x
            x = self.norm(x)
            x = self.mamba(x)
            x = residual + self.dropout(x)
        else:
            # Post-norm
            residual = x
            x = self.mamba(x)
            x = self.dropout(x)
            x = self.norm(residual + x)
        
        return x

# ============================================
# Balanced Pre-training Module (Reused)
# ============================================

class BalancedPhonemePretrainer(nn.Module):
    """
    Pre-training module with temperature-based reweighting.
    Uses focal loss and exponential temperature scaling for rare phonemes.
    """
    
    def __init__(self, vocab_size: int = 39, hidden_dim: int = 16, temperature: float = 2.0):
        super().__init__()

        phoneme_counts = {
            0: 216, 1: 443, 2: 1584, 3: 231, 4: 114, 5: 360, 6: 268, 7: 93,
            8: 772, 9: 504, 10: 477, 11: 429, 12: 231, 13: 282, 14: 119, 15: 428,
            16: 1052, 17: 570, 18: 63, 19: 430, 20: 566, 21: 518, 22: 1128, 23: 154,
            24: 226, 25: 14, 26: 276, 27: 634, 28: 743, 29: 113, 30: 1143, 31: 110,
            32: 96, 33: 236, 34: 326, 35: 428, 36: 151, 37: 456, 38: 7
        }

        total_count = sum(phoneme_counts.values())
        
        # Temperature-based scaling (exponential) to prevent extreme weights
        self.class_weights = torch.zeros(vocab_size)
        for i, count in phoneme_counts.items():
            freq = count / total_count
            # Use temperature to control the strength of reweighting
            self.class_weights[i] = math.exp(-temperature * freq)
        
        # Normalize weights to reasonable range
        self.class_weights = self.class_weights / self.class_weights.mean()
        # Clip extreme values
        self.class_weights = torch.clamp(self.class_weights, min=0.5, max=5.0)
        
    def focal_loss(self, logits: torch.Tensor, targets: torch.Tensor, 
                   gamma: float = 1.0, alpha: torch.Tensor = None):
        """Focal loss to focus on hard-to-classify phonemes."""
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** gamma * ce_loss
        
        if alpha is not None:
            focal_loss = alpha[targets] * focal_loss
        
        return focal_loss.mean()

# ============================================
# Main Mamba MEG Model
# ============================================

class MambaMEGClassifier(L.LightningModule):
    """
    Mamba-based phoneme classification for MEG signals.
    """
    
    def __init__(self,
                 meg_channels: int = 306,
                 time_points: int = 125,
                 vocab_size: int = 39,
                 hidden_dim: int = 64,
                 num_mamba_layers: int = 4,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand_factor: int = 2,
                 learning_rate: float = 1e-4,
                 loss_type: str = "cross_entropy",
                 focal_gamma: float = 1.0,
                 dropout_rate: float = 0.0,
                 label_smoothing: float = 0.0,
                 weight_decay: float = 0.0,
                 classifier_lr_multiplier: float = 1.0,
                 warmup_epochs: int = 0,
                 total_epochs: int = 100,
                 temperature: float = 2.0,
                 norm_type: str = "pre",
                 metric_type: str = "f1_macro",
                 use_balanced_mixup: bool = True,
                 mixup_alpha: float = 0.4,
                 mixup_balance_ratio: float = 0.3):
        super().__init__()  
        self.save_hyperparameters()

        print(f'Number of Mamba layers: {num_mamba_layers}')
        
        # Store metric type
        self.metric_type = metric_type
        print(f"Model initialized with metric_type: {self.metric_type}")
        
        # Balanced pre-trainer with temperature scaling
        self.pretrainer = BalancedPhonemePretrainer(vocab_size, hidden_dim, temperature)
        
        # Input projection from MEG channels to hidden dimension
        self.input_projection = nn.Sequential(
            nn.Linear(meg_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Stack of Mamba layers
        self.mamba_layers = nn.ModuleList([
            MambaLayer(
                d_model=hidden_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand_factor,
                dropout=dropout_rate,
                norm_type=norm_type
            )
            for _ in range(num_mamba_layers)
        ])
        
        # Final layer norm
        self.feature_norm = nn.LayerNorm(hidden_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * time_points, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, vocab_size)
        )
        
        # Loss parameters
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        
        # Initialize metrics based on metric_type
        if metric_type == "balanced_acc":
            self.train_metric = Accuracy(num_classes=vocab_size, average='macro', task="multiclass")
            self.val_metric = Accuracy(num_classes=vocab_size, average='macro', task="multiclass")
            self.test_metric = Accuracy(num_classes=vocab_size, average='macro', task="multiclass")
            self.metric_name = "balanced_acc"
        else:  # f1_macro (default)
            self.train_metric = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
            self.val_metric = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
            self.test_metric = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
            self.metric_name = "f1_macro"
        
        # Per-phoneme tracking
        self.phoneme_f1_scores = defaultdict(float)
        self.phoneme_counts = defaultdict(int)

        self.use_balanced_mixup = use_balanced_mixup
        self.mixup_alpha = mixup_alpha
        self.mixup_balance_ratio = mixup_balance_ratio
    
    def balanced_mixup(self, x, y):
        """Mix underrepresented phonemes into the batch"""
        if not self.use_balanced_mixup or self.current_epoch < 5:
            return x, y
        
        batch_size = x.size(0)
        device = x.device
        
        # Clone tensors to avoid in-place operations
        x = x.clone()
        y = y.clone()
        
        # Use class weights to identify underrepresented phonemes
        weights = self.pretrainer.class_weights.to(device)
        
        # Phonemes with weight > median are underrepresented
        median_weight = weights.median()
        underrep_mask = weights > median_weight
        underrep_classes = torch.where(underrep_mask)[0]
        
        # Mix subset of samples
        num_to_mix = int(batch_size * self.mixup_balance_ratio)
        indices_to_mix = torch.randperm(batch_size, device=device)[:num_to_mix]
        
        for idx in indices_to_mix:
            # Find an underrepresented class in this batch
            underrep_in_batch = [c.item() for c in underrep_classes if (y == c).any()]
            
            if underrep_in_batch:
                # Pick random underrepresented class
                target_class = np.random.choice(underrep_in_batch)
                source_idx = torch.where(y == target_class)[0]
                
                if len(source_idx) > 0:
                    j = source_idx[torch.randint(len(source_idx), (1,))].item()
                    
                    # Mix coefficient (beta distribution)
                    lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                    
                    # Mix the MEG signals
                    x[idx] = lam * x[idx] + (1 - lam) * x[j]
                    
                    # Switch label if mix is strong
                    if lam < 0.5:
                        y[idx] = y[j]
        
        return x, y
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features using Mamba layers.
        Args:
            x: (B, C, T) - MEG signal
        Returns:
            features: (B, T, D) - extracted features
        """
        B, C, T = x.shape
        
        # Transpose to (B, T, C) for processing
        x = x.transpose(1, 2)
        
        # Project to hidden dimension
        x = self.input_projection(x)  # (B, T, hidden_dim)
        
        # Apply Mamba layers
        for mamba_layer in self.mamba_layers:
            x = mamba_layer(x)
        
        # Final normalization
        x = self.feature_norm(x)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification.
        Args:
            x: (B, C, T) - MEG signal
        Returns:
            logits: (B, vocab_size) - classification logits
        """
        B, C, T = x.shape
        
        # Extract features using Mamba
        features = self.extract_features(x)  # (B, T, hidden_dim)
        
        # Flatten features for classification
        features_flat = features.reshape(B, -1)  # (B, T * hidden_dim)
        
        # Classification
        logits = self.classifier(features_flat)
        
        return logits
    
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
                
        else:  # cross_entropy (default)
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
        
        # Apply balanced mixup if enabled
        if self.use_balanced_mixup:
            x, y = self.balanced_mixup(x, y)
        
        logits = self(x)
        loss = self.compute_loss(logits, y)
        
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = (preds == y).float().mean()
            metric_value = self.train_metric(logits, y)
            
            for i in range(len(y)):
                phoneme_id = y[i].item()
                self.phoneme_counts[phoneme_id] += 1
                if preds[i] == y[i]:
                    self.phoneme_f1_scores[phoneme_id] += 1
        
        self.log('train_loss', loss, prog_bar=True)
        self.log(f'train_{self.metric_name}', metric_value, prog_bar=True)
        self.log('train_acc', acc)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        logits = self(x)
        loss = self.compute_loss(logits, y)
        
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()
        metric_value = self.val_metric(logits, y)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log(f'val_{self.metric_name}', metric_value, prog_bar=True)
        self.log('val_acc', acc)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        
        logits = self(x)
        loss = self.compute_loss(logits, y)
        
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()
        metric_value = self.test_metric(logits, y)
        
        self.log('test_loss', loss)
        self.log(f'test_{self.metric_name}', metric_value)
        self.log('test_acc', acc)
        
        return loss
    
    def on_train_epoch_end(self):
        """Log per-phoneme performance statistics."""
        if self.current_epoch % 5 == 0:
            print(f"\n{'='*50}")
            print(f"Epoch {self.current_epoch} - Per-Phoneme Performance:")
            print(f"Optimizing for: {self.metric_name}")
            
            phoneme_accuracies = {}
            for phoneme_id in self.phoneme_counts:
                if self.phoneme_counts[phoneme_id] > 0:
                    accuracy = self.phoneme_f1_scores[phoneme_id] / self.phoneme_counts[phoneme_id]
                    phoneme_accuracies[phoneme_id] = accuracy
            
            sorted_phonemes = sorted(phoneme_accuracies.items(), key=lambda x: x[1])
            
            print("Worst performing phonemes:")
            for pid, acc in sorted_phonemes[:5]:
                print(f"  Phoneme {pid}: {acc:.3f} (count: {self.phoneme_counts[pid]})")
            
            print("Best performing phonemes:")
            for pid, acc in sorted_phonemes[-5:]:
                print(f"  Phoneme {pid}: {acc:.3f} (count: {self.phoneme_counts[pid]})")
            
            print(f"{'='*50}\n")
            
            self.phoneme_f1_scores = defaultdict(float)
            self.phoneme_counts = defaultdict(int)
    
    def configure_optimizers(self):
        """Configure optimizer with different learning rates."""
        params = []
        
        # Input projection parameters
        params.append({
            'params': self.input_projection.parameters(), 
            'lr': self.hparams.learning_rate
        })
        
        # Mamba layers parameters
        params.append({
            'params': self.mamba_layers.parameters(), 
            'lr': self.hparams.learning_rate
        })
        
        # Classifier parameters with potentially different learning rate
        params.append({
            'params': self.classifier.parameters(), 
            'lr': self.hparams.learning_rate * self.hparams.classifier_lr_multiplier
        })
        
        optimizer = torch.optim.AdamW(params, weight_decay=self.hparams.weight_decay)
        
        # Cosine annealing with warmup
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
