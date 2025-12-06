# models/architectures/gru_phoneme.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import F1Score
import math


class GaussianSmoothing1D(nn.Module):
    """1D Gaussian smoothing for temporal noise reduction."""
    
    def __init__(self, channels, kernel_size, sigma):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd kernel size
            
        # Create Gaussian kernel
        x = torch.arange(kernel_size).float() - kernel_size // 2
        kernel = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        
        # Reshape for depthwise convolution
        kernel = kernel.view(1, 1, kernel_size)
        kernel = kernel.repeat(channels, 1, 1)
        
        self.register_buffer('weight', kernel)
        self.groups = channels
        self.padding = kernel_size // 2
        
    def forward(self, x):
        # x shape: (batch, channels, time)
        return F.conv1d(x, self.weight, groups=self.groups, padding=self.padding)


class SessionAdapter(nn.Module):
    """Session-specific adaptation layers for handling recording variations."""
    
    def __init__(self, n_sessions, n_channels, init_identity=True):
        super().__init__()
        self.n_sessions = n_sessions
        self.n_channels = n_channels
        
        # Session-specific weights and biases
        self.session_weights = nn.Parameter(torch.randn(n_sessions, n_channels, n_channels))
        self.session_bias = nn.Parameter(torch.zeros(n_sessions, 1, n_channels))
        
        if init_identity:
            # Initialize weights near identity
            for i in range(n_sessions):
                self.session_weights.data[i] = torch.eye(n_channels) + torch.randn(n_channels, n_channels) * 0.01
    
    def forward(self, x, session_idx=None):
        """
        Args:
            x: (batch, channels, time)
            session_idx: (batch,) tensor of session indices, or None for session 0
        """
        if session_idx is None:
            session_idx = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Get session-specific transforms
        weights = self.session_weights[session_idx]  # (batch, channels, channels)
        bias = self.session_bias[session_idx]  # (batch, 1, channels)
        
        # Apply transformation
        x_permuted = x.permute(0, 2, 1)  # (batch, time, channels)
        x_transformed = torch.bmm(x_permuted, weights) + bias
        return x_transformed.permute(0, 2, 1)  # Back to (batch, channels, time)


class GRUPhonemeClassifier(L.LightningModule):
    """GRU-based phoneme classifier adapted from the neural decoder architecture."""
    
    def __init__(
        self,
        time_points=125,
        n_channels=306,
        n_classes=39,
        # GRU parameters
        hidden_dim=256,
        n_layers=2,
        bidirectional=True,
        dropout=0.2,
        # Session adaptation
        use_session_adaptation=False,
        n_sessions=50,  # Approximate number of sessions in training data
        # Preprocessing
        use_gaussian_smoothing=True,
        gaussian_kernel_size=5,
        gaussian_sigma=1.0,
        # Input transformation
        input_projection_dim=128,
        use_input_projection=True,
        input_nonlinearity='softsign',  # 'softsign', 'relu', 'tanh', 'none'
        # Temporal aggregation
        aggregation='attention',  # 'last', 'mean', 'max', 'attention'
        # Augmentation
        white_noise_std=0.0,
        constant_offset_std=0.0,
        # Output
        fc_hidden_dim=512,
        use_fc_hidden=True,
        # Training
        learning_rate=1e-3,
        weight_decay=1e-4,
        lr_schedule='cosine',  # 'none', 'step', 'cosine'
        lr_step_size=10,
        lr_gamma=0.5,
        label_smoothing=0.0,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Store dimensions
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.time_points = time_points
        
        # Preprocessing layers
        if use_gaussian_smoothing:
            self.gaussian_smoother = GaussianSmoothing1D(
                n_channels, gaussian_kernel_size, gaussian_sigma
            )
        else:
            self.gaussian_smoother = None
            
        # Session adaptation
        if use_session_adaptation:
            self.session_adapter = SessionAdapter(n_sessions, n_channels, init_identity=True)
        else:
            self.session_adapter = None
            
        # Input projection
        if use_input_projection:
            self.input_projection = nn.Conv1d(n_channels, input_projection_dim, 1)
            gru_input_dim = input_projection_dim
        else:
            self.input_projection = None
            gru_input_dim = n_channels
            
        # Input nonlinearity
        if input_nonlinearity == 'softsign':
            self.input_nonlinearity = nn.Softsign()
        elif input_nonlinearity == 'relu':
            self.input_nonlinearity = nn.ReLU()
        elif input_nonlinearity == 'tanh':
            self.input_nonlinearity = nn.Tanh()
        else:
            self.input_nonlinearity = None
            
        # GRU layers
        self.gru = nn.GRU(
            gru_input_dim,
            hidden_dim,
            n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Initialize GRU weights
        for name, param in self.gru.named_parameters():
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
                
        # Calculate GRU output dimension
        gru_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        # Attention mechanism for aggregation
        if aggregation == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(gru_output_dim, gru_output_dim // 2),
                nn.Tanh(),
                nn.Linear(gru_output_dim // 2, 1)
            )
        else:
            self.attention = None
            
        # Output layers
        if use_fc_hidden:
            self.fc = nn.Sequential(
                nn.Linear(gru_output_dim, fc_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_hidden_dim, n_classes)
            )
        else:
            self.fc = nn.Linear(gru_output_dim, n_classes)
            
        # Loss and metrics
        if label_smoothing > 0:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.f1_macro = F1Score(num_classes=n_classes, average='macro', task='multiclass')
        
        # Store hyperparameters for aggregation
        self.aggregation = aggregation
        self.white_noise_std = white_noise_std
        self.constant_offset_std = constant_offset_std
        
    def forward(self, x, session_idx=None):
        """
        Args:
            x: (batch, channels, time) tensor of MEG data
            session_idx: Optional (batch,) tensor of session indices
        Returns:
            logits: (batch, n_classes) tensor
        """
        # Apply Gaussian smoothing
        if self.gaussian_smoother is not None:
            x = self.gaussian_smoother(x)
            
        # Apply session-specific adaptation
        if self.session_adapter is not None:
            x = self.session_adapter(x, session_idx)
            
        # Apply input projection
        if self.input_projection is not None:
            x = self.input_projection(x)
            
        # Apply input nonlinearity
        if self.input_nonlinearity is not None:
            x = self.input_nonlinearity(x)
            
        # Reshape for GRU: (batch, channels, time) -> (batch, time, channels)
        x = x.transpose(1, 2)
        
        # Apply GRU
        gru_out, _ = self.gru(x)  # (batch, time, hidden_dim * num_directions)
        
        # Aggregate temporal output
        if self.aggregation == 'last':
            # Use last timestep
            aggregated = gru_out[:, -1, :]
        elif self.aggregation == 'mean':
            # Average over time
            aggregated = gru_out.mean(dim=1)
        elif self.aggregation == 'max':
            # Max pooling over time
            aggregated, _ = gru_out.max(dim=1)
        elif self.aggregation == 'attention':
            # Attention-weighted aggregation
            attn_weights = self.attention(gru_out)  # (batch, time, 1)
            attn_weights = F.softmax(attn_weights, dim=1)
            aggregated = (gru_out * attn_weights).sum(dim=1)  # (batch, hidden_dim * num_directions)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")
            
        # Final classification
        logits = self.fc(aggregated)
        
        return logits
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # Data augmentation
        if self.training and self.white_noise_std > 0:
            x = x + torch.randn_like(x) * self.white_noise_std
            
        if self.training and self.constant_offset_std > 0:
            offset = torch.randn(x.size(0), x.size(1), 1, device=x.device) * self.constant_offset_std
            x = x + offset
            
        # Forward pass
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        f1_macro = self.f1_macro(preds, y)
        
        # Logging
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_f1_macro', f1_macro, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        # Forward pass (no augmentation during validation)
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        f1_macro = self.f1_macro(preds, y)
        
        # Logging
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_f1_macro', f1_macro, prog_bar=True, sync_dist=True)
        
        return loss
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        
        # Forward pass
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        f1_macro = self.f1_macro(preds, y)
        
        # Logging
        self.log('test_loss', loss, sync_dist=True)
        self.log('test_f1_macro', f1_macro, sync_dist=True)
        
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        if self.hparams.lr_schedule == 'none':
            return optimizer
        elif self.hparams.lr_schedule == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.hparams.lr_step_size,
                gamma=self.hparams.lr_gamma
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch'
                }
            }
        elif self.hparams.lr_schedule == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs if hasattr(self, 'trainer') else 50,
                eta_min=1e-6
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch'
                }
            }