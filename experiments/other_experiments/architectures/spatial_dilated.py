# spatial_dilated.py - OPTIMIZED VERSION
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import F1Score
import math
import numpy as np
import json
from pathlib import Path


class SpatialAttention(nn.Module):
    """
    Optimized spatial attention layer using vectorized Fourier basis computation.
    """
    
    def __init__(self, in_channels, out_channels, harmonics=32, dropout_radius=0.2, 
                 sensor_positions_3d=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.harmonics = harmonics
        self.dropout_radius = dropout_radius
        
        # Fourier coefficients - reshape for efficient computation
        self.fourier_coeffs = nn.Parameter(
            torch.randn(out_channels, harmonics * harmonics, 2) * 0.01
        )
        
        # Pre-compute harmonic indices for vectorized computation
        k_idx, l_idx = torch.meshgrid(
            torch.arange(harmonics), 
            torch.arange(harmonics), 
            indexing='ij'
        )
        self.register_buffer('k_idx', k_idx.flatten())
        self.register_buffer('l_idx', l_idx.flatten())
        
        # Initialize sensor positions
        if sensor_positions_3d is not None:
            positions_2d = self._project_3d_to_2d(sensor_positions_3d)
            self.sensor_positions = nn.Parameter(torch.tensor(positions_2d, dtype=torch.float32))
            print(f"Initialized spatial attention with actual sensor positions")
        else:
            self.sensor_positions = nn.Parameter(torch.rand(in_channels, 2))
            print(f"Initialized spatial attention with random positions")
        
        # Post-processing with 1x1 grouped convolution for efficiency
        self.post_conv = nn.Conv1d(out_channels, out_channels, 1, bias=False)
        
        # Pre-compute dropout mask during init if not training
        self.register_buffer('eval_dropout_mask', torch.ones(1, in_channels, 1))
    
    def _project_3d_to_2d(self, positions_3d):
        """Project 3D to 2D using PCA."""
        positions = np.array(positions_3d)
        positions_centered = positions - positions.mean(axis=0)
        
        # Use SVD for more stable computation
        U, S, Vt = np.linalg.svd(positions_centered, full_matrices=False)
        positions_2d = U[:, :2] @ np.diag(S[:2])
        
        # Normalize to [0.1, 0.9]
        min_vals = positions_2d.min(axis=0)
        max_vals = positions_2d.max(axis=0)
        positions_2d = (positions_2d - min_vals) / (max_vals - min_vals + 1e-8)
        positions_2d = positions_2d * 0.8 + 0.1
        
        return positions_2d
    
    def _compute_fourier_attention_vectorized(self, positions):
        """
        Fully vectorized Fourier basis computation - MUCH faster!
        """
        x_pos = positions[:, 0:1]  # (in_channels, 1)
        y_pos = positions[:, 1:2]  # (in_channels, 1)
        
        # Compute phase for all harmonics at once
        # k_idx, l_idx are (harmonics^2,)
        # x_pos, y_pos are (in_channels, 1)
        phase = 2 * math.pi * (
            self.k_idx.unsqueeze(0) * x_pos + 
            self.l_idx.unsqueeze(0) * y_pos
        )  # (in_channels, harmonics^2)
        
        # Compute cos and sin for all positions and harmonics
        cos_phase = torch.cos(phase)  # (in_channels, harmonics^2)
        sin_phase = torch.sin(phase)  # (in_channels, harmonics^2)
        
        # Compute weights using matrix multiplication
        # fourier_coeffs is (out_channels, harmonics^2, 2)
        real_coeffs = self.fourier_coeffs[:, :, 0]  # (out_channels, harmonics^2)
        imag_coeffs = self.fourier_coeffs[:, :, 1]  # (out_channels, harmonics^2)
        
        # Compute attention weights for all output channels at once
        weights = (
            real_coeffs @ cos_phase.T + 
            imag_coeffs @ sin_phase.T
        )  # (out_channels, in_channels)
        
        return weights
    
    def forward(self, x):
        """Optimized forward pass."""
        batch_size, _, time_points = x.shape
        
        # Use cached positions
        positions = torch.sigmoid(self.sensor_positions) * 0.8 + 0.1
        
        # Compute attention weights (vectorized)
        attention_weights = self._compute_fourier_attention_vectorized(positions)
        
        # Apply spatial dropout only during training
        if self.training and self.dropout_radius > 0:
            dropout_center = torch.rand(2, device=x.device) * 0.8 + 0.1
            distances = torch.norm(positions - dropout_center, dim=1)
            dropout_mask = (distances > self.dropout_radius).float()
            attention_weights = attention_weights * dropout_mask.unsqueeze(0)
        
        # Softmax normalization
        attention_weights = F.softmax(attention_weights, dim=-1)  # (out_channels, in_channels)
        
        # Efficient batched matrix multiplication
        # Reshape for batch processing
        attention_weights = attention_weights.unsqueeze(0).expand(batch_size, -1, -1)
        x = torch.bmm(attention_weights, x)  # (batch_size, out_channels, time_points)
        
        # Post-processing
        x = self.post_conv(x)
        
        return x


class ResidualDilatedBlock(nn.Module):
    """
    Optimized residual block with fused operations.
    """
    
    def __init__(self, channels, kernel_size, dilation1, dilation2, dropout_rate=0.1):
        super().__init__()
        
        # Use Conv1d with groups for efficiency
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, 
                               dilation=dilation1, padding=dilation1 * (kernel_size - 1) // 2,
                               groups=1)  # Could use groups=channels//8 for depthwise
        self.bn1 = nn.BatchNorm1d(channels)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               dilation=dilation2, padding=dilation2 * (kernel_size - 1) // 2,
                               groups=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.conv3 = nn.Conv1d(channels, channels * 2, kernel_size,
                               padding=(kernel_size - 1) // 2)
        
    def forward(self, x):
        # First block with fused operations
        residual = x
        x = self.dropout1(F.gelu(self.bn1(self.conv1(x))))
        x = x + residual
        
        # Second block
        residual = x
        x = self.dropout2(F.gelu(self.bn2(self.conv2(x))))
        x = x + residual
        
        # GLU
        x = F.glu(self.conv3(x), dim=1)
        
        return x


class SpatialDilatedNetwork(L.LightningModule):
    """
    Optimized MEG phoneme classification model.
    """
    
    def __init__(self, 
                 time_points=125,
                 n_channels=306,
                 n_classes=39,
                 learning_rate=0.0003,
                 weight_decay=0.0001,
                 label_smoothing=0.0,
                 # Architecture parameters
                 spatial_output_channels=306,  # Changed to 306 for no reduction
                 fourier_harmonics=16,  # Reduced from 32 for speed
                 spatial_dropout_radius=0.2,
                 sensor_coordinates_path=None,
                 hidden_channels=256,  # Reduced from 320
                 n_blocks=4,  # Reduced from 5
                 kernel_size=3,
                 dropout_rate=0.1,
                 use_batch_norm=True,
                 temporal_shift_ms=50):
        super().__init__()
        self.save_hyperparameters()
        
        # Load sensor coordinates
        sensor_positions_3d = None
        if sensor_coordinates_path and Path(sensor_coordinates_path).exists():
            with open(sensor_coordinates_path, 'r') as f:
                sensor_positions_3d = json.load(f)
        
        # Temporal shift
        self.temporal_shift = int(temporal_shift_ms * 250 / 1000)
        
        # Optimized spatial attention
        self.spatial_attention = SpatialAttention(
            n_channels, 
            spatial_output_channels,
            harmonics=fourier_harmonics,
            dropout_radius=spatial_dropout_radius,
            sensor_positions_3d=sensor_positions_3d
        )
        
        # Initial projection
        self.input_proj = nn.Conv1d(spatial_output_channels, hidden_channels, 1)
        
        # Residual blocks
        self.blocks = nn.ModuleList()
        for block_idx in range(n_blocks):
            dilation1 = 2 ** ((2 * block_idx) % 5)
            dilation2 = 2 ** ((2 * block_idx + 1) % 5)
            
            self.blocks.append(
                ResidualDilatedBlock(
                    hidden_channels,
                    kernel_size,
                    dilation1,
                    dilation2,
                    dropout_rate
                )
            )
        
        # Final layers
        self.final_conv1 = nn.Conv1d(hidden_channels, hidden_channels * 2, 1)
        self.final_conv2 = nn.Conv1d(hidden_channels * 2, hidden_channels, 1)
        
        # Classification head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_channels, n_classes)
        
        # Loss and metrics
        if label_smoothing > 0:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Use torch.compile for additional speedup (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            self.spatial_attention = torch.compile(self.spatial_attention)
            for block in self.blocks:
                block = torch.compile(block)
        
        self.f1_macro = F1Score(num_classes=n_classes, average='macro', task="multiclass")
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
    
    def forward(self, x):
        if self.temporal_shift > 0:
            x = F.pad(x, (self.temporal_shift, 0))[:, :, :-self.temporal_shift]
        
        x = self.spatial_attention(x)
        x = self.input_proj(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = F.gelu(self.final_conv1(x))
        x = self.final_conv2(x)
        
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        f1_macro = self.f1_macro(y_hat, y)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_f1_macro', f1_macro, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        f1_macro = self.f1_macro(y_hat, y)
        
        self.log('val_loss', loss)
        self.log('val_f1_macro', f1_macro, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        f1_macro = self.f1_macro(y_hat, y)
        
        self.log('test_loss', loss)
        self.log('test_f1_macro', f1_macro)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}
        }

