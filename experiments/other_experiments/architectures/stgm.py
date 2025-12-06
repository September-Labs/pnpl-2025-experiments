"""
Spatial-Temporal Graph Mamba (STGM) for MEG Phoneme Classification
A minimal, optimal architecture combining spatial graph structure with temporal Mamba processing
Includes spatial attention for learning phoneme-specific brain regions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np
from torchmetrics import F1Score
import math
import json
from pathlib import Path
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
from typing import List

# ============================================
# Spatial Graph Construction
# ============================================

class SpatialGraphBuilder:
    """Build spatial adjacency matrix from MEG sensor coordinates."""
    
    @staticmethod
    def load_coordinates(coord_path: str) -> np.ndarray:
        """Load MEG sensor coordinates from JSON file."""
        with open(coord_path, 'r') as f:
            coordinates = json.load(f)
        return np.array(coordinates)
    
    @staticmethod
    def build_adjacency(coordinates: np.ndarray, k_neighbors: int = 15, sigma: Optional[float] = None) -> torch.Tensor:
        """
        Build adjacency matrix using k-nearest neighbors with Gaussian weighting.
        
        Args:
            coordinates: (n_sensors, 3) array of sensor positions
            k_neighbors: Number of nearest neighbors to connect
            sigma: Gaussian kernel bandwidth (auto-computed if None)
        
        Returns:
            (n_sensors, n_sensors) adjacency matrix
        """
        n_sensors = coordinates.shape[0]
        
        # Compute pairwise distances
        distances = np.zeros((n_sensors, n_sensors))
        for i in range(n_sensors):
            for j in range(n_sensors):
                distances[i, j] = np.linalg.norm(coordinates[i] - coordinates[j])
        
        # Auto-compute sigma if not provided
        if sigma is None:
            # Use median distance to k-th neighbor as sigma
            kth_distances = np.sort(distances, axis=1)[:, k_neighbors]
            sigma = np.median(kth_distances)
        
        # Build adjacency matrix with k-NN and Gaussian weights
        adjacency = np.zeros((n_sensors, n_sensors))
        for i in range(n_sensors):
            # Get k nearest neighbors (excluding self)
            neighbor_indices = np.argsort(distances[i])[1:k_neighbors+1]
            
            # Apply Gaussian kernel
            for j in neighbor_indices:
                weight = np.exp(-distances[i, j]**2 / (2 * sigma**2))
                adjacency[i, j] = weight
                adjacency[j, i] = weight  # Symmetric
        
        # Add self-loops
        adjacency += np.eye(n_sensors)
        
        # Normalize (symmetric normalization)
        degree = np.sum(adjacency, axis=1)
        degree_inv_sqrt = np.power(degree, -0.5)
        degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0
        D_inv_sqrt = np.diag(degree_inv_sqrt)
        adjacency_norm = D_inv_sqrt @ adjacency @ D_inv_sqrt
        
        return torch.FloatTensor(adjacency_norm)
    
    @staticmethod
    def compute_brain_regions(coordinates: np.ndarray, n_regions: int = 8) -> torch.Tensor:
        """
        Cluster sensors into brain regions based on coordinates.
        Simple spatial binning if sklearn is not available.
        
        Args:
            coordinates: (n_sensors, 3) array of sensor positions
            n_regions: Number of brain regions to identify
        
        Returns:
            (n_sensors,) tensor of region assignments
        """
        try:
            from sklearn.cluster import KMeans
            # Use KMeans to identify spatial clusters
            kmeans = KMeans(n_clusters=n_regions, random_state=42)
            region_labels = kmeans.fit_predict(coordinates)
        except ImportError:
            # Fallback: simple spatial binning based on coordinates
            # Divide space into regions based on coordinate ranges
            n_sensors = coordinates.shape[0]
            region_labels = np.zeros(n_sensors, dtype=int)
            
            # Use x and y coordinates to create a simple grid
            x_bins = np.linspace(coordinates[:, 0].min(), coordinates[:, 0].max(), int(np.sqrt(n_regions)) + 1)
            y_bins = np.linspace(coordinates[:, 1].min(), coordinates[:, 1].max(), int(np.sqrt(n_regions)) + 1)
            
            for i in range(n_sensors):
                x_idx = np.digitize(coordinates[i, 0], x_bins) - 1
                y_idx = np.digitize(coordinates[i, 1], y_bins) - 1
                region_labels[i] = min(x_idx * len(y_bins) + y_idx, n_regions - 1)
        
        return torch.LongTensor(region_labels)

# ============================================
# Spatial Attention Module
# ============================================

class SpatialPhonemeAttention(nn.Module):
    """
    Learn phoneme-specific spatial attention patterns.
    Especially important for rare phonemes that might have distinct spatial signatures.
    """
    
    def __init__(self, n_sensors: int, vocab_size: int, hidden_dim: int, 
                 coordinates: np.ndarray, n_regions: int = 8):
        super().__init__()
        self.n_sensors = n_sensors
        self.vocab_size = vocab_size
        self.n_regions = n_regions
        
        # Compute brain regions
        self.register_buffer('sensor_regions', SpatialGraphBuilder.compute_brain_regions(coordinates, n_regions))
        
        # Coordinate-based positional encoding
        self.register_buffer('coord_encoding', self._compute_positional_encoding(coordinates))
        
        # Phoneme-specific spatial attention
        self.phoneme_spatial_query = nn.Embedding(vocab_size, hidden_dim)
        self.spatial_key = nn.Linear(3 + hidden_dim, hidden_dim)  # 3 for coordinates
        self.spatial_value = nn.Linear(n_sensors, hidden_dim)
        
        # Region-level attention for rare phonemes
        self.region_phoneme_affinity = nn.Parameter(torch.randn(n_regions, vocab_size) * 0.01)
        
        # Rare phoneme detector (learns which phonemes are rare and need special attention)
        self.rare_phoneme_gate = nn.Sequential(
            nn.Embedding(vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, n_sensors)
        
    def _compute_positional_encoding(self, coordinates: np.ndarray, max_freq: int = 10) -> torch.Tensor:
        """
        Compute sinusoidal positional encoding based on 3D coordinates.
        """
        n_sensors = coordinates.shape[0]
        encoding_dim = 64  # Dimension of positional encoding
        
        encoding = np.zeros((n_sensors, encoding_dim))
        
        # Use sinusoidal encoding for each coordinate dimension
        for i in range(3):  # x, y, z
            for j in range(encoding_dim // 6):  # Divide encoding dimension by 6 (3 coords * 2 for sin/cos)
                freq = 2 ** (j / (encoding_dim // 6)) * np.pi
                encoding[:, j*6 + i*2] = np.sin(coordinates[:, i] * freq)
                encoding[:, j*6 + i*2 + 1] = np.cos(coordinates[:, i] * freq)
        
        return torch.FloatTensor(encoding)
    
    def forward(self, x: torch.Tensor, phoneme_logits: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply phoneme-specific spatial attention.
        
        Args:
            x: (batch, n_sensors, time_points) MEG data
            phoneme_logits: (batch, vocab_size) optional phoneme predictions for guided attention
        
        Returns:
            attended_x: (batch, n_sensors, time_points) spatially attended MEG data
            attention_weights: (batch, n_sensors) attention weights
        """
        batch, n_sensors, time_points = x.shape
        device = x.device
        
        # Move buffers to correct device if needed
        if self.coord_encoding.device != device:
            self.coord_encoding = self.coord_encoding.to(device)
            self.sensor_regions = self.sensor_regions.to(device)
        
        # If we have phoneme predictions, use them to guide attention
        if phoneme_logits is not None:
            # Get predicted phoneme probabilities
            phoneme_probs = F.softmax(phoneme_logits, dim=-1)  # (batch, vocab_size)
            
            # Compute weighted phoneme query
            phoneme_queries = self.phoneme_spatial_query.weight  # (vocab_size, hidden_dim)
            weighted_query = torch.matmul(phoneme_probs, phoneme_queries)  # (batch, hidden_dim)
            
            # Check if predicted phonemes are rare
            predicted_phonemes = torch.argmax(phoneme_logits, dim=-1)  # (batch,)
            rare_gates = self.rare_phoneme_gate(predicted_phonemes)  # (batch, 1)
        else:
            # Use a general query if no phoneme predictions available
            weighted_query = self.phoneme_spatial_query.weight.mean(dim=0).unsqueeze(0).expand(batch, -1)
            rare_gates = torch.zeros(batch, 1, device=device)
        
        # Compute spatial features
        spatial_features = x.mean(dim=2)  # Average over time: (batch, n_sensors)
        
        # Combine coordinates with spatial features
        coord_features = self.coord_encoding.unsqueeze(0).expand(batch, -1, -1)  # (batch, n_sensors, encoding_dim)
        
        # Truncate or pad coord_features to match hidden_dim
        hidden_dim = weighted_query.shape[1]
        if coord_features.shape[2] > hidden_dim:
            coord_features = coord_features[:, :, :hidden_dim]
        else:
            pad_size = hidden_dim - coord_features.shape[2]
            coord_features = F.pad(coord_features, (0, pad_size))
        
        # Compute attention scores
        keys = self.spatial_key(torch.cat([
            self.coord_encoding[:, :3].unsqueeze(0).expand(batch, -1, -1),
            coord_features
        ], dim=-1))  # (batch, n_sensors, hidden_dim)
        
        attention_scores = torch.matmul(weighted_query.unsqueeze(1), keys.transpose(1, 2))  # (batch, 1, n_sensors)
        attention_scores = attention_scores.squeeze(1) / math.sqrt(hidden_dim)  # (batch, n_sensors)
        
        # Add region-based attention for rare phonemes
        if phoneme_logits is not None:
            # Compute region-level attention boost
            region_scores = torch.matmul(phoneme_probs, self.region_phoneme_affinity.T)  # (batch, n_regions)
            
            # Map region scores to sensor scores
            sensor_region_scores = region_scores.gather(1, self.sensor_regions.unsqueeze(0).expand(batch, -1))  # (batch, n_sensors)
            
            # Apply rare phoneme gating
            attention_scores = attention_scores + rare_gates * sensor_region_scores
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch, n_sensors)
        
        # Apply attention to input
        attended_x = x * attention_weights.unsqueeze(-1)  # (batch, n_sensors, time_points)
        
        return attended_x, attention_weights

# ============================================
# Graph Convolution Layer
# ============================================

class SimpleGCN(nn.Module):
    """Simple Graph Convolution Network layer."""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_nodes, in_features)
            adjacency: (n_nodes, n_nodes)
        
        Returns:
            (batch, n_nodes, out_features)
        """
        # Apply linear transformation
        x = self.linear(x)  # (batch, n_nodes, out_features)
        
        # Graph convolution: x = A @ x
        x = torch.matmul(adjacency, x.transpose(1, 2)).transpose(1, 2)
        
        # Add bias
        x = x + self.bias
        
        return x
        
# ============================================
# Minimal Mamba Block
# ============================================

@dataclass
class MambaConfig:
    d_model: int = 256
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2

class MambaBlock(nn.Module):
    """Simplified Mamba block for temporal processing."""
    
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        d_inner = config.d_model * config.expand
        
        # Input projection
        self.in_proj = nn.Linear(config.d_model, d_inner * 2, bias=False)
        
        # Convolution
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, 
            kernel_size=config.d_conv,
            padding=config.d_conv - 1,
            groups=d_inner
        )
        
        # SSM parameters - Fixed: dt_proj should take 1 input, not d_state
        self.x_proj = nn.Linear(d_inner, config.d_state + config.d_state + 1, bias=False)
        self.dt_proj = nn.Linear(1, d_inner, bias=True)  # Changed from d_state to 1
        
        # Initialize dt bias specially
        dt_init_std = config.d_state ** -0.5
        nn.init.uniform_(self.dt_proj.bias, -dt_init_std, dt_init_std)
        
        # SSM state parameters
        A = torch.arange(1, config.d_state + 1).reshape(1, config.d_state).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(d_inner, config.d_model, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape
        
        # Input projection
        xz = self.in_proj(x)  # (batch, seq_len, 2 * d_inner)
        x, z = xz.chunk(2, dim=-1)  # Each: (batch, seq_len, d_inner)
        
        # Convolution
        x = x.transpose(1, 2)  # (batch, d_inner, seq_len)
        x = self.conv1d(x)[:, :, :seq_len]  # Ensure correct length
        x = x.transpose(1, 2)  # (batch, seq_len, d_inner)
        
        # Apply SiLU activation
        x = F.silu(x)
        
        # SSM computation
        y = self.ssm(x)
        
        # Gating
        y = y * F.silu(z)
        
        # Output projection
        output = self.out_proj(y)
        
        return output
    
    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        """State Space Model computation."""
        batch, seq_len, d_inner = x.shape
        d_state = self.config.d_state
        
        # Compute SSM parameters
        deltaBC = self.x_proj(x)  # (batch, seq_len, d_state + d_state + 1)
        delta, B, C = torch.split(deltaBC, [1, d_state, d_state], dim=-1)
        # delta: (batch, seq_len, 1), B: (batch, seq_len, d_state), C: (batch, seq_len, d_state)
        
        # Project delta from 1 dimension to d_inner dimensions
        delta = F.softplus(self.dt_proj(delta))  # (batch, seq_len, d_inner)
        
        # Discretize A
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        
        # SSM step (simplified parallel scan)
        y = torch.zeros_like(x)
        h = torch.zeros(batch, d_inner, d_state, device=x.device)
        
        for t in range(seq_len):
            # State update
            # delta[:, t, :]: (batch, d_inner)
            # A: (d_inner, d_state)
            # We need to broadcast properly
            delta_t = delta[:, t, :].unsqueeze(-1)  # (batch, d_inner, 1)
            A_exp = torch.exp(delta_t * A.unsqueeze(0))  # (batch, d_inner, d_state)
            
            h = h * A_exp + (x[:, t, :].unsqueeze(-1) * B[:, t, :].unsqueeze(1))
            
            # Output
            y[:, t, :] = torch.sum(h * C[:, t, :].unsqueeze(1), dim=-1) + self.D * x[:, t, :]
        
        return y

# ============================================
# Main STGM Model with Spatial Attention
# ============================================

# ============================================
# Graph Convolution Layer
# ============================================

class SimpleGCN(nn.Module):
    """Simple Graph Convolution Network layer."""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_nodes, in_features)
            adjacency: (n_nodes, n_nodes)
        
        Returns:
            (batch, n_nodes, out_features)
        """
        # Apply linear transformation
        x = self.linear(x)  # (batch, n_nodes, out_features)
        
        # Graph convolution: x = A @ x
        x = torch.matmul(adjacency, x.transpose(1, 2)).transpose(1, 2)
        
        # Add bias
        x = x + self.bias
        
        return x

# ============================================
# Main STGM Model with Spatial Attention
# ============================================

class STGM(L.LightningModule):
    """Spatial-Temporal Graph Mamba for MEG phoneme classification."""
    
    def __init__(self,
                 meg_channels: int = 306,
                 time_points: int = 125,
                 vocab_size: int = 39,
                 hidden_dim: int = 256,
                 coord_path: str = "./sensor_xyz.json",
                 k_neighbors: int = 15,
                 use_spatial_attention: bool = True,
                 n_brain_regions: int = 8,
                 mamba_d_state: int = 16,
                 mamba_d_conv: int = 4,
                 mamba_expand: int = 2,
                 dropout: float = 0.2,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 label_smoothing: float = 0.1,
                 use_scheduler: bool = True,
                 scheduler_patience: int = 5):
        super().__init__()
        self.save_hyperparameters()
        
        # Load MEG sensor coordinates
        self.coordinates = SpatialGraphBuilder.load_coordinates(coord_path)
        
        # Build adjacency matrix
        self.adjacency = SpatialGraphBuilder.build_adjacency(self.coordinates, k_neighbors)
        
        # Spatial attention module (optional)
        self.use_spatial_attention = use_spatial_attention
        if use_spatial_attention:
            self.spatial_attention = SpatialPhonemeAttention(
                n_sensors=meg_channels,
                vocab_size=vocab_size,
                hidden_dim=hidden_dim,
                coordinates=self.coordinates,
                n_regions=n_brain_regions
            )
        
        # Spatial processing: GCN layer
        self.spatial_norm = nn.LayerNorm(meg_channels)  # Normalize over channels
        
        # Projection to hidden dimension
        self.projection = nn.Linear(meg_channels, hidden_dim)
        
        # Temporal processing: Mamba block
        mamba_config = MambaConfig(
            d_model=hidden_dim,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand
        )
        self.mamba = MambaBlock(mamba_config)
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, vocab_size)
        
        # Loss and metrics
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.f1_macro = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
        
        # Track attention weights for analysis
        self.register_buffer('last_attention_weights', torch.zeros(1, meg_channels))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, time_points) MEG data
        
        Returns:
            (batch, vocab_size) logits
        """
        batch, channels, time_points = x.shape
        
        # Initial classification for spatial attention guidance (optional)
        if self.use_spatial_attention:
            # Quick initial prediction without attention
            x_pooled = x.mean(dim=2)  # (batch, channels)
            initial_logits = self.classifier(self.dropout(self.projection(x_pooled)))
            
            # Apply spatial attention
            x, attention_weights = self.spatial_attention(x, initial_logits)
            self.last_attention_weights = attention_weights.detach()
        
        # Move adjacency to correct device
        if self.adjacency.device != x.device:
            self.adjacency = self.adjacency.to(x.device)
        
        # Apply spatial graph convolution
        # Reshape to process spatial dimension: (batch*time, channels)
        x_reshaped = x.transpose(1, 2).reshape(batch * time_points, channels)  # (batch*time, channels)
        
        # Apply adjacency-based spatial mixing
        x_reshaped = torch.matmul(self.adjacency, x_reshaped.T).T  # (batch*time, channels)
        
        # Reshape back to (batch, time, channels)
        x = x_reshaped.reshape(batch, time_points, channels)
        x = F.relu(self.spatial_norm(x))
        
        # Project to hidden dimension
        x = self.projection(x)  # (batch, time, hidden)
        
        # Temporal processing with Mamba
        x = self.mamba(x)  # (batch, time, hidden)
        x = self.temporal_norm(x)
        
        # Global average pooling over time
        x = x.mean(dim=1)  # (batch, hidden)
        
        # Classification
        x = self.dropout(x)
        logits = self.classifier(x)  # (batch, vocab_size)
        
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Metrics
        f1_macro = self.f1_macro(y_hat, y)
        
        # Logging
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_f1_macro', f1_macro)
        
        # Log attention statistics if using spatial attention
        if self.use_spatial_attention and batch_idx % 100 == 0:
            attention_entropy = -torch.sum(
                self.last_attention_weights * torch.log(self.last_attention_weights + 1e-10), 
                dim=-1
            ).mean()
            self.log('attention_entropy', attention_entropy)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Metrics
        f1_macro = self.f1_macro(y_hat, y)
        
        # Logging
        self.log('val_loss', loss)
        self.log('val_f1_macro', f1_macro, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Metrics
        f1_macro = self.f1_macro(y_hat, y)
        
        # Logging
        self.log('test_loss', loss)
        self.log('test_f1_macro', f1_macro)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        if self.hparams.use_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=self.hparams.scheduler_patience
                # Removed verbose=True as it's not supported in all PyTorch versions
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_f1_macro'
                }
            }
        
        return optimizer

class FocalLoss(nn.Module):
    """Focal Loss for handling extreme class imbalance."""
    
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class RarePhonemeAugmentation(nn.Module):
    """Augmentation specifically for rare phonemes."""
    
    def __init__(self, rare_phoneme_ids: List[int], augment_prob: float = 0.5):
        super().__init__()
        self.rare_phoneme_ids = set(rare_phoneme_ids)
        self.augment_prob = augment_prob
        
    def forward(self, x, y):
        """
        Apply augmentation to rare phoneme samples.
        x: (batch, channels, time)
        y: (batch,) labels
        """
        batch_size = x.shape[0]
        device = x.device
        
        augmented_x = x.clone()
        
        for i in range(batch_size):
            if y[i].item() in self.rare_phoneme_ids and torch.rand(1).item() < self.augment_prob:
                # Time warping (safer implementation)
                if torch.rand(1).item() < 0.5:
                    warp_factor = 1.0 + (torch.rand(1).item() - 0.5) * 0.2  # 0.9 to 1.1
                    time_dim = augmented_x[i].shape[-1]
                    new_time_dim = int(time_dim * warp_factor)
                    augmented_x[i] = F.interpolate(
                        augmented_x[i].unsqueeze(0), 
                        size=time_dim, 
                        mode='linear',
                        align_corners=False
                    ).squeeze(0)
                
                # Channel dropout
                if torch.rand(1).item() < 0.3:
                    channel_mask = torch.bernoulli(torch.ones(x.shape[1]) * 0.9).to(device)
                    augmented_x[i] = augmented_x[i] * channel_mask.unsqueeze(1)
                
                # Gaussian noise
                noise_level = 0.01
                augmented_x[i] = augmented_x[i] + torch.randn_like(augmented_x[i]) * noise_level
                
        return augmented_x, y

class HierarchicalClassifier(nn.Module):
    """Two-stage classifier: first broad category, then specific phoneme."""
    
    def __init__(self, input_dim: int, num_groups: int, vocab_size: int, 
                 phoneme_to_group: Dict[int, int]):
        super().__init__()
        self.phoneme_to_group = phoneme_to_group
        self.num_groups = num_groups
        self.vocab_size = vocab_size
        
        # Group classifier
        self.group_classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_groups)
        )
        
        # Phoneme classifier (conditioned on group)
        self.phoneme_classifier = nn.Sequential(
            nn.Linear(input_dim + num_groups, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, vocab_size)
        )
        
    def forward(self, x, return_group=False):
        # Predict group
        group_logits = self.group_classifier(x)
        group_probs = F.softmax(group_logits, dim=-1)
        
        # Condition phoneme prediction on group
        x_with_group = torch.cat([x, group_probs], dim=-1)
        phoneme_logits = self.phoneme_classifier(x_with_group)
        
        if return_group:
            return phoneme_logits, group_logits
        return phoneme_logits

class STGMBalanced(STGM):
    """STGM with enhanced handling for imbalanced phoneme distribution."""
    
    def __init__(self,
                 # Base STGM parameters
                 meg_channels: int = 306,
                 time_points: int = 125,
                 vocab_size: int = 39,
                 hidden_dim: int = 256,
                 coord_path: str = "./sensor_xyz.json",
                 k_neighbors: int = 20,
                 use_spatial_attention: bool = True,
                 n_brain_regions: int = 12,
                 mamba_d_state: int = 24,
                 mamba_d_conv: int = 4,
                 mamba_expand: int = 2,
                 dropout: float = 0.3,
                 learning_rate: float = 0.0005,
                 weight_decay: float = 0.0001,
                 label_smoothing: float = 0.05,
                 use_scheduler: bool = True,
                 scheduler_patience: int = 10,
                 # Class balancing parameters
                 use_focal_loss: bool = True,
                 focal_gamma: float = 2.0,
                 focal_alpha: Optional[torch.Tensor] = None,
                 use_class_weights: bool = True,
                 weight_type: str = "effective_num",
                 weights_path: Optional[str] = None,
                 rare_threshold: int = 100,
                 rare_boost_factor: float = 2.0,
                 use_mixup: bool = True,
                 mixup_alpha: float = 0.2,
                 use_hierarchical: bool = True,
                 phoneme_groups: Optional[Dict] = None):
        
        # Initialize base STGM
        super().__init__(
            meg_channels=meg_channels,
            time_points=time_points,
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            coord_path=coord_path,
            k_neighbors=k_neighbors,
            use_spatial_attention=use_spatial_attention,
            n_brain_regions=n_brain_regions,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            label_smoothing=label_smoothing,
            use_scheduler=use_scheduler,
            scheduler_patience=scheduler_patience
        )
        
        # Save additional hyperparameters
        self.save_hyperparameters()
        
        # Load class weights and identify rare phonemes
        self.rare_phoneme_ids = []
        if use_class_weights and weights_path and Path(weights_path).exists():
            with open(weights_path, 'r') as f:
                weight_data = json.load(f)
            
            # Get weights for specified type
            weights = weight_data['weights'][weight_type]
            class_weights = torch.tensor([weights[str(i)] for i in range(vocab_size)])
            
            # Identify rare phonemes
            counts = weight_data['phoneme_counts']
            self.rare_phoneme_ids = [int(i) for i, count in counts.items() if int(count) < rare_threshold]
            
            # Boost weights for rare phonemes
            for idx in self.rare_phoneme_ids:
                class_weights[idx] *= rare_boost_factor
            
            # Normalize weights
            class_weights = class_weights / class_weights.mean()
            self.register_buffer('class_weights', class_weights)
            
            print(f"Identified {len(self.rare_phoneme_ids)} rare phonemes: {self.rare_phoneme_ids}")
        else:
            self.register_buffer('class_weights', torch.ones(vocab_size))
            print("No weight file found, using uniform weights")
        
        # Initialize augmentation for rare phonemes
        if len(self.rare_phoneme_ids) > 0:
            self.rare_augmentation = RarePhonemeAugmentation(
                self.rare_phoneme_ids, 
                augment_prob=0.5
            )
        else:
            self.rare_augmentation = None
        
        # Override loss function with class-balanced version
        if use_focal_loss:
            # Move weights to CUDA if available
            if torch.cuda.is_available():
                self.class_weights = self.class_weights.cuda()
            
            self.criterion = FocalLoss(
                gamma=focal_gamma,
                alpha=self.class_weights if use_class_weights else focal_alpha
            )
        else:
            # Move weights to CUDA if available for CrossEntropyLoss too
            if torch.cuda.is_available() and use_class_weights:
                self.class_weights = self.class_weights.cuda()
            
            self.criterion = nn.CrossEntropyLoss(
                weight=self.class_weights if use_class_weights else None,
                label_smoothing=label_smoothing
            )
        
        # Initialize hierarchical classifier
        self.use_hierarchical = use_hierarchical
        if use_hierarchical and phoneme_groups:
            # Create phoneme to group mapping
            phoneme_to_group = {}
            for group_idx, (group_name, phoneme_list) in enumerate(phoneme_groups.items()):
                for phoneme_id in phoneme_list:
                    phoneme_to_group[phoneme_id] = group_idx
            
            self.hierarchical_classifier = HierarchicalClassifier(
                input_dim=hidden_dim,
                num_groups=len(phoneme_groups),
                vocab_size=vocab_size,
                phoneme_to_group=phoneme_to_group
            )
            
            # Additional loss for group classification
            self.group_criterion = nn.CrossEntropyLoss()
        else:
            self.hierarchical_classifier = None
        
        # Mixup parameters
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        
        # Additional metrics for rare phonemes
        self.f1_rare = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification (for hierarchical classifier)."""
        batch, channels, time_points = x.shape
        
        # Apply spatial attention if enabled
        if self.use_spatial_attention:
            x_pooled = x.mean(dim=2)
            initial_logits = self.classifier(self.dropout(self.projection(x_pooled)))
            x, _ = self.spatial_attention(x, initial_logits)
        
        # Apply spatial graph convolution
        if self.adjacency.device != x.device:
            self.adjacency = self.adjacency.to(x.device)
        
        x_reshaped = x.transpose(1, 2).reshape(batch * time_points, channels)
        x_reshaped = torch.matmul(self.adjacency, x_reshaped.T).T
        x = x_reshaped.reshape(batch, time_points, channels)
        x = F.relu(self.spatial_norm(x))
        
        # Project and process with Mamba
        x = self.projection(x)
        x = self.mamba(x)
        x = self.temporal_norm(x)
        
        # Global pooling
        features = x.mean(dim=1)
        
        return features
    
    def forward(self, x: torch.Tensor, use_hierarchical: bool = False) -> torch.Tensor:
        """Forward pass with optional hierarchical classification."""
        if use_hierarchical and self.hierarchical_classifier is not None:
            features = self.get_features(x)
            return self.hierarchical_classifier(features)
        else:
            return super().forward(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # Apply rare phoneme augmentation
        if self.rare_augmentation is not None and self.training:
            x, y = self.rare_augmentation(x, y)
        
        # Apply mixup augmentation
        if self.use_mixup and self.training and torch.rand(1).item() < 0.5:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            batch_size = x.size(0)
            index = torch.randperm(batch_size).to(x.device)
            
            mixed_x = lam * x + (1 - lam) * x[index]
            y_a, y_b = y, y[index]
            
            # Forward pass with mixed input
            y_hat = self(mixed_x)
            
            # Mixed loss
            loss = lam * self.criterion(y_hat, y_a) + (1 - lam) * self.criterion(y_hat, y_b)
            
            # Use original labels for metrics
            y_hat_metrics = self(x)
            f1_macro = self.f1_macro(y_hat_metrics, y)
        else:
            y_hat = self(x)
            loss = self.criterion(y_hat, y)
            f1_macro = self.f1_macro(y_hat, y)
        
        # Additional loss for hierarchical classification
        if self.use_hierarchical and self.hierarchical_classifier is not None:
            features = self.get_features(x)
            phoneme_logits, group_logits = self.hierarchical_classifier(features, return_group=True)
            
            # Create group labels
            group_labels = torch.tensor([
                self.hierarchical_classifier.phoneme_to_group.get(y_i.item(), 0) 
                for y_i in y
            ]).to(y.device)
            
            group_loss = self.group_criterion(group_logits, group_labels)
            loss = loss + 0.3 * group_loss
            
            self.log('train_group_loss', group_loss)
        
        # Track performance on rare phonemes
        if len(self.rare_phoneme_ids) > 0:
            rare_mask = torch.zeros_like(y, dtype=torch.bool)
            for rare_id in self.rare_phoneme_ids:
                rare_mask |= (y == rare_id)
            
            if rare_mask.any():
                y_hat_probs = F.softmax(y_hat, dim=-1)
                rare_preds = y_hat_probs[rare_mask].argmax(dim=-1)
                rare_acc = (rare_preds == y[rare_mask]).float().mean()
                self.log('train_rare_acc', rare_acc, prog_bar=True)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_f1_macro', f1_macro)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        f1_macro = self.f1_macro(y_hat, y)
        
        # Track rare phoneme performance
        if len(self.rare_phoneme_ids) > 0:
            rare_mask = torch.zeros_like(y, dtype=torch.bool)
            for rare_id in self.rare_phoneme_ids:
                rare_mask |= (y == rare_id)
            
            if rare_mask.any():
                y_hat_probs = F.softmax(y_hat, dim=-1)
                rare_preds = y_hat_probs[rare_mask].argmax(dim=-1)
                rare_acc = (rare_preds == y[rare_mask]).float().mean()
                self.log('val_rare_acc', rare_acc)
        
        self.log('val_loss', loss)
        self.log('val_f1_macro', f1_macro, prog_bar=True)
        
        return loss