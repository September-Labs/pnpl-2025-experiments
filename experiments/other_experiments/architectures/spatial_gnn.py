"""
MEG Spatial Graph Neural Network for Phoneme Classification
Treats MEG sensors as nodes in a spatial graph to capture topographic patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np
from typing import Optional, Dict, List, Tuple
from torchmetrics import F1Score
import math


class GraphConvolution(nn.Module):
    """Graph Convolution Layer for processing spatial MEG patterns."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, input: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: (B, N, in_features) where N is number of nodes (MEG channels)
            adj: (N, N) adjacency matrix
        Returns:
            (B, N, out_features)
        """
        support = torch.matmul(input, self.weight)  # (B, N, out_features)
        
        # Apply adjacency matrix to each batch element
        # We need to expand adj to match batch dimension
        adj_expanded = adj.unsqueeze(0).expand(input.size(0), -1, -1)  # (B, N, N)
        output = torch.bmm(adj_expanded, support)  # (B, N, out_features)
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class SpatialAttention(nn.Module):
    """Spatial attention mechanism for MEG channels."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor, adj_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, N, hidden_dim) where N is number of MEG channels
            adj_mask: Optional (N, N) adjacency mask for local attention
        Returns:
            (B, N, hidden_dim)
        """
        B, N, _ = x.shape
        
        # Linear transformations and split into heads
        Q = self.q_linear(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, N, head_dim)
        K = self.k_linear(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, heads, N, N)
        
        # Apply adjacency mask if provided (for local attention)
        if adj_mask is not None:
            # Expand mask for batch and heads
            mask = adj_mask.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, -1, -1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # (B, heads, N, head_dim)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, self.hidden_dim)
        
        # Final linear layer
        output = self.out_linear(attn_output)
        
        return output


class MEGGraphBlock(nn.Module):
    """A single block combining graph convolution and spatial attention."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1,
                 use_graph_conv: bool = True, use_spatial_attention: bool = True):
        super().__init__()
        self.use_graph_conv = use_graph_conv
        self.use_spatial_attention = use_spatial_attention
        
        if use_graph_conv:
            self.graph_conv = GraphConvolution(hidden_dim, hidden_dim)
            self.norm1 = nn.LayerNorm(hidden_dim)
        
        if use_spatial_attention:
            self.spatial_attn = SpatialAttention(hidden_dim, num_heads)
            self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor, adj_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, N, hidden_dim)
            adj: (N, N) adjacency matrix
            adj_mask: Optional (N, N) mask for spatial attention
        Returns:
            (B, N, hidden_dim)
        """
        # Graph convolution
        if self.use_graph_conv:
            residual = x
            x = self.graph_conv(x, adj)
            x = self.norm1(self.dropout(x) + residual)
        
        # Spatial attention
        if self.use_spatial_attention:
            residual = x
            x = self.spatial_attn(x, adj_mask)
            x = self.norm2(self.dropout(x) + residual)
        
        # Feed-forward
        residual = x
        x = self.ffn(x)
        x = self.norm3(x + residual)
        
        return x


class MEGSpatialGNN(L.LightningModule):
    """Spatial Graph Neural Network for MEG Phoneme Classification."""
    
    def __init__(self,
                 meg_channels: int = 306,
                 time_points: int = 125,
                 vocab_size: int = 39,
                 hidden_dim: int = 256,
                 num_graph_blocks: int = 3,
                 num_heads: int = 8,
                 k_neighbors: int = 20,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01,
                 dropout: float = 0.1,
                 label_smoothing: float = 0.0,
                 use_graph_conv: bool = True,
                 use_spatial_attention: bool = True,
                 temporal_pooling: str = 'mean',  # 'mean', 'max', 'attention'
                 magnetometer_gradiometer_split: bool = True,
                 use_learnable_adjacency: bool = False,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        self.meg_channels = meg_channels
        self.time_points = time_points
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.magnetometer_gradiometer_split = magnetometer_gradiometer_split
        
        # Create adjacency matrix for MEG sensors
        self.adjacency_matrix = self._create_adjacency_matrix(k_neighbors)
        self.register_buffer('adj', self.adjacency_matrix)
        
        # Optional learnable adjacency
        if use_learnable_adjacency:
            self.adj_weight = nn.Parameter(torch.ones(meg_channels, meg_channels) * 0.1)
        else:
            self.adj_weight = None
        
        # Dual-stream processing for magnetometers and gradiometers
        if magnetometer_gradiometer_split:
            # Magnetometers: channels 0-101 (102 total)
            # Gradiometers: channels 102-305 (204 total)
            self.mag_encoder = nn.Linear(time_points, hidden_dim)
            self.grad_encoder = nn.Linear(time_points, hidden_dim)
            # Fusion layer to combine mag and grad features
            self.mag_projection = nn.Linear(hidden_dim, hidden_dim)
            self.grad_projection = nn.Linear(hidden_dim, hidden_dim)
        else:
            # Single stream for all channels
            self.channel_encoder = nn.Linear(time_points, hidden_dim)
        
        # Graph blocks
        self.graph_blocks = nn.ModuleList([
            MEGGraphBlock(
                hidden_dim, 
                num_heads, 
                dropout,
                use_graph_conv,
                use_spatial_attention
            )
            for _ in range(num_graph_blocks)
        ])
        
        # Temporal pooling
        self.temporal_pooling = temporal_pooling
        if temporal_pooling == 'attention':
            self.temporal_attention = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Softmax(dim=1)
            )
        
        # Channel pooling with learned importance
        self.channel_importance = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size)
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # Metrics
        self.train_f1 = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
        self.val_f1 = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
    
    def _create_adjacency_matrix(self, k_neighbors: int) -> torch.Tensor:
        """
        Create adjacency matrix based on approximate MEG sensor positions.
        Using a simple spatial arrangement assumption.
        """
        # Create a simple adjacency based on channel proximity
        # In reality, you'd want to use actual sensor positions from MNE
        n_channels = self.meg_channels
        adj = torch.zeros(n_channels, n_channels)
        
        # For simplicity, connect each channel to its k nearest neighbors
        # Assuming channels are roughly ordered by spatial location
        for i in range(n_channels):
            # Connect to nearby channels (simple proximity assumption)
            for offset in range(1, k_neighbors // 2 + 1):
                if i - offset >= 0:
                    adj[i, i - offset] = 1.0 / offset  # Weight by distance
                if i + offset < n_channels:
                    adj[i, i + offset] = 1.0 / offset
            
            # Self-connection
            adj[i, i] = 1.0
        
        # Normalize adjacency matrix (symmetric normalization)
        degree = adj.sum(dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.
        adj = degree_inv_sqrt.unsqueeze(1) * adj * degree_inv_sqrt.unsqueeze(0)
        
        return adj
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, meg_channels, time_points)
        Returns:
            (B, vocab_size) logits
        """
        B, C, T = x.shape
        
        # Get adjacency matrix (with optional learned component)
        if self.adj_weight is not None:
            adj = self.adj + F.softmax(self.adj_weight, dim=-1)
            # Renormalize
            degree = adj.sum(dim=1)
            degree_inv_sqrt = torch.pow(degree, -0.5)
            degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.
            adj = degree_inv_sqrt.unsqueeze(1) * adj * degree_inv_sqrt.unsqueeze(0)
        else:
            adj = self.adj
        
        # Encode channels: (B, C, T) -> (B, C, hidden_dim)
        if self.magnetometer_gradiometer_split:
            # Split magnetometers and gradiometers
            mag_data = x[:, :102, :]  # (B, 102, T)
            grad_data = x[:, 102:, :]  # (B, 204, T)
            
            # Encode separately with full hidden dimension
            mag_features = self.mag_encoder(mag_data)  # (B, 102, hidden_dim)
            grad_features = self.grad_encoder(grad_data)  # (B, 204, hidden_dim)
            
            # Project features
            mag_features = self.mag_projection(mag_features)  # (B, 102, hidden_dim)
            grad_features = self.grad_projection(grad_features)  # (B, 204, hidden_dim)
            
            # Concatenate along channel dimension
            node_features = torch.cat([mag_features, grad_features], dim=1)  # (B, 306, hidden_dim)
        else:
            # Single stream encoding
            node_features = self.channel_encoder(x)  # (B, C, hidden_dim)
        
        # Apply graph blocks
        for block in self.graph_blocks:
            node_features = block(node_features, adj)  # (B, C, hidden_dim)
        
        # Channel-wise importance weighting
        channel_weights = self.channel_importance(node_features)  # (B, C, 1)
        weighted_features = node_features * channel_weights  # (B, C, hidden_dim)
        
        # Pool across channels (spatial pooling)
        if self.temporal_pooling == 'mean':
            pooled = weighted_features.mean(dim=1)  # (B, hidden_dim)
        elif self.temporal_pooling == 'max':
            pooled, _ = weighted_features.max(dim=1)  # (B, hidden_dim)
        elif self.temporal_pooling == 'attention':
            attn_weights = self.temporal_attention(weighted_features)  # (B, C, 1)
            pooled = (weighted_features * attn_weights).sum(dim=1)  # (B, hidden_dim)
        else:
            pooled = weighted_features.mean(dim=1)
        
        # Classification
        logits = self.classifier(pooled)  # (B, vocab_size)
        
        return logits
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Metrics
        f1_macro = self.train_f1(y_hat, y)
        
        # Logging
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_f1_macro', f1_macro, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Metrics
        f1_macro = self.val_f1(y_hat, y)
        
        # Logging
        self.log('val_loss', loss)
        self.log('val_f1_macro', f1_macro, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        # Separate parameter groups
        params = [
            {'params': self.parameters(), 'lr': self.hparams.learning_rate}
        ]
        
        if self.adj_weight is not None:
            # Learnable adjacency gets lower learning rate
            params = [
                {'params': [p for n, p in self.named_parameters() if 'adj_weight' not in n], 
                 'lr': self.hparams.learning_rate},
                {'params': [self.adj_weight], 'lr': self.hparams.learning_rate * 0.1}
            ]
        
        optimizer = torch.optim.AdamW(params, weight_decay=self.hparams.weight_decay)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=50,
            eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_f1_macro'
            }
        }
    
    def on_train_epoch_end(self):
        # Reset metrics
        self.train_f1.reset()
    
    def on_validation_epoch_end(self):
        # Reset metrics
        self.val_f1.reset()