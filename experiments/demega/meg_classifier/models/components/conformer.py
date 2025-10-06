"""
MEG Conformer Layer with DeBERTa Attention

This module implements a Conformer layer specifically designed for MEG signal processing,
incorporating DeBERTa-style attention and pre-layer normalization.
"""

import torch
import torch.nn as nn
from .deberta_attention import DisentangledSelfAttention


def normalize(x: torch.Tensor, eps: float = 1e-6, mode: str = "replicate") -> torch.Tensor:
    """
    Robust normalization using median and interquartile range.

    Args:
        x: Input tensor to normalize
        eps: Small constant for numerical stability
        mode: Padding mode (not used in current implementation)

    Returns:
        Normalized tensor
    """
    dim = -1
    median = x.median(dim=dim, keepdim=True)[0]
    q75 = x.quantile(0.75, dim=dim, keepdim=True)
    q25 = x.quantile(0.25, dim=dim, keepdim=True)
    iqr = q75 - q25
    y = (x - median) / (iqr + eps)
    return y


class MEGConformerLayer(nn.Module):
    """
    Conformer layer adapted for MEG signal processing.

    Combines convolution, self-attention, and feed-forward modules with
    residual connections and layer normalization.

    Args:
        dim: Hidden dimension
        num_heads: Number of attention heads
        ff_dim: Feed-forward dimension (defaults to 2*dim)
        kernel_size: Kernel size for depthwise convolution
        dropout: Dropout rate
        norm_type: Type of normalization ('pre' or 'post')
    """

    def __init__(self, dim: int, num_heads: int = 1, ff_dim: int = None,
                 kernel_size: int = 3, dropout: float = 0.0,
                 norm_type: str = "pre"):
        super().__init__()
        ff_dim = ff_dim or 2 * dim
        self.norm_type = norm_type

        # Convolutional module
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, 1)
        )

        # Layer normalization
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ln3 = nn.LayerNorm(dim)

        # DeBERTa-style attention
        self.attention = DisentangledSelfAttention(
            hidden_size=dim,
            num_heads=num_heads,
            attention_dropout=dropout,
            hidden_dropout=dropout,
            relative_attention=True,
            position_buckets=32,
            max_relative_positions=128
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass through the Conformer layer.

        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]

        Returns:
            Output tensor with same shape as input
        """
        if self.norm_type == "pre":
            # Pre-layer normalization (more stable training)
            # Convolutional module
            res = x
            x_norm = self.ln1(x)
            x_conv = x_norm.transpose(1, 2)
            x_conv = self.conv(x_conv).transpose(1, 2)
            x = res + self.dropout(x_conv)

            # Attention module
            res = x
            x_norm = self.ln2(x)
            attn_out = self.attention(x_norm)[0]
            x = res + self.dropout(attn_out)

            # Feed-forward module
            res = x
            x_norm = self.ln3(x)
            ff_out = self.ffn(x_norm)
            x = res + ff_out

        else:
            # Post-layer normalization (original Conformer)
            # Convolutional module
            res = x
            x_conv = x.transpose(1, 2)
            x_conv = self.conv(x_conv).transpose(1, 2)
            x = self.ln1(x_conv + res)

            # Attention module
            res = x
            attn_out = self.attention(x)[0]
            x = self.ln2(self.dropout(attn_out) + res)

            # Feed-forward module
            res = x
            x = self.ffn(x)
            x = self.ln3(x + res)

        return x