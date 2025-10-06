"""
DeBERTa-style Disentangled Self-Attention for MEG Signal Processing

This module implements the disentangled self-attention mechanism from DeBERTa,
adapted for processing MEG (Magnetoencephalography) signals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple, Iterable


def prepare_attention_mask(attention_mask):
    """Prepare attention mask for multi-head attention."""
    if attention_mask.dim() <= 2:
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
    elif attention_mask.dim() == 3:
        attention_mask = attention_mask.unsqueeze(1)
    return attention_mask


@torch.jit.script
def make_log_bucket_position(relative_pos: torch.Tensor, bucket_size: int, max_position: int):
    """Create log-scaled position buckets for relative position encoding."""
    sign = torch.sign(relative_pos)
    mid = bucket_size // 2
    abs_pos = torch.where(
        (relative_pos < mid) & (relative_pos > -mid),
        torch.tensor(mid - 1).type_as(relative_pos),
        torch.abs(relative_pos),
    )
    log_pos = (
        torch.ceil(
            torch.log(abs_pos / mid)
            / torch.log(torch.tensor((max_position - 1) / mid))
            * (mid - 1)
        ) + mid
    )
    bucket_pos = torch.where(abs_pos <= mid, relative_pos.type_as(log_pos), log_pos * sign)
    return bucket_pos


def build_relative_position(query_layer, key_layer, bucket_size: int = -1, max_position: int = -1):
    """Build relative position matrix for attention computation."""
    query_size = query_layer.size(-2)
    key_size = key_layer.size(-2)

    q_ids = torch.arange(query_size, dtype=torch.long, device=query_layer.device)
    k_ids = torch.arange(key_size, dtype=torch.long, device=key_layer.device)
    rel_pos_ids = q_ids[:, None] - k_ids[None, :]
    if bucket_size > 0 and max_position > 0:
        rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position)
    rel_pos_ids = rel_pos_ids.to(torch.long)
    rel_pos_ids = rel_pos_ids[:query_size, :]
    rel_pos_ids = rel_pos_ids.unsqueeze(0)
    return rel_pos_ids


@torch.jit.script
def scaled_size_sqrt(query_layer: torch.Tensor, scale_factor: int):
    """Calculate scaled square root for attention normalization."""
    return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)


@torch.jit.script
def build_rpos(
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    relative_pos: torch.Tensor,
    position_buckets: int,
    max_relative_positions: int,
):
    """Build relative positions for key-query size mismatch cases."""
    if key_layer.size(-2) != query_layer.size(-2):
        return build_relative_position(
            key_layer, key_layer, bucket_size=position_buckets, max_position=max_relative_positions
        )
    else:
        return relative_pos


class DisentangledSelfAttention(nn.Module):
    """
    DeBERTa-style disentangled self-attention mechanism.

    This attention mechanism disentangles content and position information,
    computing attention scores using both content-to-content and position-aware
    interactions.

    Args:
        hidden_size: Dimension of hidden states
        num_heads: Number of attention heads
        attention_dropout: Dropout rate for attention weights
        hidden_dropout: Dropout rate for hidden states
        attention_bias: Whether to use bias in attention projections
        pos_att_type: Types of position attention ('c2p' and/or 'p2c')
        relative_attention: Whether to use relative position encoding
        position_buckets: Number of position buckets for log-scaled positions
        max_relative_positions: Maximum relative position distance
        share_att_key: Whether to share attention keys for position encoding
        max_position_embeddings: Maximum sequence length
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        attention_bias: bool = True,
        pos_att_type: Iterable[str] = ("c2p", "p2c"),
        relative_attention: bool = True,
        position_buckets: int = -1,
        max_relative_positions: int = -1,
        share_att_key: bool = False,
        max_position_embeddings: Optional[int] = None,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.all_head_size = hidden_size

        self.query_proj = nn.Linear(hidden_size, self.all_head_size, bias=attention_bias)
        self.key_proj = nn.Linear(hidden_size, self.all_head_size, bias=attention_bias)
        self.value_proj = nn.Linear(hidden_size, self.all_head_size, bias=attention_bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=attention_bias)

        self.dropout_attn = nn.Dropout(attention_dropout) if attention_dropout > 0.0 else nn.Identity()
        self.dropout_out = nn.Dropout(hidden_dropout) if hidden_dropout > 0.0 else nn.Identity()
        self.pos_dropout = nn.Dropout(hidden_dropout) if hidden_dropout > 0.0 else nn.Identity()

        self.share_att_key = bool(share_att_key)
        self.pos_att_type = tuple(pos_att_type) if pos_att_type is not None else tuple()
        self.relative_attention = bool(relative_attention)

        self.position_buckets = int(position_buckets)
        self.max_relative_positions = int(max_relative_positions)
        if self.max_relative_positions < 1:
            self.max_relative_positions = int(max_position_embeddings or 512)

        self.pos_ebd_size = self.position_buckets if self.position_buckets > 0 else self.max_relative_positions

        if self.relative_attention:
            self.rel_embeddings = nn.Embedding(self.pos_ebd_size*2, hidden_size)
            self.norm_rel_ebd = nn.LayerNorm(hidden_size)

        if self.relative_attention and not self.share_att_key:
            if "c2p" in self.pos_att_type:
                self.pos_key_proj = nn.Linear(hidden_size, self.all_head_size, bias=True)
            if "p2c" in self.pos_att_type:
                self.pos_query_proj = nn.Linear(hidden_size, self.all_head_size, bias=True)

    def get_rel_embedding(self):
        """Get normalized relative position embeddings."""
        rel_embeddings = self.rel_embeddings.weight if self.relative_attention else None
        if rel_embeddings is not None and hasattr(self, "norm_rel_ebd"):
            rel_embeddings = self.norm_rel_ebd(rel_embeddings)
        return rel_embeddings

    def _shape_qkv(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape and permute for multi-head attention computation."""
        B, S, _ = x.shape
        return x.view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3).reshape(B * self.num_heads, S, self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        query_states: Optional[torch.Tensor] = None,
        relative_pos: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of disentangled self-attention.

        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask
            output_attentions: Whether to return attention weights
            query_states: Optional separate query states
            relative_pos: Optional pre-computed relative positions

        Returns:
            Tuple of (output_states,) or (output_states, attention_weights)
        """
        if query_states is None:
            query_states = hidden_states

        B, S, _ = hidden_states.shape

        Q = self.query_proj(query_states)
        K = self.key_proj(hidden_states)
        V = self.value_proj(hidden_states)

        q = self._shape_qkv(Q)
        k = self._shape_qkv(K)
        v = self._shape_qkv(V)

        scale_factor = 1
        if "c2p" in self.pos_att_type:
            scale_factor += 1
        if "p2c" in self.pos_att_type:
            scale_factor += 1
        scale = scaled_size_sqrt(q, scale_factor).to(dtype=q.dtype, device=q.device)
        attn_scores = torch.bmm(q, k.transpose(-1, -2)) / scale

        if self.relative_attention and (("c2p" in self.pos_att_type) or ("p2c" in self.pos_att_type)):
            if self.rel_embeddings is None:
                raise ValueError("rel_embeddings must be provided when relative_attention=True")
            rel_embeddings = self.get_rel_embedding()
            rel_att = self._disentangled_attention_bias(q, k, relative_pos, rel_embeddings)
            attn_scores = attn_scores + rel_att

        attn_scores = attn_scores.view(B, self.num_heads, S, S)

        if attention_mask is not None:
            attention_mask = prepare_attention_mask(attention_mask).to(attn_scores.dtype)
            attn_scores = attn_scores + attention_mask

        attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_probs = self.dropout_attn(attn_probs)

        attn_probs_flat = attn_probs.view(B * self.num_heads, S, S)
        ctx = torch.bmm(attn_probs_flat, v)
        ctx = ctx.view(B, self.num_heads, S, self.head_dim).permute(0, 2, 1, 3).contiguous().view(B, S, self.all_head_size)

        ctx = self.out_proj(ctx)
        ctx = self.dropout_out(ctx)

        if output_attentions:
            return ctx, attn_probs
        return (ctx,)

    def _disentangled_attention_bias(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        relative_pos: Optional[torch.Tensor],
        rel_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute disentangled attention bias for position-aware attention.

        This computes content-to-position (c2p) and position-to-content (p2c)
        attention scores based on relative positions.
        """
        if relative_pos is None:
            relative_pos = build_relative_position(
                query_layer, key_layer,
                bucket_size=self.position_buckets,
                max_position=self.max_relative_positions,
            )
        elif relative_pos.dim() == 2:
            relative_pos = relative_pos.unsqueeze(0)
        elif relative_pos.dim() == 3:
            if relative_pos.size(0) != 1:
                relative_pos = relative_pos[:1]
        else:
            raise ValueError(f"relative_pos must have dim 2 or 3; got {relative_pos.dim()}")

        S = query_layer.size(-2)
        att_span = self.pos_ebd_size
        device = query_layer.device
        dtype = query_layer.dtype

        rel_embeddings = rel_embeddings[: (att_span * 2), :].unsqueeze(0)

        if self.share_att_key:
            pos_query_layer = self._shape_qkv(self.query_proj(rel_embeddings))
            pos_key_layer = self._shape_qkv(self.key_proj(rel_embeddings))
        else:
            if "c2p" in self.pos_att_type:
                pos_key_layer = self._shape_qkv(self.pos_key_proj(rel_embeddings))
            if "p2c" in self.pos_att_type:
                pos_query_layer = self._shape_qkv(self.pos_query_proj(rel_embeddings))

        repeat_factor = query_layer.size(0) // self.num_heads
        if "c2p" in self.pos_att_type:
            pos_key_layer = pos_key_layer.repeat(repeat_factor, 1, 1)
        if "p2c" in self.pos_att_type:
            pos_query_layer = pos_query_layer.repeat(repeat_factor, 1, 1)

        score = 0.0

        if "c2p" in self.pos_att_type:
            scale = scaled_size_sqrt(pos_key_layer, 2 if ("p2c" in self.pos_att_type) else 1).to(dtype=query_layer.dtype, device=device)
            c2p = torch.bmm(query_layer, pos_key_layer.transpose(-1, -2))
            c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
            c2p = torch.gather(c2p, dim=-1, index=c2p_pos.expand(query_layer.size(0), S, S))
            score = score + (c2p / scale)

        if "p2c" in self.pos_att_type:
            scale = scaled_size_sqrt(pos_query_layer, 2 if ("c2p" in self.pos_att_type) else 1).to(dtype=query_layer.dtype, device=device)
            r_pos = build_rpos(query_layer, key_layer, relative_pos,
                             position_buckets=self.position_buckets,
                             max_relative_positions=self.max_relative_positions)
            p2c_pos = torch.clamp(-r_pos + att_span, 0, att_span * 2 - 1)
            p2c = torch.bmm(key_layer, pos_query_layer.transpose(-1, -2))
            p2c = torch.gather(p2c, dim=-1, index=p2c_pos.expand(key_layer.size(0), S, S)).transpose(-1, -2)
            score = score + (p2c / scale)

        return score