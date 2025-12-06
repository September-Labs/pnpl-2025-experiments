"""
Temporal Triangulation MEG Model for Phoneme Classification
Trains three classifiers for first, middle, and last thirds of each 0.5s window
to account for phoneme bleed-over from adjacent phonemes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import F1Score, Accuracy
from collections import defaultdict
from typing import Optional, Union, Tuple, Iterable
import math

# ============================================
# DeBERTa Attention Components (from demega.py)
# ============================================

def prepare_attention_mask(attention_mask):
    if attention_mask.dim() <= 2:
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
    elif attention_mask.dim() == 3:
        attention_mask = attention_mask.unsqueeze(1)
    return attention_mask

@torch.jit.script
def make_log_bucket_position(relative_pos: torch.Tensor, bucket_size: int, max_position: int):
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
    return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)

@torch.jit.script
def build_rpos(
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    relative_pos: torch.Tensor,
    position_buckets: int,
    max_relative_positions: int,
):
    if key_layer.size(-2) != query_layer.size(-2):
        return build_relative_position(
            key_layer, key_layer, bucket_size=position_buckets, max_position=max_relative_positions
        )
    else:
        return relative_pos

class DisentangledSelfAttention(nn.Module):
    """DeBERTa-style disentangled self-attention."""
    
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
        rel_embeddings = self.rel_embeddings.weight if self.relative_attention else None
        if rel_embeddings is not None and hasattr(self, "norm_rel_ebd"):
            rel_embeddings = self.norm_rel_ebd(rel_embeddings)
        return rel_embeddings
    
    def _shape_qkv(self, x: torch.Tensor) -> torch.Tensor:
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

# ============================================
# Balanced Pre-training Module
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
# MEG Conformer Layer with Pre-LayerNorm
# ============================================

class MEGConformerLayer(nn.Module):
    """
    Conformer layer with pre-layer normalization and DeBERTa attention.
    """
    
    def __init__(self, dim: int, num_heads: int = 1, ff_dim: int = None, 
                 kernel_size: int = 3, dropout: float = 0.0, 
                 norm_type: str = "pre"):  # "pre", "post", or "mixed"
        super().__init__()
        ff_dim = ff_dim or 2 * dim
        self.norm_type = norm_type
        
        # Depthwise separable convolution with residual
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim),
            nn.SiLU(),  # Using SiLU instead of ReLU
            nn.Conv1d(dim, dim, 1)
        )
        
        # Layer norms
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ln3 = nn.LayerNorm(dim)
        
        # DeBERTa attention instead of standard multihead
        self.attention = DisentangledSelfAttention(
            hidden_size=dim,
            num_heads=num_heads,
            attention_dropout=dropout,
            hidden_dropout=dropout,
            relative_attention=True,
            position_buckets=32,
            max_relative_positions=128
        )
        
        # Improved FFN with SiLU
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Pre-layer norm implementation
        if self.norm_type == "pre":
            # Convolution module with pre-norm
            res = x
            x_norm = self.ln1(x)
            x_conv = x_norm.transpose(1, 2)
            x_conv = self.conv(x_conv).transpose(1, 2)
            x = res + self.dropout(x_conv)
            
            # Self-attention module with pre-norm
            res = x
            x_norm = self.ln2(x)
            attn_out = self.attention(x_norm)[0]
            x = res + self.dropout(attn_out)
            
            # Feed-forward module with pre-norm
            res = x
            x_norm = self.ln3(x)
            ff_out = self.ffn(x_norm)
            x = res + ff_out
            
        else:  # post-norm (original)
            res = x
            x_conv = x.transpose(1, 2)
            x_conv = self.conv(x_conv).transpose(1, 2)
            x = self.ln1(x_conv + res)
            
            res = x
            attn_out = self.attention(x)[0]
            x = self.ln2(self.dropout(attn_out) + res)
            
            res = x
            x = self.ffn(x)
            x = self.ln3(x + res)
        
        return x

# ============================================
# Temporal Triangulation MEG Model
# ============================================

class TemporalTriangulationMEGClassifier(L.LightningModule):
    """
    Temporal triangulation phoneme classification with three separate classifiers
    for first third, middle third, and last third of each 0.5s window.
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
                 temporal_weight_strategy: str = "adaptive",  # "equal", "center_heavy", "adaptive"
                 use_swa: bool = False):
        super().__init__()  
        self.save_hyperparameters()

        # Store parameters
        self.metric_type = metric_type
        self.temporal_weight_strategy = temporal_weight_strategy
        self.use_swa = use_swa
        
        # Calculate temporal segments (handle non-divisible time points)
        self.time_points = time_points
        base_segment = time_points // 3
        remainder = time_points % 3
        
        # Distribute remainder to segments
        first_size = base_segment + (1 if remainder > 0 else 0)
        middle_size = base_segment + (1 if remainder > 1 else 0)
        last_size = base_segment
        
        self.first_third = slice(0, first_size)
        self.middle_third = slice(first_size, first_size + middle_size)
        self.last_third = slice(first_size + middle_size, time_points)
        
        # Store segment sizes for classifier input
        self.first_segment_size = first_size
        self.middle_segment_size = middle_size
        self.last_segment_size = last_size
        
        # Balanced pre-trainer with temperature scaling
        self.pretrainer = BalancedPhonemePretrainer(vocab_size, hidden_dim, temperature)
        
        # Shared MEG encoder
        if use_conformer:
            # Input projection with residual connection
            self.input_projection = nn.Sequential(
                nn.Conv1d(meg_channels, hidden_dim, kernel_size=5, padding=2),
                nn.SiLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
            )
            
            # Add skip connection for input
            self.input_skip = nn.Conv1d(meg_channels, hidden_dim, kernel_size=1)
            
            # Shared Conformer layers with DeBERTa attention
            self.shared_encoder = nn.ModuleList([
                MEGConformerLayer(hidden_dim, 4, hidden_dim*2, 
                                dropout=dropout_rate, norm_type=norm_type) 
                for _ in range(num_conformers // 2)  # Half for shared processing
            ])
            
            # Temporal-specific encoders for each third
            self.first_encoder = nn.ModuleList([
                MEGConformerLayer(hidden_dim, 4, hidden_dim*2, 
                                dropout=dropout_rate, norm_type=norm_type) 
                for _ in range(num_conformers // 2)
            ])
            
            self.middle_encoder = nn.ModuleList([
                MEGConformerLayer(hidden_dim, 4, hidden_dim*2, 
                                dropout=dropout_rate, norm_type=norm_type) 
                for _ in range(num_conformers // 2)
            ])
            
            self.last_encoder = nn.ModuleList([
                MEGConformerLayer(hidden_dim, 4, hidden_dim*2, 
                                dropout=dropout_rate, norm_type=norm_type) 
                for _ in range(num_conformers // 2)
            ])
            
            self.encoder_output_dim = hidden_dim
        else:
            self.input_projection = None
            self.input_skip = None
            # LSTM alternative
            self.shared_encoder = nn.LSTM(
                meg_channels, hidden_dim, num_conformers // 2,
                batch_first=True, bidirectional=True, dropout=dropout_rate
            )
            self.first_encoder = nn.LSTM(
                hidden_dim * 2, hidden_dim, num_conformers // 2,
                batch_first=True, bidirectional=True, dropout=dropout_rate
            )
            self.middle_encoder = nn.LSTM(
                hidden_dim * 2, hidden_dim, num_conformers // 2,
                batch_first=True, bidirectional=True, dropout=dropout_rate
            )
            self.last_encoder = nn.LSTM(
                hidden_dim * 2, hidden_dim, num_conformers // 2,
                batch_first=True, bidirectional=True, dropout=dropout_rate
            )
            self.encoder_output_dim = hidden_dim * 2
        
        self.use_conformer = use_conformer
        
        # Feature normalization layers
        self.feature_norm = nn.LayerNorm(self.encoder_output_dim)
        
        # Three separate classifiers for each temporal segment with correct input dimensions
        first_input_dim = self.encoder_output_dim * self.first_segment_size
        middle_input_dim = self.encoder_output_dim * self.middle_segment_size
        last_input_dim = self.encoder_output_dim * self.last_segment_size
        
        self.first_classifier = nn.Sequential(
            nn.Linear(first_input_dim, 128),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, vocab_size)
        )
        
        self.middle_classifier = nn.Sequential(
            nn.Linear(middle_input_dim, 128),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, vocab_size)
        )
        
        self.last_classifier = nn.Sequential(
            nn.Linear(last_input_dim, 128),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, vocab_size)
        )
        
        # Learnable weights for combining temporal predictions
        if temporal_weight_strategy == "adaptive":
            self.temporal_weights = nn.Parameter(torch.tensor([0.25, 0.5, 0.25]))
        elif temporal_weight_strategy == "center_heavy":
            self.register_buffer('temporal_weights', torch.tensor([0.2, 0.6, 0.2]))
        else:  # equal
            self.register_buffer('temporal_weights', torch.tensor([1/3, 1/3, 1/3]))
        
        # Loss parameters
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        
        # Initialize metrics
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
    
    def extract_shared_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract shared features using the shared encoder."""
        B, C, T = x.shape
        
        if self.use_conformer:
            # Apply initial convolution with residual
            features_main = self.input_projection(x)
            features_skip = self.input_skip(x)
            features = features_main + features_skip  # Residual connection
            
            features = features.transpose(1, 2)  # (B, T, hidden_dim)
            
            # Apply shared conformer layers
            for conformer in self.shared_encoder:
                features = conformer(features)
        else:
            x = x.transpose(1, 2)
            features, _ = self.shared_encoder(x)
        
        return features
    
    def extract_temporal_features(self, shared_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract temporal-specific features for each third."""
        # Split features into three segments
        first_features = shared_features[:, self.first_third, :]
        middle_features = shared_features[:, self.middle_third, :]
        last_features = shared_features[:, self.last_third, :]
        
        if self.use_conformer:
            # Process each segment with its specific encoder
            for conformer in self.first_encoder:
                first_features = conformer(first_features)
            
            for conformer in self.middle_encoder:
                middle_features = conformer(middle_features)
            
            for conformer in self.last_encoder:
                last_features = conformer(last_features)
            
            # Apply feature normalization
            first_features = self.feature_norm(first_features)
            middle_features = self.feature_norm(middle_features)
            last_features = self.feature_norm(last_features)
        else:
            first_features, _ = self.first_encoder(first_features)
            middle_features, _ = self.middle_encoder(middle_features)
            last_features, _ = self.last_encoder(last_features)
        
        return first_features, middle_features, last_features
    
    def forward(self, x: torch.Tensor, return_all: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass for classification.
        
        Args:
            x: Input tensor of shape (batch_size, meg_channels, time_points)
            return_all: If True, returns tuple of (combined, first, middle, last).
                       If False, returns only combined logits for inference.
        
        Returns:
            Combined logits or tuple of all logits depending on return_all flag.
        """
        B = x.shape[0]
        
        # Extract shared features
        shared_features = self.extract_shared_features(x)
        
        # Extract temporal-specific features
        first_features, middle_features, last_features = self.extract_temporal_features(shared_features)
        
        # Flatten features for classification
        first_flat = first_features.reshape(B, -1)
        middle_flat = middle_features.reshape(B, -1)
        last_flat = last_features.reshape(B, -1)
        
        # Get predictions from each classifier
        first_logits = self.first_classifier(first_flat)
        middle_logits = self.middle_classifier(middle_flat)
        last_logits = self.last_classifier(last_flat)
        
        # Combine predictions with learned/fixed weights
        if hasattr(self, 'temporal_weights'):
            weights = F.softmax(self.temporal_weights, dim=0) if self.temporal_weight_strategy == "adaptive" else self.temporal_weights
            combined_logits = (weights[0] * first_logits + 
                             weights[1] * middle_logits + 
                             weights[2] * last_logits)
        else:
            combined_logits = (first_logits + middle_logits + last_logits) / 3
        
        # For inference/generation, return only combined logits
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
        
        combined_logits, first_logits, middle_logits, last_logits = self(x, return_all=True)
        
        # Compute loss for each temporal segment and combined
        combined_loss = self.compute_loss(combined_logits, y)
        first_loss = self.compute_loss(first_logits, y)
        middle_loss = self.compute_loss(middle_logits, y)
        last_loss = self.compute_loss(last_logits, y)
        
        # Total loss with emphasis on combined prediction
        total_loss = 0.4 * combined_loss + 0.2 * first_loss + 0.2 * middle_loss + 0.2 * last_loss
        
        with torch.no_grad():
            preds = combined_logits.argmax(dim=-1)
            acc = (preds == y).float().mean()
            metric_value = self.train_metric(combined_logits, y)
            
            # Track per-phoneme performance
            for i in range(len(y)):
                phoneme_id = y[i].item()
                self.phoneme_counts[phoneme_id] += 1
                if preds[i] == y[i]:
                    self.phoneme_f1_scores[phoneme_id] += 1
        
        self.log('train_loss', total_loss, prog_bar=True)
        self.log(f'train_{self.metric_name}', metric_value, prog_bar=True)
        self.log('train_acc', acc)
        
        # Log individual losses
        self.log('train_combined_loss', combined_loss)
        self.log('train_first_loss', first_loss)
        self.log('train_middle_loss', middle_loss)
        self.log('train_last_loss', last_loss)
        
        # Log temporal weights if adaptive
        if self.temporal_weight_strategy == "adaptive":
            weights = F.softmax(self.temporal_weights, dim=0)
            self.log('weight_first', weights[0])
            self.log('weight_middle', weights[1])
            self.log('weight_last', weights[2])
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        # Get all logits for validation metrics
        result = self(x, return_all=True)
        if isinstance(result, tuple):
            combined_logits, first_logits, middle_logits, last_logits = result
        else:
            # Fallback if only combined is returned
            combined_logits = result
            first_logits = middle_logits = last_logits = combined_logits
        
        # Compute losses
        combined_loss = self.compute_loss(combined_logits, y)
        first_loss = self.compute_loss(first_logits, y)
        middle_loss = self.compute_loss(middle_logits, y)
        last_loss = self.compute_loss(last_logits, y)
        
        total_loss = 0.4 * combined_loss + 0.2 * first_loss + 0.2 * middle_loss + 0.2 * last_loss
        
        preds = combined_logits.argmax(dim=-1)
        acc = (preds == y).float().mean()
        metric_value = self.val_metric(combined_logits, y)
        
        # Also compute metrics for individual segments
        first_preds = first_logits.argmax(dim=-1)
        middle_preds = middle_logits.argmax(dim=-1)
        last_preds = last_logits.argmax(dim=-1)
        
        first_acc = (first_preds == y).float().mean()
        middle_acc = (middle_preds == y).float().mean()
        last_acc = (last_preds == y).float().mean()
        
        self.log('val_loss', total_loss, prog_bar=True)
        self.log(f'val_{self.metric_name}', metric_value, prog_bar=True)
        self.log('val_acc', acc)
        
        # Log individual segment performance
        self.log('val_first_acc', first_acc)
        self.log('val_middle_acc', middle_acc)
        self.log('val_last_acc', last_acc)
        
        return total_loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        
        # Get all logits for test metrics
        result = self(x, return_all=True)
        if isinstance(result, tuple):
            combined_logits, first_logits, middle_logits, last_logits = result
        else:
            # Fallback if only combined is returned
            combined_logits = result
            first_logits = middle_logits = last_logits = combined_logits
        
        combined_loss = self.compute_loss(combined_logits, y)
        first_loss = self.compute_loss(first_logits, y)
        middle_loss = self.compute_loss(middle_logits, y)
        last_loss = self.compute_loss(last_logits, y)
        
        total_loss = 0.4 * combined_loss + 0.2 * first_loss + 0.2 * middle_loss + 0.2 * last_loss
        
        preds = combined_logits.argmax(dim=-1)
        acc = (preds == y).float().mean()
        metric_value = self.test_metric(combined_logits, y)
        
        self.log('test_loss', total_loss)
        self.log(f'test_{self.metric_name}', metric_value)
        self.log('test_acc', acc)
        
        return total_loss
    
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
            
            if self.temporal_weight_strategy == "adaptive":
                weights = F.softmax(self.temporal_weights, dim=0)
                print(f"\nTemporal weights - First: {weights[0]:.3f}, Middle: {weights[1]:.3f}, Last: {weights[2]:.3f}")
            
            print(f"{'='*50}\n")
            
            self.phoneme_f1_scores = defaultdict(float)
            self.phoneme_counts = defaultdict(int)
    
    def configure_optimizers(self):
        """Configure optimizer with different learning rates."""
        params = []
        
        # Shared encoder parameters
        if self.use_conformer and self.input_projection is not None:
            params.append({
                'params': list(self.input_projection.parameters()) + 
                         list(self.input_skip.parameters()), 
                'lr': self.hparams.learning_rate
            })
        
        params.append({
            'params': self.shared_encoder.parameters(), 
            'lr': self.hparams.learning_rate
        })
        
        # Temporal-specific encoder parameters
        params.append({
            'params': list(self.first_encoder.parameters()) + 
                     list(self.middle_encoder.parameters()) + 
                     list(self.last_encoder.parameters()), 
            'lr': self.hparams.learning_rate
        })
        
        # Classifier parameters with potentially different learning rate
        params.append({
            'params': list(self.first_classifier.parameters()) + 
                     list(self.middle_classifier.parameters()) + 
                     list(self.last_classifier.parameters()), 
            'lr': self.hparams.learning_rate * self.hparams.classifier_lr_multiplier
        })
        
        # Temporal weights if adaptive
        if self.temporal_weight_strategy == "adaptive":
            params.append({
                'params': [self.temporal_weights],
                'lr': self.hparams.learning_rate * 0.1  # Slower learning for weights
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