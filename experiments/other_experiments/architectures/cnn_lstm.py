import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import F1Score
from torchmetrics.classification import MulticlassF1Score
from typing import Optional, Tuple, Union, Iterable
from transformers.activations import ACT2FN

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

class CNNLSTMPhonemeModel(L.LightningModule):
    """
    CNN-LSTM model for phoneme classification from MEG data.
    
    Architecture:
    - CNN layers for spatial feature extraction
    - LSTM for temporal modeling
    - Classification head
    """
    
    def __init__(self, 
                 cnn_channels=[512, 1024, 512],
                 lstm_hidden_size=256,
                 lstm_num_layers=2,
                 dropout=0.3,
                 learning_rate=0.0005,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        # CNN layers
        cnn_layers = []
        in_channels = 306
        
        for out_channels in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=1, stride=2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(2, stride=2),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Calculate CNN output size
        cnn_output_size = 125  # Initial time dimension
        for _ in cnn_channels:
            cnn_output_size = cnn_output_size // 2  # MaxPool1d reduces by factor of 2
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, 128),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 39)
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.f1_macro = F1Score(num_classes=39, average='macro', task="multiclass")
        
    def forward(self, x):
        # x shape: (batch, 306, 125)
        # CNN feature extraction
        cnn_out = self.cnn(x)  # (batch, channels, time)

        # Prepare for LSTM: (batch, time, channels)
        cnn_out = cnn_out.transpose(1, 2)

        # LSTM
        lstm_out, _ = self.lstm(cnn_out)  # (batch, time, hidden*2)

        # Classification for each time step
        time_logits = self.classifier(lstm_out)  # (batch, time, num_classes)

        selected_logits = torch.sum(time_logits, dim=1)
        return selected_logits
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        f1_macro = self.f1_macro(y_hat, y)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_f1_macro', f1_macro)
        
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=3,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_f1_macro'
            }
        }



class BalancedPhonemePretrainer(nn.Module):
    """
    Linear class weights with positive floor:
    lin_i ∈ [0,1]  →  w_i = w_min + (w_max - w_min) * lin_i
    Then normalize so mean(w)=1.
    """
    def __init__(self, vocab_size: int = 39, w_min: float = 0.1, w_max: float = 3.0, *_args, **_kwargs):
        super().__init__()

        phoneme_counts = {
            0: 216, 1: 443, 2: 1584, 3: 231, 4: 114, 5: 360, 6: 268, 7: 93,
            8: 772, 9: 504, 10: 477, 11: 429, 12: 231, 13: 282, 14: 119, 15: 428,
            16: 1052, 17: 570, 18: 63, 19: 430, 20: 566, 21: 518, 22: 1128, 23: 154,
            24: 226, 25: 14, 26: 276, 27: 634, 28: 743, 29: 113, 30: 1143, 31: 110,
            32: 96, 33: 236, 34: 326, 35: 428, 36: 151, 37: 456, 38: 7
        }

        total = float(sum(phoneme_counts.values()))
        freqs = torch.zeros(vocab_size, dtype=torch.float32)
        for i, c in phoneme_counts.items():
            freqs[i] = c / total

        f_max, f_min = freqs.max(), freqs.min()
        lin = (f_max - freqs) / (f_max - f_min + 1e-12)  # 0 for most common → 1 for rarest

        # Map to (w_min, w_max], so nothing is zero
        class_weights = w_min + (w_max - w_min) * lin

        # Keep loss scale stable
        class_weights = class_weights / class_weights.mean().clamp_min(1e-12)

        self.register_buffer("class_weights", class_weights)

class ResidualBlock1D(nn.Module):
    """
    2a/2b: Conv1D (k=3, s=1, same) -> Conv1D (k=1, s=1, same) with identity skip.
    No activation inside; the ELU comes after the block (layer 3).
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return x + out  # residual add



class RotaryEmbedding(nn.Module):
    """
    Standard RoPE (rotary) embedding that returns (cos, sin) tensors for a batch of position_ids.

    Args:
        dim: head_dim of each attention head (must be even).
        max_position_embeddings: maximum seq length the cache will cover.
        base: theta; larger values stretch frequencies (e.g., 10_000.0 like GPT-NeoX).
        attention_scaling: optional multiplicative scaling applied to cos/sin (default 1.0).
        device: optional device to initialize buffers on.

    Forward:
        (x_like, position_ids) -> (cos, sin) with shapes [bs, seq, dim]
    """
    inv_freq: torch.Tensor  # for register_buffer typing

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int,
        base: float = 10_000.0,
        attention_scaling: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RotaryEmbedding requires even head_dim; got {dim}")

        self.dim = dim
        self.max_seq_len_cached = max_position_embeddings
        self.original_max_seq_len = max_position_embeddings
        self.base = float(base)
        self.attention_scaling = float(attention_scaling)

        # inv_freq has shape [dim/2]
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq 

    @torch.no_grad()
    def forward(self, x_like: torch.Tensor, position_ids: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_like: any tensor on the target device (e.g., qkv or hidden_states) to infer dtype/device
            position_ids: LongTensor [bs, seq] with absolute positions per token
        Returns:
            cos, sin: [bs, seq, dim] in x_like.dtype
        """
        # [bs, 1, dim/2]
        inv = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        # [bs, seq, 1]
        pos = position_ids[:, None, :].float()

        # force float32 math for stability; avoid MPS autocast weirdness
        device_type = x_like.device.type if (isinstance(x_like.device.type, str) and x_like.device.type != "mps") else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # [bs, seq, dim/2]
            freqs = (inv.float() @ pos.float()).transpose(1, 2) 
            # duplicate across last dim to match head_dim
            emb = torch.cat([freqs, freqs], dim=-1)  # [bs, seq, dim]

            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x_like.dtype, device=x_like.device), sin.to(dtype=x_like.dtype, device=x_like.device)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate last-dim halves: (x1, x2) -> (-x2, x1)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE to q, k.

    Shapes:
        q, k: [bs, nheads, seq, head_dim]  (or [bs, seq, nheads, head_dim] if you change unsqueeze_dim to 2)
        cos, sin: [bs, seq, head_dim]
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0, eps: float = 1e-7):
    """
    Multi-class Dice loss for single-label classification.
    Computes Dice on softmax probabilities vs one-hot targets.
    logits: [N, C]
    targets: [N] long
    """
    N, C = logits.shape
    probs = F.softmax(logits, dim=1)                          # [N, C]
    one_hot = F.one_hot(targets, num_classes=C).float()       # [N, C]

    # per-class Dice across the batch (macro-like)
    intersection = torch.sum(probs * one_hot, dim=0)          # [C]
    sums = torch.sum(probs, dim=0) + torch.sum(one_hot, dim=0)  # [C]
    dice = (2 * intersection + smooth) / (sums + smooth + eps)   # [C]
    loss = 1.0 - dice                                          # [C]
    return loss.mean()                                         # scalar


def supervised_nt_xent(emb: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1, eps: float = 1e-8):
    """
    Supervised contrastive loss (NT-Xent variant) on a single view.
    For each anchor i, positives are samples with the same label (excluding i).
    emb: [N, D] L2-normalized embeddings
    labels: [N] long
    returns scalar loss averaged over anchors that have at least one positive.
    """
    N, D = emb.shape
    if N <= 1:
        return emb.new_zeros(())  # no contrast available

    # cosine similarity matrix
    sim = torch.matmul(emb, emb.T)                             # [N, N], in [-1, 1]
    sim = sim / temperature

    # mask out self-comparisons
    logits_mask = torch.ones_like(sim, dtype=torch.bool)
    logits_mask.fill_(True)
    logits_mask.fill_diagonal_(False)

    # positives mask (same label, not self)
    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.T) & logits_mask              # [N, N] bool

    # for numerical stability, subtract row-wise max
    sim_max, _ = torch.max(sim.masked_fill(~logits_mask, float("-inf")), dim=1, keepdim=True)
    sim = sim - sim_max.detach()

    # denominator: sum over all j != i
    exp_sim = torch.exp(sim) * logits_mask                     # [N, N]
    denom = exp_sim.sum(dim=1) + eps                           # [N]

    # numerator: sum over positives j
    pos_exp_sim = exp_sim * pos_mask
    num = pos_exp_sim.sum(dim=1)                               # [N]

    # anchors with at least one positive
    valid = pos_mask.any(dim=1)
    if valid.sum() == 0:
        return emb.new_zeros(())

    loss_i = -torch.log((num[valid] + eps) / denom[valid])     # [N_valid]
    return loss_i.mean()


def focal_loss_mean_over_present_classes(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: torch.Tensor | float | None = None,
    eps: float = 1e-8,
    *,
    # --- entropy regularization params ---
    entropy_reg_weight: float = 0.1,   # weight for batch-avg KL(avg_probs || uniform)
    entropy_bonus: float = 0.1,        # reward per-sample entropy (subtracted from loss)
    vocab_size: int | None = None,     # if None, defaults to C = logits.size(-1)
) -> torch.Tensor:
    """
    Multi-class single-label focal loss with 'mean-over-classes-present' reduction
    + optional entropy regularization.

    Steps:
      1) Per-sample focal CE:  -(alpha_t) * (1 - p_t)^gamma * log(p_t)
      2) Average within each unique target class in the batch
      3) Mean across these class means
      4) + entropy regularization terms (optional):
         - entropy_reg_weight * KL(avg_probs || uniform)
         - minus entropy_bonus * mean_entropy(probs)

    Args:
        logits: [N, C] unnormalized scores
        targets: [N] long, class indices in [0..C-1]
        gamma: focusing parameter
        alpha: None, scalar, or [C]-tensor of class weights inside focal term
        eps: numerical stability for logs/ratios
        entropy_reg_weight: strength of distribution matching to uniform
        entropy_bonus: encourages higher per-sample entropy (use small values)
        vocab_size: cardinality for the target uniform; defaults to C

    Returns:
        Scalar tensor.
    """
    if logits.ndim != 2:
        raise ValueError("logits must be [N, C]")
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError("targets must be [N] and match batch size of logits")

    N, C = logits.shape
    V = int(vocab_size) if (vocab_size is not None) else C

    # ----- focal base (unchanged) -----
    log_probs = F.log_softmax(logits, dim=1)                     # [N, C]
    log_pt = log_probs.gather(1, targets.view(-1, 1)).squeeze(1) # [N]
    pt = log_pt.exp().clamp(min=eps, max=1 - eps)                # [N]

    if alpha is None:
        alpha_t = torch.ones_like(pt)
    elif isinstance(alpha, float):
        alpha_t = torch.full_like(pt, float(alpha))
    else:
        # alpha is a [C] tensor
        alpha_t = alpha.to(logits.device, dtype=log_probs.dtype).gather(0, targets)

    loss_i = -alpha_t * ((1.0 - pt) ** gamma) * log_pt           # [N]

    # mean over present classes
    uniq, inv = targets.unique(sorted=False, return_inverse=True)  # uniq: [K]
    K = uniq.numel()
    sums = torch.zeros(K, device=logits.device, dtype=loss_i.dtype)
    cnts = torch.zeros(K, device=logits.device, dtype=loss_i.dtype)
    sums.index_add_(0, inv, loss_i)
    cnts.index_add_(0, inv, torch.ones_like(loss_i))
    class_means = sums / cnts.clamp_min(1)
    base_loss = class_means.mean()

    if (entropy_reg_weight != 0.0) or (entropy_bonus != 0.0):
        probs = F.softmax(logits, dim=-1).clamp_min(eps)         # [N, C]
        top_p, top_c = probs.max(dim=1)                          # [N], [N] winners

        # For each class k, average the winning probabilities among items where k is argmax
        win_sums = torch.zeros(C, device=logits.device, dtype=probs.dtype)
        win_cnts = torch.zeros(C, device=logits.device, dtype=probs.dtype)
        win_sums.index_add_(0, top_c, top_p)                     # sum of winners' max-probs per class
        win_cnts.index_add_(0, top_c, torch.ones_like(top_p))    # count of winners per class

        # mean of winning probs per class; 0 for classes with no winners
        mean_win_p = win_sums / win_cnts.clamp_min(1.0)          # [C]

        # turn into a proper distribution (if degenerate, fall back to uniform)
        total = mean_win_p.sum()
        if total <= eps:
            pred_class_dist = torch.full((C,), 1.0 / float(C), device=logits.device, dtype=probs.dtype)
        else:
            pred_class_dist = (mean_win_p / total).clamp_min(eps)  # [C], sums to 1

        # uniform over classes (use C to match vector length)
        uniform = torch.full_like(pred_class_dist, 1.0 / float(C)).clamp_min(eps)

        # KL(pred_class_dist || uniform) = sum_i p_i * (log p_i - log u_i)
        kl_div = (pred_class_dist * (pred_class_dist.log() - uniform.log())).sum()

        # (optional) per-sample entropy reward
        entropy = -(probs * probs.log()).sum(dim=-1).mean()

        loss = base_loss + entropy_reg_weight * kl_div - entropy_bonus * entropy
        return loss

    return base_loss

class DynTable10Net(L.LightningModule):
    """
    Implements "Table 10: Architecture Hyperparameters (Layer-by-Layer)".

    Expected default input: (N, 306, 125)
    Layers:
      [RoPE] apply before first conv
      1)  Conv1d: 306 -> 128, k=7, s=1, padding='same'
      2a) ResBlock conv: k=3, s=1, same
      2b) ResBlock conv: k=1, s=1, same
      3)  ELU
      4)  Conv1d: 128 -> 128, k=50, s=25, padding=0  (downsample)
      [RoPE] apply after downsample
      5)  ELU
      6)  Conv1d: 128 -> 128, k=7, s=1, same
      7)  ELU
      8)  Flatten
      9)  Linear: 512 -> 512
      10) ReLU
      11) Dropout
      12) Linear: 512 -> num_classes
    """
    def __init__(self, num_classes: int = 39, 
                 seq_len: int = 125, 
                 dropout_p: float = 0.5,
                 learning_rate: float = 5e-4,
                 temperature: float = 0.1,           # used for contrastive
                 dice_smooth: float = 1.0,           # smoothing for Dice
                 dice_weight: float = 1.0,           # weight for Dice loss
                 contrastive_weight: float = 0.3,    # weight for contrastive loss
                 proj_dim: int = 64,                 # projection head output size
                 rope_base: float = 10_000.0,
                 rope_scale: float = 1.0,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        hidden_dim = 500
        self.pretrainer = BalancedPhonemePretrainer(num_classes)

        # 1) Conv1D 306->128, k=7, same
        self.conv1 = nn.Conv1d(306, 128, kernel_size=7, stride=1, padding=3, bias=True)

        # 2a/2b) Residual block
        self.resblock = ResidualBlock1D(128)

        # 3) ELU
        self.elu1 = nn.ELU()

        # 4) Downsampling conv: k=50, s=25, no padding => L: 125 -> 4
        self.down = nn.Conv1d(128, 128, kernel_size=50, stride=25, padding=0, bias=True)

        # 5) ELU
        self.elu2 = nn.ELU()

        # 6) Conv1D k=7, same
        self.conv2 = nn.Conv1d(128, 128, kernel_size=7, stride=1, padding=3, bias=True)

        # 7) ELU
        self.elu3 = nn.ELU()

        # Infer post-convolution temporal length for the head
        with torch.no_grad():
            dummy = torch.zeros(1, 306, seq_len)
            feats = self._forward_features(dummy)
            L = feats.shape[-1]
        self.to_len4 = nn.Identity() if (128 * L == 512) else nn.AdaptiveAvgPool1d(4)

        # 8) Flatten then 9-12) MLP head
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * (L if isinstance(self.to_len4, nn.Identity) else 4), 512)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(512, num_classes)

        # small projection head for contrastive
        self.proj = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, proj_dim)
        )

        # metrics
        self.train_f1 = MulticlassF1Score(num_classes=num_classes, average='macro')
        self.val_f1   = MulticlassF1Score(num_classes=num_classes, average='macro')
        self.test_f1  = MulticlassF1Score(num_classes=num_classes, average='macro')

        self.learning_rate = learning_rate

    @staticmethod
    def _apply_rope_to_channels(x: torch.Tensor, rope: "RotaryEmbedding") -> torch.Tensor:
        # (kept as in your original; not used below)
        N, C, L = x.shape
        x_seq = x.transpose(1, 2).contiguous()  # [N, L, C]
        pos = torch.arange(L, device=x.device, dtype=torch.long).unsqueeze(0).expand(N, -1)
        cos, sin = rope(x_seq, pos)
        x_rot = (x_seq * cos) + (rotate_half(x_seq) * sin)
        return x_rot.transpose(1, 2).contiguous()

    def _forward_features(self, x):
        x = self.conv1(x)          # layer 1
        x = self.resblock(x)       # layers 2a & 2b
        x = self.elu1(x)           # layer 3
        x = self.down(x)           # layer 4 (downsample 125 -> 4)
        x = self.elu2(x)           # layer 5
        x = self.conv2(x)          # layer 6
        x = self.elu3(x)           # layer 7
        return x

    def encode(self, x):
        """
        Returns the penultimate 512-d feature used for classification (after ReLU+Dropout).
        """
        x = self._forward_features(x)
        x = self.to_len4(x)                # ensure Flatten -> 512 features
        x = self.flatten(x)                # layer 8
        x = self.fc1(x)                    # layer 9
        x = self.relu(x)                   # layer 10
        x = self.drop(x)                   # layer 11
        return x                           # [N, 512]

    def forward(self, x):
        feat = self.encode(x)
        logits = self.fc2(feat)            # layer 12
        return logits

    # -------- losses --------

    def compute_losses(self, logits, features, targets):
        # Dice classification loss
        # dice = dice_loss_from_logits(logits, targets, smooth=self.hparams.dice_smooth)
        dice = focal_loss_mean_over_present_classes(logits, targets)

        # Supervised contrastive loss on projected normalized features
        z = self.proj(features)                    # [N, proj_dim]
        z = F.normalize(z, dim=1)
        contr = supervised_nt_xent(z, targets, temperature=self.hparams.temperature)

        total = self.hparams.dice_weight * dice + self.hparams.contrastive_weight * contr
        return total, dice.detach(), contr.detach()

    # -------- steps --------

    def training_step(self, batch, batch_idx):
        x, y = batch
        feat = self.encode(x)
        logits = self.fc2(feat)
        total, dice, contr = self.compute_losses(logits, feat, y)

        self.train_f1.update(logits, y)
        self.log('train_loss', total, prog_bar=True, on_step=True, on_epoch=False)
        self.log('train_dice_loss', dice, on_step=True, on_epoch=False)
        self.log('train_contrastive_loss', contr, on_step=True, on_epoch=False)
        self.log('train_f1_macro', self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        return total
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        feat = self.encode(x)
        logits = self.fc2(feat)
        total, dice, contr = self.compute_losses(logits, feat, y)

        self.val_f1.update(logits, y)
        self.log('val_loss', total, on_step=False, on_epoch=True)
        self.log('val_dice_loss', dice, on_step=False, on_epoch=True)
        self.log('val_contrastive_loss', contr, on_step=False, on_epoch=True)
        self.log('val_f1_macro', self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        return total

    def test_step(self, batch, batch_idx):
        x, y = batch
        feat = self.encode(x)
        logits = self.fc2(feat)
        total, dice, contr = self.compute_losses(logits, feat, y)

        self.test_f1.update(logits, y)
        self.log('test_loss', total, on_step=False, on_epoch=True)
        self.log('test_dice_loss', dice, on_step=False, on_epoch=True)
        self.log('test_contrastive_loss', contr, on_step=False, on_epoch=True)
        self.log('test_f1_macro', self.test_f1, on_step=False, on_epoch=True)
        return total
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=3,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_f1_macro'
            }
        }


import torch
import torch.nn as nn
import lightning as L
from torchmetrics.classification import F1Score


class ResidualBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.3):
        super().__init__()
        mid_channels = out_channels // 4  # bottleneck
        self.conv1 = nn.Conv1d(in_channels, mid_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(mid_channels)
        self.conv2 = nn.Conv1d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(mid_channels)
        self.conv3 = nn.Conv1d(mid_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.dropout(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        return self.relu(out)


class CNNLSTMPhonemeModel1(L.LightningModule):
    """
    CNN-LSTM model for phoneme classification from MEG data.

    Updated Architecture:
    - Residual bottleneck CNN layers with stride=2 downsampling
    - Positional embeddings (cropped at inference)
    - LSTM for temporal modeling
    - Max pooling + classification head
    """

    def __init__(self,
                 cnn_channels=[64, 128, 256],
                 lstm_hidden_size=128,
                 lstm_num_layers=2,
                 dropout=0.3,
                 learning_rate=0.0005,
                 max_seq_len=1024,
                 num_classes=39,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # CNN with residual bottleneck blocks
        cnn_layers = []
        in_channels = 306
        for out_channels in cnn_channels:
            cnn_layers.append(
                ResidualBottleneckBlock(in_channels, out_channels, stride=2, dropout=dropout)
            )
            in_channels = out_channels
        self.cnn = nn.Sequential(*cnn_layers)

        # Positional embeddings
        self.positional_embeddings = nn.Parameter(torch.randn(max_seq_len, cnn_channels[-1]))

        # LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=True
        )

        # Max pooling after LSTM
        self.pool = nn.AdaptiveMaxPool1d(1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.f1_macro = F1Score(num_classes=num_classes, average='macro', task="multiclass")

    def forward(self, x):
        # x shape: (batch, 306, 125)

        # CNN
        cnn_out = self.cnn(x)  # (batch, channels, time)

        # Add positional embeddings
        b, c, t = cnn_out.shape
        pos_emb = self.positional_embeddings[:t, :].unsqueeze(0).expand(b, t, c)  # (batch, time, channels)
        cnn_out = cnn_out.transpose(1, 2) + pos_emb  # (batch, time, channels)

        # LSTM
        lstm_out, _ = self.lstm(cnn_out)  # (batch, time, hidden*2)

        # Max pooling over time
        lstm_out = lstm_out.transpose(1, 2)  # (batch, hidden*2, time)
        pooled = self.pool(lstm_out).squeeze(-1)  # (batch, hidden*2)

        # Classification
        logits = self.classifier(pooled)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        f1_macro = self.f1_macro(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_f1_macro', f1_macro)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=3,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_f1_macro'}
        }










class GlobalMaxPooling1D(nn.Module):
    """Applies Global Max Pooling on the timesteps dimension."""

    def forward(self, x: torch.Tensor):
        return x.amax(dim=1)


class FirstTokenPooling1D(nn.Module):
    """Takes the first token's embedding."""

    def forward(self, x: torch.Tensor):
        return x[:, 0, :]


class LastTokenPooling1D(nn.Module):
    """Takes the last token's embedding."""

    def forward(self, x: torch.Tensor):
        return x[:, -1, :]


class GlobalAvgPooling1D(nn.Module):
    """Applies Global Average Pooling on the timesteps dimension."""

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ):
        if attention_mask is not None:
            attention_mask = attention_mask.repeat((1, 1, x.shape[-1])).to(
                dtype=x.dtype
            )
            x = x * attention_mask
            return x.sum(1) / attention_mask.sum(1)
        else:
            return x.mean(dim=1)


class GlobalSumPooling1D(nn.Module):
    """Applies Global Sum Pooling on the timesteps dimension."""

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        if attention_mask is not None:
            x = x * attention_mask
        return x.sum(dim=1)


class GlobalRMSPooling1D(nn.Module):
    """Applies Global RMS Pooling on the timesteps dimension."""

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        if attention_mask is not None:
            attention_mask = attention_mask.repeat((1, 1, x.shape[-1])).to(
                dtype=x.dtype
            )
            x = x * attention_mask
            return (x.pow(2).sum(dim=1) / attention_mask.sum(1)).sqrt()
        else:
            return x.pow(2).mean(dim=1).sqrt()


class GlobalAbsMaxPooling1D(nn.Module):
    """Applies Global Max Pooling of absolute values on the timesteps dimension."""

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        if attention_mask is not None:
            attention_mask = attention_mask.repeat((1, 1, x.shape[-1])).to(
                dtype=x.dtype
            )
            x = x * attention_mask
        return x.abs().amax(dim=1)


class GlobalAbsAvgPooling1D(nn.Module):
    """Applies Global Average Pooling of absolute values on the timesteps dimension."""

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        if attention_mask is not None:
            attention_mask = attention_mask.repeat((1, 1, x.shape[-1])).to(
                dtype=x.dtype
            )
            x = (x * attention_mask).abs()
            return x.sum(dim=1) / attention_mask.sum(1)
        else:
            return x.abs().mean(dim=1)

POOLING2OBJECT = {
    'max': GlobalMaxPooling1D,
    'first': FirstTokenPooling1D,
    'last': LastTokenPooling1D,
    'avg': GlobalAvgPooling1D,
    'sum': GlobalSumPooling1D,
    'rms': GlobalRMSPooling1D,
    'abs_max': GlobalAbsMaxPooling1D,
    'abs_avg': GlobalAbsAvgPooling1D
}


class KernelsAttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_kernels, pooling_type='avg'):
        super().__init__()
        self.pooling = POOLING2OBJECT[pooling_type]()
        self.to_scores = nn.Sequential(nn.Linear(hidden_size, hidden_size*2),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(hidden_size*2, num_kernels))

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.LongTensor] = None, temperature=1):
        out = self.pooling(hidden_states, attention_mask)
        scores = self.to_scores(out)
        return F.softmax(scores / temperature, dim=-1)


# --- LayerNorm over channels for (B, C, L) tensors ---
class ChannelLayerNorm1d(nn.Module):
    """
    Applies LayerNorm over the channel dimension at each time step.
    Internally uses nn.LayerNorm over the last dim after (B,C,L) -> (B,L,C).
    """
    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, L)
        return self.ln(x.transpose(1, 2)).transpose(1, 2)

# --- A thin adapter so DynamicConvLayer can be used like Conv1d (input: B,C,L) ---
class DynamicConv1d(nn.Module):
    """
    Wraps your DynamicConvLayer (which expects [B, L, C]) so it plugs into CNN code that
    uses [B, C, L]. Supports stride and explicit padding for full Conv1d parity.
    """
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        *,
        num_heads: int = 1,
        dilation: int = 1,
        stride: int = 1,
        padding: Optional[int] = None,   # None -> "same" when stride==1, else 0 by default (match provided Table10Net)
        hidden_dropout_prob: float = 0.0,
        conv_act: str = "relu",
        nof_kernels: int = 16,
        pooling_type: str = "avg",
        bias: bool = True,
        temperature: float = 1.0,        # default mixture temp
    ):
        super().__init__()
        # we extend your DynamicConvLayer to accept stride + padding (see next cell)
        self.inner = DynamicConvLayer(
            hidden_size=channels,
            kernel_size=kernel_size,
            num_heads=num_heads,
            dilation=dilation,
            hidden_dropout_prob=hidden_dropout_prob,
            conv_act=conv_act,
            nof_kernels=nof_kernels,
            pooling_type=pooling_type,
            bias=bias,
            stride=stride,
            padding=padding,
        )
        self.temperature = float(temperature)

    def forward(self, x_bcl: torch.Tensor) -> torch.Tensor:
        # (B,C,L) -> (B,L,C)
        x_blc = x_bcl.transpose(1, 2).contiguous()
        y_blc = self.inner(x_blc, temperature=self.temperature)  # [B,L,C]
        return y_blc.transpose(1, 2).contiguous()



class DynamicConvLayer(nn.Module):
    """
    Dynamic (sample-conditioned) 1D grouped convolution.

    Args:
        hidden_size: in/out channels (must match).
        kernel_size: kernel width.
        num_heads: number of groups (must divide hidden_size).
        dilation: conv dilation.
        stride: conv stride (default 1).
        padding: explicit padding; if None, uses 'same' when stride==1 else 0.
        hidden_dropout_prob: dropout after conv.
        conv_act: key into ACT2FN.
        nof_kernels: number of candidate kernels in the bank.
        pooling_type: how to pool for kernel attention.
        bias: conv bias.
    """
    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        num_heads: int,
        dilation: int = 1,
        *,
        stride: int = 1,
        padding: Optional[int] = None,
        hidden_dropout_prob: float = 0.0,
        conv_act: str = "relu",
        nof_kernels: int = 16,
        pooling_type: str = "avg",
        bias: bool = True,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.hidden_size = hidden_size
        self.groups = num_heads
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.stride = int(stride)
        # keep original 'same' default for stride=1; your downsample uses padding=0 explicitly
        if padding is None:
            self.padding = (self.kernel_size - 1) // 2 if self.stride == 1 else 0
        else:
            self.padding = int(padding)

        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.conv_act = conv_act

        self.attention = KernelsAttentionLayer(hidden_size, nof_kernels, pooling_type)

        c_in_per_group = hidden_size // self.groups
        self.nof_kernels = nof_kernels
        self.kernels_weights = nn.Parameter(
            torch.empty(nof_kernels, hidden_size, c_in_per_group, self.kernel_size)
        )
        if bias:
            self.kernels_bias = nn.Parameter(torch.empty(nof_kernels, hidden_size))
        else:
            self.register_parameter("kernels_bias", None)

        self._reset_parameters()

    def _reset_parameters(self):
        for k in range(self.nof_kernels):
            nn.init.kaiming_uniform_(self.kernels_weights[k], a=math.sqrt(5))
        if self.kernels_bias is not None:
            fan_in = self.kernels_weights[0, 0].numel()
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.kernels_bias, -bound, bound)

    @torch.no_grad()
    def _check_shapes(self, x_bcl: torch.Tensor):
        B, C, L = x_bcl.shape
        assert C == self.hidden_size

    def forward(
        self,
        hidden_states: torch.Tensor,                      # [B, L, C]
        temperature: float = 1.0,
    ) -> torch.Tensor:
        alphas = self.attention(hidden_states, temperature=temperature)  # [B, K]

        x = hidden_states.permute(0, 2, 1).contiguous()  # [B, C, L]
        B, C_in, L = x.shape
        self._check_shapes(x)

        w = self.kernels_weights
        w_b = (alphas.view(B, self.nof_kernels, 1, 1, 1) * w.view(1, *w.shape)).sum(dim=1)  # [B, C_out, C_in/G, Kw]

        b_b = (alphas @ self.kernels_bias) if self.kernels_bias is not None else None  # [B, C_out]

        groups_total = B * self.groups
        w_b_flat = w_b.reshape(B * self.hidden_size, C_in // self.groups, self.kernel_size).contiguous()
        x_grouped = x.reshape(1, B * C_in, L)  # [1, B*C_in, L]
        b_flat = b_b.reshape(-1) if b_b is not None else None

        out = F.conv1d(
            x_grouped,
            w_b_flat,
            b_flat,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=groups_total,
        )  # [1, B*C_out, L']
        out = out.view(B, self.hidden_size, out.shape[-1]).permute(0, 2, 1).contiguous()

        output_states = ACT2FN[self.conv_act](self.dropout(out))
        return output_states

class DynamicConvLayer(nn.Module):
    """
    Dynamic (sample-conditioned) 1D grouped convolution.

    Args:
        hidden_size: in/out channels (must match).
        kernel_size: kernel width.
        num_heads: number of groups (must divide hidden_size).
        dilation: conv dilation.
        stride: conv stride (default 1).
        padding: explicit padding; if None, uses 'same' when stride==1 else 0.
        hidden_dropout_prob: dropout after conv.
        conv_act: key into ACT2FN.
        nof_kernels: number of candidate kernels in the bank.
        pooling_type: how to pool for kernel attention.
        bias: conv bias.
    """
    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        num_heads: int,
        dilation: int = 1,
        *,
        stride: int = 1,
        padding: Optional[int] = None,
        hidden_dropout_prob: float = 0.0,
        conv_act: str = "relu",
        nof_kernels: int = 16,
        pooling_type: str = "avg",
        bias: bool = True,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.hidden_size = hidden_size
        self.groups = num_heads
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.stride = int(stride)
        # keep original 'same' default for stride=1; your downsample uses padding=0 explicitly
        if padding is None:
            self.padding = (self.kernel_size - 1) // 2 if self.stride == 1 else 0
        else:
            self.padding = int(padding)

        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.conv_act = conv_act

        self.attention = KernelsAttentionLayer(hidden_size, nof_kernels, pooling_type)

        c_in_per_group = hidden_size // self.groups
        self.nof_kernels = nof_kernels
        self.kernels_weights = nn.Parameter(
            torch.empty(nof_kernels, hidden_size, c_in_per_group, self.kernel_size)
        )
        if bias:
            self.kernels_bias = nn.Parameter(torch.empty(nof_kernels, hidden_size))
        else:
            self.register_parameter("kernels_bias", None)

        self._reset_parameters()

    def _reset_parameters(self):
        for k in range(self.nof_kernels):
            nn.init.kaiming_uniform_(self.kernels_weights[k], a=math.sqrt(5))
        if self.kernels_bias is not None:
            fan_in = self.kernels_weights[0, 0].numel()
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.kernels_bias, -bound, bound)

    @torch.no_grad()
    def _check_shapes(self, x_bcl: torch.Tensor):
        B, C, L = x_bcl.shape
        assert C == self.hidden_size

    def forward(
        self,
        hidden_states: torch.Tensor,                      # [B, L, C]
        temperature: float = 1.0,
    ) -> torch.Tensor:
        alphas = self.attention(hidden_states, temperature=temperature)  # [B, K]

        x = hidden_states.permute(0, 2, 1).contiguous()  # [B, C, L]
        B, C_in, L = x.shape
        self._check_shapes(x)

        w = self.kernels_weights
        w_b = (alphas.view(B, self.nof_kernels, 1, 1, 1) * w.view(1, *w.shape)).sum(dim=1)  # [B, C_out, C_in/G, Kw]

        b_b = (alphas @ self.kernels_bias) if self.kernels_bias is not None else None  # [B, C_out]

        groups_total = B * self.groups
        w_b_flat = w_b.reshape(B * self.hidden_size, C_in // self.groups, self.kernel_size).contiguous()
        x_grouped = x.reshape(1, B * C_in, L)  # [1, B*C_in, L]
        b_flat = b_b.reshape(-1) if b_b is not None else None

        out = F.conv1d(
            x_grouped,
            w_b_flat,
            b_flat,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=groups_total,
        )  # [1, B*C_out, L']
        out = out.view(B, self.hidden_size, out.shape[-1]).permute(0, 2, 1).contiguous()

        output_states = ACT2FN[self.conv_act](self.dropout(out))
        return output_states

class ResidualBlock1D(nn.Module):
    """
    PreNorm -> DynConv(k=3) -> PreNorm -> DynConv(k=1) with identity skip.
    No activation inside the block (you keep ELU after the block).
    """
    def __init__(
        self,
        channels: int,
        *,
        num_heads: int = 1,
        nof_kernels: int = 16,
        conv_act: str = "relu",
        dropout: float = 0.0,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.ln1 = ChannelLayerNorm1d(channels)
        self.conv1 = DynamicConv1d(
            channels, kernel_size=3,
            num_heads=num_heads, dilation=1, stride=1, padding=1,
            hidden_dropout_prob=dropout, conv_act=conv_act,
            nof_kernels=nof_kernels, temperature=temperature
        )
        self.ln2 = ChannelLayerNorm1d(channels)
        self.conv2 = DynamicConv1d(
            channels, kernel_size=1,
            num_heads=num_heads, dilation=1, stride=1, padding=0,
            hidden_dropout_prob=dropout, conv_act=conv_act,
            nof_kernels=nof_kernels, temperature=temperature
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.ln1(x))
        out = self.conv2(self.ln2(out))
        return x + out


class Table10Net(L.LightningModule):
    def __init__(self, num_classes: int = 39, 
                 seq_len: int = 125, 
                 dropout_p: float = 0.5,
                 learning_rate: float = 5e-4,
                 temperature: float = 0.1,           # for contrastive; dyn-kernels use dyn_temperature below
                 dice_smooth: float = 1.0,
                 dice_weight: float = 1.0,
                 contrastive_weight: float = 0.3,
                 proj_dim: int = 64,
                 rope_base: float = 10_000.0,
                 rope_scale: float = 1.0,
                 *,
                 dyn_heads: int = 1,
                 dyn_kernels: int = 16,
                 dyn_dropout: float = 0.0,
                 dyn_conv_act: str = "relu",
                 dyn_temperature: float = 1.0,
                 **kwargs):
        super().__init__()
        print("preparing dynamical conv version")
        self.save_hyperparameters()
        hidden_dim = 500
        self.pretrainer = BalancedPhonemePretrainer(num_classes)

        # (1) "Conv1d 306->128, k=7" becomes: 1x1 proj -> LN -> DynamicConv1d(k=7)
        self.proj_in = nn.Conv1d(306, 128, kernel_size=1, stride=1, padding=0, bias=True)
        self.ln1_pre = ChannelLayerNorm1d(128)
        self.conv1_dyn = DynamicConv1d(
            channels=128, kernel_size=7,
            num_heads=dyn_heads, dilation=1, stride=1, padding=3,
            hidden_dropout_prob=dyn_dropout, conv_act=dyn_conv_act,
            nof_kernels=dyn_kernels, temperature=dyn_temperature
        )

        # (2a/2b) Residual block with dynamic convs + pre-norm
        self.resblock = ResidualBlock1D(
            128, num_heads=dyn_heads, nof_kernels=dyn_kernels,
            conv_act=dyn_conv_act, dropout=dyn_dropout, temperature=dyn_temperature
        )

        # (3) ELU
        self.elu1 = nn.ELU()

        # (4) Downsampling dynamic conv: k=50, s=25, padding=0
        self.ln_down = ChannelLayerNorm1d(128)
        self.down_dyn = DynamicConv1d(
            channels=128, kernel_size=50,
            num_heads=dyn_heads, dilation=1, stride=25, padding=0,
            hidden_dropout_prob=dyn_dropout, conv_act=dyn_conv_act,
            nof_kernels=dyn_kernels, temperature=dyn_temperature
        )

        # (5) ELU
        self.elu2 = nn.ELU()

        # (6) Conv1d k=7 -> dynamic
        self.ln6 = ChannelLayerNorm1d(128)
        self.conv2_dyn = DynamicConv1d(
            channels=128, kernel_size=7,
            num_heads=dyn_heads, dilation=1, stride=1, padding=3,
            hidden_dropout_prob=dyn_dropout, conv_act=dyn_conv_act,
            nof_kernels=dyn_kernels, temperature=dyn_temperature
        )

        # (7) ELU
        self.elu3 = nn.ELU()

        # Infer post-convolution temporal length for the head (unchanged)
        with torch.no_grad():
            dummy = torch.zeros(1, 306, seq_len)
            feats = self._forward_features(dummy)
            L = feats.shape[-1]
        self.to_len4 = nn.Identity() if (128 * L == 512) else nn.AdaptiveAvgPool1d(4)

        # Head (unchanged)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * (L if isinstance(self.to_len4, nn.Identity) else 4), 512)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(512, num_classes)

        # Projection head & metrics (unchanged)
        self.proj = nn.Sequential(nn.Linear(512, 128), nn.ReLU(inplace=True), nn.Linear(128, proj_dim))
        self.train_f1 = MulticlassF1Score(num_classes=num_classes, average='macro')
        self.val_f1   = MulticlassF1Score(num_classes=num_classes, average='macro')
        self.test_f1  = MulticlassF1Score(num_classes=num_classes, average='macro')
        self.learning_rate = learning_rate

    # _apply_rope_to_channels unchanged

    def _forward_features(self, x):
        # 1) proj -> LN -> DynConv(k=7)
        x = self.proj_in(x)
        x = self.conv1_dyn(self.ln1_pre(x))

        # 2a/2b) Dynamic residual block
        x = self.resblock(x)

        # 3) ELU
        x = self.elu1(x)

        # 4) LN -> Dynamic downsample (k=50, s=25, pad=0)
        x = self.down_dyn(self.ln_down(x))

        # 5) ELU
        x = self.elu2(x)

        # 6) LN -> Dynamic k=7
        x = self.conv2_dyn(self.ln6(x))

        # 7) ELU
        x = self.elu3(x)
        return x

    def encode(self, x):
        """
        Returns the penultimate 512-d feature used for classification (after ReLU+Dropout).
        """
        x = self._forward_features(x)
        x = self.to_len4(x)                # ensure Flatten -> 512 features
        x = self.flatten(x)                # layer 8
        x = self.fc1(x)                    # layer 9
        x = self.relu(x)                   # layer 10
        x = self.drop(x)                   # layer 11
        return x                           # [N, 512]

    def forward(self, x):
        feat = self.encode(x)
        logits = self.fc2(feat)            # layer 12
        return logits

    def compute_losses(self, logits, features, targets):
        # Dice classification loss
        # dice = dice_loss_from_logits(logits, targets, smooth=self.hparams.dice_smooth)
        dice = focal_loss_mean_over_present_classes(logits, targets)

        # Supervised contrastive loss on projected normalized features
        z = self.proj(features)                    # [N, proj_dim]
        z = F.normalize(z, dim=1)
        contr = supervised_nt_xent(z, targets, temperature=self.hparams.temperature)

        total = self.hparams.dice_weight * dice + self.hparams.contrastive_weight * contr
        return total, dice.detach(), contr.detach()

    # -------- steps --------

    def training_step(self, batch, batch_idx):
        x, y = batch
        feat = self.encode(x)
        logits = self.fc2(feat)
        total, dice, contr = self.compute_losses(logits, feat, y)

        self.train_f1.update(logits, y)
        self.log('train_loss', total, prog_bar=True, on_step=True, on_epoch=False)
        self.log('train_dice_loss', dice, on_step=True, on_epoch=False)
        self.log('train_contrastive_loss', contr, on_step=True, on_epoch=False)
        self.log('train_f1_macro', self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        return total
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        feat = self.encode(x)
        logits = self.fc2(feat)
        total, dice, contr = self.compute_losses(logits, feat, y)

        self.val_f1.update(logits, y)
        self.log('val_loss', total, on_step=False, on_epoch=True)
        self.log('val_dice_loss', dice, on_step=False, on_epoch=True)
        self.log('val_contrastive_loss', contr, on_step=False, on_epoch=True)
        self.log('val_f1_macro', self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        return total

    def test_step(self, batch, batch_idx):
        x, y = batch
        feat = self.encode(x)
        logits = self.fc2(feat)
        total, dice, contr = self.compute_losses(logits, feat, y)

        self.test_f1.update(logits, y)
        self.log('test_loss', total, on_step=False, on_epoch=True)
        self.log('test_dice_loss', dice, on_step=False, on_epoch=True)
        self.log('test_contrastive_loss', contr, on_step=False, on_epoch=True)
        self.log('test_f1_macro', self.test_f1, on_step=False, on_epoch=True)
        return total
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=3,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_f1_macro'
            }
        }




class FourierEmbedding(nn.Module):
    """Fourier positional embeddings for 1D sequences"""
    def __init__(self, embed_dim=64, max_len=10000):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: (batch_size, channels, seq_len)
        batch_size, channels, seq_len = x.shape
        pos_embed = self.pe[:seq_len].T.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Concatenate input with positional embeddings along channel dimension
        return torch.cat([x, pos_embed], dim=1)

class DilatedResidualBlock(nn.Module):
    """Dilated residual block for 1D convolutions"""
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, 
                              dilation=dilation, padding=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, 
                              dilation=dilation, padding=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        out += residual  # Residual connection
        return F.relu(out)

class Table10Net1(L.LightningModule):
    """
    Updated architecture with Fourier embeddings and dilated residual blocks.
    
    Architecture:
    - Fourier embeddings + spatial pooling 
    - Conv1d (64 → 32)
    - 7 dilated residual blocks (32 channels, dilation up to 64)
    - flatten (48k) 
    - fully connected (48k → 256 → 32 → num_classes)
    
    Expected default input: (N, 306, 125)
    """
    
    def __init__(self, 
                 num_classes: int = 39, 
                 seq_len: int = 125, 
                 dropout_p: float = 0.5,
                 learning_rate: float = 5e-4,
                 temperature: float = 0.1,  # used for contrastive
                 dice_smooth: float = 1.0,  # smoothing for Dice
                 dice_weight: float = 1.0,  # weight for Dice loss
                 contrastive_weight: float = 0.3,  # weight for contrastive loss
                 proj_dim: int = 64,  # projection head output size
                 rope_base: float = 10_000.0,
                 rope_scale: float = 1.0,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        # Keep the pretrainer if needed
        # self.pretrainer = BalancedPhonemePretrainer(num_classes)
        
        # Fourier embeddings (adds 64 dimensions to input channels)
        self.fourier_embed = FourierEmbedding(embed_dim=64)
        fourier_output_dim = 306 + 64  # 370 channels after fourier
        
        # Spatial pooling to manage sequence length
        self.spatial_pool = nn.AdaptiveAvgPool1d(seq_len)  # Keep same length initially
        
        # Project from fourier output to 64 channels
        self.input_proj = nn.Conv1d(fourier_output_dim, 64, kernel_size=1)
        
        # Conv1d: 64 → 32 channels
        self.conv1 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        
        # 7 dilated residual blocks with increasing dilation (1, 2, 4, 8, 16, 32, 64)
        self.residual_blocks = nn.ModuleList([
            DilatedResidualBlock(32, dilation=1),
            DilatedResidualBlock(32, dilation=2),
            DilatedResidualBlock(32, dilation=4),
            DilatedResidualBlock(32, dilation=8),
            DilatedResidualBlock(32, dilation=16),
            DilatedResidualBlock(32, dilation=32),
            DilatedResidualBlock(32, dilation=64)
        ])
        
        # Calculate flattened size after conv blocks
        with torch.no_grad():
            dummy = torch.zeros(1, 306, seq_len)
            feats = self._forward_features(dummy)
            flattened_size = feats.view(1, -1).size(1)
        
        # Adjust to target ~48k features if needed
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1500) if flattened_size < 40000 else nn.Identity()
        
        # Recalculate after adaptive pooling
        with torch.no_grad():
            dummy = torch.zeros(1, 306, seq_len)
            feats = self._forward_features(dummy)
            if not isinstance(self.adaptive_pool, nn.Identity):
                feats = self.adaptive_pool(feats)
            final_flattened_size = feats.view(1, -1).size(1)
        
        # Flatten
        self.flatten = nn.Flatten()
        
        # Fully connected layers: flattened_size → 256 → 32 → num_classes
        self.fc1 = nn.Linear(final_flattened_size, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, num_classes)
        
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=dropout_p)
        
        # Small projection head for contrastive learning
        self.proj = nn.Sequential(
            nn.Linear(32, 64),  # Project from 32-d penultimate features
            nn.ReLU(inplace=True),
            nn.Linear(64, proj_dim)
        )
        
        # Metrics
        self.train_f1 = MulticlassF1Score(num_classes=num_classes, average='macro')
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average='macro')
        self.test_f1 = MulticlassF1Score(num_classes=num_classes, average='macro')
        
        self.learning_rate = learning_rate
    
    def _forward_features(self, x):
        """Forward through convolutional feature extraction layers"""
        # Apply Fourier embeddings
        x = self.fourier_embed(x)  # (batch_size, 306+64, seq_len)
        
        # Spatial pooling
        x = self.spatial_pool(x)
        
        # Project to 64 channels
        x = self.input_proj(x)
        
        # Conv1d: 64 → 32
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Apply dilated residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        return x
    
    def encode(self, x):
        """
        Returns the penultimate 32-d feature used for classification.
        """
        x = self._forward_features(x)
        
        # Apply adaptive pooling if needed
        if not isinstance(self.adaptive_pool, nn.Identity):
            x = self.adaptive_pool(x)
        
        # Flatten
        x = self.flatten(x)
        
        # FC layers: flattened → 256 → 32
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.relu(self.fc2(x))
        x = self.drop(x)
        
        return x  # [N, 32]
    
    def forward(self, x):
        feat = self.encode(x)
        logits = self.fc3(feat)  # 32 → num_classes
        return logits
    
    def compute_losses(self, logits, features, targets):
        # Dice classification loss
        # dice = dice_loss_from_logits(logits, targets, smooth=self.hparams.dice_smooth)
        dice = focal_loss_mean_over_present_classes(logits, targets)

        # Supervised contrastive loss on projected normalized features
        z = self.proj(features)                    # [N, proj_dim]
        z = F.normalize(z, dim=1)
        contr = supervised_nt_xent(z, targets, temperature=self.hparams.temperature)

        total = self.hparams.dice_weight * dice + self.hparams.contrastive_weight * contr
        return total, dice.detach(), contr.detach()
    
    # -------- steps --------
    def training_step(self, batch, batch_idx):
        x, y = batch
        feat = self.encode(x)
        logits = self.fc3(feat)
        total, dice, contr = self.compute_losses(logits, feat, y)
        
        self.train_f1.update(logits, y)
        self.log('train_loss', total, prog_bar=True, on_step=True, on_epoch=False)
        self.log('train_dice_loss', dice, on_step=True, on_epoch=False)
        self.log('train_contrastive_loss', contr, on_step=True, on_epoch=False)
        self.log('train_f1_macro', self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        return total
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        feat = self.encode(x)
        logits = self.fc3(feat)
        total, dice, contr = self.compute_losses(logits, feat, y)
        
        self.val_f1.update(logits, y)
        self.log('val_loss', total, on_step=False, on_epoch=True)
        self.log('val_dice_loss', dice, on_step=False, on_epoch=True)
        self.log('val_contrastive_loss', contr, on_step=False, on_epoch=True)
        self.log('val_f1_macro', self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        return total
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        feat = self.encode(x)
        logits = self.fc3(feat)
        total, dice, contr = self.compute_losses(logits, feat, y)
        
        self.test_f1.update(logits, y)
        self.log('test_loss', total, on_step=False, on_epoch=True)
        self.log('test_dice_loss', dice, on_step=False, on_epoch=True)
        self.log('test_contrastive_loss', contr, on_step=False, on_epoch=True)
        self.log('test_f1_macro', self.test_f1, on_step=False, on_epoch=True)
        return total
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_f1_macro'
            }
        }
