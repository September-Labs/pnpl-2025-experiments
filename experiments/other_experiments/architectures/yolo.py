"""
YOLO-MEG: YOLO Backbone Adapted for MEG Phoneme Classification
LibriBrain Competition Implementation
"""

import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple, Optional
import torchmetrics
import lightning as L


# ============================================
# Helper Functions
# ============================================

def autopad(k, p=None):
    """Auto-padding for convolutions."""
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


def round_ch(c: int, width: float, divisor: int = 8) -> int:
    """Round channels to multiple of divisor for tensor core friendliness."""
    c = max(1, int(c * width))
    return int(math.ceil(c / divisor) * divisor)


def round_n(n: int, depth: float) -> int:
    """Round number of blocks based on depth multiplier."""
    return max(1, int(round(n * depth)))


# ============================================
# YOLO Building Blocks for 1D MEG Data
# ============================================

class Conv1D(nn.Module):
    """Conv -> BN -> SiLU for 1D signals (N, C, L)."""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv1d(c1, c2, k, s, autopad(k, p), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm1d(c2)
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv1D(Conv1D):
    """Depthwise separable convolution for 1D."""
    def __init__(self, c1, c2, k=3, s=1, act=True):
        super().__init__(c1, c2, k, s, g=c1, act=act)


class Bottleneck1D(nn.Module):
    """Standard bottleneck with 1x1 -> k and optional residual for 1D."""
    def __init__(self, c1, c2, shortcut=True, g=1, k=3, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv1D(c1, c_, 1, 1)
        self.cv2 = Conv1D(c_, c2, k, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C3k1D(nn.Module):
    """C3-style block with adjustable kernel size k for 1D."""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv1D(c1, c_, 1, 1)
        self.cv2 = Conv1D(c1, c_, 1, 1)
        self.cv3 = Conv1D(c_, c2, 1, 1)
        self.m = nn.Sequential(*(Bottleneck1D(c_, c_, shortcut, g, k=k, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(self.m(self.cv2(x)) + self.cv1(x))


class C3k2_1D(nn.Module):
    """YOLO11-like C2f/C3 hybrid for 1D."""
    def __init__(self, c1, c2, n=1, use_c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c_ = int(c2 * e)
        self.cv1 = Conv1D(c1, self.c_, 1, 1)
        self.cv2 = Conv1D(c1, self.c_, 1, 1)
        self.cv3 = Conv1D((2 + n) * self.c_, c2, 1, 1)
        self.m = nn.ModuleList(
            (C3k1D(self.c_, self.c_, n=2, shortcut=shortcut, g=g)
             if use_c3k
             else Bottleneck1D(self.c_, self.c_, shortcut, g, k=3, e=1.0))
            for _ in range(n)
        )

    def forward(self, x):
        y = [self.cv2(x), self.cv1(x)]
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv3(torch.cat(y, 1))


class SPPF1D(nn.Module):
    """Spatial Pyramid Pooling - Fast for 1D MEG signals."""
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv1D(c1, c_, 1, 1)
        self.m = nn.MaxPool1d(kernel_size=k, stride=1, padding=k // 2)
        self.cv2 = Conv1D(c_ * 4, c2, 1, 1)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), 1))


class Attention1D(nn.Module):
    """Attention mechanism for 1D MEG signals."""
    def __init__(self, c, attn_ratio=0.5, num_heads=4, dropout=0.0, use_sdpa=True):
        super().__init__()
        self.h = max(1, int(num_heads))
        qk = max(self.h, int(c * attn_ratio / self.h) * self.h)
        vc = (c // self.h) * self.h
        self.d = qk // self.h
        self.dv = vc // self.h
        self.use_sdpa = use_sdpa and hasattr(F, "scaled_dot_product_attention")
        self.q = Conv1D(c, qk, 1, act=False)
        self.k = Conv1D(c, qk, 1, act=False)
        self.v = Conv1D(c, vc, 1, act=False)
        self.proj = Conv1D(vc, c, 1, act=False)

    def forward(self, x):
        b, _, L = x.shape
        N = L
        q = self.q(x).reshape(b, self.h, self.d, N).transpose(2, 3)
        k = self.k(x).reshape(b, self.h, self.d, N).transpose(2, 3)
        v = self.v(x).reshape(b, self.h, self.dv, N).transpose(2, 3)
        
        if self.use_sdpa:
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            attn = (q.float() @ k.float().transpose(-2, -1)) * (self.d ** -0.5)
            attn = attn.softmax(-1)
            out = (attn @ v.float()).to(q.dtype)
        
        out = out.transpose(2, 3).reshape(b, self.dv * self.h, L)
        return self.proj(out)


class C2PSA1D(nn.Module):
    """Position-Sensitive Attention module for 1D."""
    def __init__(self, c1, c2, n: int = 2, inter_coef: float = 0.5):
        super().__init__()
        assert c1 == c2
        c = int(c1 * inter_coef)
        self.cv1 = Conv1D(c1, 2 * c, 1, 1)
        self.cv2 = Conv1D(2 * c, c1, 1, 1)
        self.m = nn.Sequential(*(nn.Sequential(
            Attention1D(c, attn_ratio=0.5, num_heads=max(1, c // 64)),
            Conv1D(c, 2 * c, 1),
            Conv1D(2 * c, c, 1, act=False)
        ) for _ in range(n)))

    def forward(self, x):
        a, b = self.cv1(x).split((self.cv1.conv.out_channels // 2, 
                                  self.cv1.conv.out_channels // 2), dim=1)
        b = self.m(b) + b
        return self.cv2(torch.cat((a, b), dim=1))


# ============================================
# MEG-YOLO Backbone
# ============================================

class MEGYoloBackbone1D(nn.Module):
    """YOLO backbone adapted for MEG data (306 channels, 125 time points)."""
    
    def __init__(self, 
                 n_channels: int = 306,
                 depth: float = 0.50,
                 width: float = 0.25,
                 use_deconv: bool = False):
        super().__init__()
        self.depth = float(depth)
        self.width = float(width)
        self.use_deconv = bool(use_deconv)

        # Channel width scaling for MEG
        c64   = round_ch(64,   self.width)
        c128  = round_ch(128,  self.width)
        c256  = round_ch(256,  self.width)
        c512  = round_ch(512,  self.width)
        c1024 = round_ch(1024, self.width)

        # Initial MEG channel projection
        self.meg_proj = Conv1D(n_channels, c64, 1)
        
        # Encoder path
        self.b0  = Conv1D(c64, c64, 3, 2)                                      # P1/2
        self.b1  = Conv1D(c64, c128, 3, 2)                                     # P2/4
        self.b2  = C3k2_1D(c128, c256, n=round_n(2, self.depth), use_c3k=False, e=0.25)
        self.b3  = Conv1D(c256, c256, 3, 2)                                    # P3/8
        self.b4  = C3k2_1D(c256, c512, n=round_n(2, self.depth), use_c3k=False, e=0.25)
        self.b5  = Conv1D(c512, c512, 3, 2)                                    # P4/16
        self.b6  = C3k2_1D(c512, c512, n=round_n(2, self.depth), use_c3k=True)
        self.b7  = Conv1D(c512, c1024, 3, 2)                                   # P5/32
        self.b8  = C3k2_1D(c1024, c1024, n=round_n(2, self.depth), use_c3k=True)
        self.b9  = SPPF1D(c1024, c1024, k=5)
        self.b10 = C2PSA1D(c1024, c1024, n=round_n(2, self.depth))

        # Decoder path
        if self.use_deconv:
            self.up1 = nn.ConvTranspose1d(c1024, c1024, 2, 2, 0, bias=False)
            self.up2 = nn.ConvTranspose1d(c512, c512, 2, 2, 0, bias=False)
        else:
            self.up = nn.Upsample(scale_factor=2, mode="nearest")

        self.cat = nn.Identity()  # Concatenation placeholder
        self.h13 = C3k2_1D(c1024 + c512, c512, n=round_n(2, self.depth), use_c3k=False)
        self.h16 = C3k2_1D(c512 + c512, c256, n=round_n(2, self.depth), use_c3k=False)
        self.down3 = Conv1D(c256, c256, 3, 2)
        self.h19 = C3k2_1D(c256 + c512, c512, n=round_n(2, self.depth), use_c3k=False)
        self.down4 = Conv1D(c512, c512, 3, 2)
        self.h22 = C3k2_1D(c512 + c1024, c1024, n=round_n(2, self.depth), use_c3k=True)

        self.out_channels = (c256, c512, c1024)

    def forward(self, x):
        # x: (B, C, L) where C=306 (MEG channels), L=125 (time points)
        x = self.meg_proj(x)
        
        # Encoder
        x0  = self.b0(x)
        x1  = self.b1(x0)
        x2  = self.b2(x1)
        x3  = self.b3(x2)
        x4  = self.b4(x3)    # P3 skip
        x5  = self.b5(x4)
        x6  = self.b6(x5)    # P4 skip
        x7  = self.b7(x6)
        x8  = self.b8(x7)
        x9  = self.b9(x8)
        x10 = self.b10(x9)   # Top feature
        
        # Decoder with skip connections
        if self.use_deconv:
            u4 = self.up1(x10)
            # Ensure size matching
            if u4.size(-1) != x6.size(-1):
                u4 = F.interpolate(u4, size=x6.shape[-1], mode="nearest")
        else:
            u4 = F.interpolate(x10, size=x6.shape[-1], mode="nearest")

        p4 = self.h13(torch.cat([u4, x6], dim=1))
        
        if self.use_deconv:
            u3 = self.up2(p4)
            if u3.size(-1) != x4.size(-1):
                u3 = F.interpolate(u3, size=x4.shape[-1], mode="nearest")
        else:
            u3 = F.interpolate(p4, size=x4.shape[-1], mode="nearest")

        p3 = self.h16(torch.cat([u3, x4], dim=1))
        p4_2 = self.h19(torch.cat([self.down3(p3), p4], dim=1))
        p5_2 = self.h22(torch.cat([self.down4(p4_2), x10], dim=1))
        
        return [p3, p4_2, p5_2]


# ============================================
# MEG-YOLO Phoneme Classification Model
# ============================================

class MEGYOLOPhonemeModel(nn.Module):
    """
    YOLO-based phoneme classifier for MEG data.
    Uses multi-scale features from YOLO backbone with parallel LSTM heads.
    """
    
    def __init__(self,
                 n_channels: int = 306,
                 model_dim: int = 512,
                 num_classes: int = 39,
                 depth: float = 0.5,
                 width: float = 0.25,
                 lstm_layers: int = 2,
                 bidirectional: bool = False,
                 dropout: float = 0.3,
                 share_lstm: bool = False,
                 pool: str = "last",
                 use_attention: bool = True):
        super().__init__()
        assert pool in {"last", "mean", "max"}
        self.pool = pool
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.model_dim = int(model_dim)
        self.use_attention = use_attention

        # YOLO Backbone
        self.backbone = MEGYoloBackbone1D(
            n_channels=n_channels,
            depth=depth,
            width=width,
            use_deconv=False
        )

        c256, c512, c1024 = self.backbone.out_channels
        
        # Project each scale to common dimension
        self.proj3 = nn.Conv1d(c256, model_dim, 1)
        self.proj4 = nn.Conv1d(c512, model_dim, 1)
        self.proj5 = nn.Conv1d(c1024, model_dim, 1)

        self.down1 = nn.Conv1d(model_dim, model_dim, kernel_size=24, stride=12, padding=0, bias=True)
        self.down2 = nn.Conv1d(model_dim, model_dim, kernel_size=24, stride=12, padding=0, bias=True)
        self.down3 = nn.Conv1d(model_dim, model_dim, kernel_size=24, stride=12, padding=0, bias=True)

        self.elu = nn.ELU()

        self.conv1 = nn.Conv1d(model_dim, model_dim, kernel_size=3, stride=1, padding=3, bias=True)
        self.conv2 = nn.Conv1d(model_dim, model_dim, kernel_size=3, stride=1, padding=3, bias=True)
        self.conv3 = nn.Conv1d(model_dim, model_dim, kernel_size=3, stride=1, padding=3, bias=True)

        # Optional attention fusion
        out_dim = model_dim * (2 if bidirectional else 1)
        if use_attention:
            self.scale_attention = nn.Sequential(
                nn.Linear(out_dim * 3, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, 3),
                nn.Softmax(dim=-1)
            )
        
        # Classification heads
        self.head = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim // 2, num_classes)
        )

    def _run_branch(self, feat: torch.Tensor, proj: nn.Module, lstm: nn.Module) -> torch.Tensor:
        """Process a single scale branch."""
        z = proj(feat)                              # (B, M, L_s)
        z = z.transpose(1, 2).contiguous()          # (B, L_s, M)
        out, (hn, cn) = lstm(z)                     # out: (B, L_s, H)
        
        if self.pool == "last":
            pooled = out[:, -1, :]
        elif self.pool == "mean":
            pooled = out.mean(dim=1)
        else:  # max
            pooled = out.max(dim=1)[0]
        
        return pooled  # (B, H)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L) where C=306, L=125
        p3, p4, p5 = self.backbone(x)
        
        # Process each scale
        feat3 = self._run_branch(p3, self.proj3, self.lstm3)
        feat4 = self._run_branch(p4, self.proj4, self.lstm4)
        feat5 = self._run_branch(p5, self.proj5, self.lstm5)
        
        feat3 = self.down1(feat3)
        feat4 = self.down2(feat4)
        feat5 = self.down3(feat5)

        # Fusion strategy
        if self.use_attention:
            # Attention-weighted fusion
            combined = torch.stack([feat3, feat4, feat5], dim=1)  # (B, 3, H)
            attention_input = torch.cat([feat3, feat4, feat5], dim=-1)  # (B, H*3)
            attention_weights = self.scale_attention(attention_input)  # (B, 3)
            attention_weights = attention_weights.unsqueeze(-1)  # (B, 3, 1)
            fused = (combined * attention_weights).sum(dim=1)  # (B, H)
        else:
            # Simple average
            fused = (feat3 + feat4 + feat5) / 3.0
        
        # Final classification
        logits = self.head(fused)
        return logits

class MEGYOLOPhonemeModelLargeK(nn.Module):
    """
    Conv-only YOLO-based phoneme model with large temporal kernels.
    - For each backbone scale: 1x1 proj -> upsample to L=125 ->
      Conv7(same) -> ELU -> Conv50(stride=25, no pad) -> ELU -> Conv7(same) -> ELU
    - Each branch ends at temporal length 4; we flatten and concat branches.
    - Final head: MLP -> num_classes.

    Input: (B, 306, 125)
    Output: (B, num_classes)
    """

    def __init__(self,
                 n_channels: int = 306,
                 num_classes: int = 39,
                 model_dim: int = 128,  # channel count after 1x1 proj per scale
                 depth: float = 0.5,
                 width: float = 0.25,
                 dropout: float = 0.5,
                 seq_len: int = 125, **kwargs):
        super().__init__()
        assert seq_len == 125, "This head uses k=50,s=25 assuming L=125. Change seq_len or downsampling if needed."
        self.seq_len = int(seq_len)
        self.model_dim = int(model_dim)

        # YOLO-like backbone (all convs)
        self.backbone = MEGYoloBackbone1D(
            n_channels=n_channels,
            depth=depth,
            width=width,
            use_deconv=False,
        )
        c256, c512, c1024 = self.backbone.out_channels

        # 1x1 projections to a common channel dim
        self.proj3 = nn.Conv1d(c256,  model_dim, kernel_size=1, bias=True)
        self.proj4 = nn.Conv1d(c512,  model_dim, kernel_size=1, bias=True)
        self.proj5 = nn.Conv1d(c1024, model_dim, kernel_size=1, bias=True)

        # Per-branch large-kernel conv stack (mirrors Table10’s k=7 -> k=50,s=25 -> k=7 with ELUs)
        def make_largek_branch():
            return nn.Sequential(
                nn.Conv1d(model_dim, model_dim, kernel_size=7, stride=1, padding=3, bias=True),  # same
                nn.ELU(inplace=True),
                nn.Conv1d(model_dim, model_dim, kernel_size=50, stride=25, padding=0, bias=True), # 125 -> 4
                nn.ELU(inplace=True),
                nn.Conv1d(model_dim, model_dim, kernel_size=7, stride=1, padding=3, bias=True),  # keep 4
                nn.ELU(inplace=True),
            )

        self.branch3 = make_largek_branch()
        self.branch4 = make_largek_branch()
        self.branch5 = make_largek_branch()

        # After branch processing: each has shape (B, model_dim, 4)
        # Flatten and concatenate three scales -> (B, model_dim * 12)
        in_dim = model_dim * 12

        self.head = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    @staticmethod
    def _upsample_to_len(x: torch.Tensor, target_len: int) -> torch.Tensor:
        # x: (B, C, Ls) -> (B, C, target_len)
        if x.shape[-1] == target_len:
            return x
        return F.interpolate(x, size=target_len, mode="linear", align_corners=False)

    def _run_branch(self, feat: torch.Tensor, proj: nn.Module, branch: nn.Module) -> torch.Tensor:
        z = proj(feat)                                  # (B, M, Ls)
        z = self._upsample_to_len(z, self.seq_len)      # (B, M, 125)
        z = branch(z)                                   # (B, M, 4)
        return z.flatten(1)                             # (B, M*4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 306, 125)
        assert x.ndim == 3 and x.shape[-1] == self.seq_len, f"Expected (B, C, {self.seq_len})"
        p3, p4, p5 = self.backbone(x)

        f3 = self._run_branch(p3, self.proj3, self.branch3)
        f4 = self._run_branch(p4, self.proj4, self.branch4)
        f5 = self._run_branch(p5, self.proj5, self.branch5)

        fused = torch.cat([f3, f4, f5], dim=1)          # (B, model_dim*12)
        return self.head(fused)              


# ============================================
# Lightning Module for Training
# ============================================

class MEGYOLOPhonemeClassifier(L.LightningModule):
    """Lightning module for MEG-YOLO phoneme classification."""
    
    def __init__(self,
                 input_dim: int = 306,
                 model_dim: int = 512,
                 depth: float = 0.5,
                 width: float = 0.25,
                 lr: float = 5e-4,
                 weight_decay: float = 0.01,
                 dropout: float = 0.3,
                 label_smoothing: float = 0.1,
                 lstm_layers: int = 2,
                 bidirectional: bool = True,
                 share_lstm: bool = False,
                 pool: str = "mean",
                 use_attention: bool = True,
                 warmup_epochs: int = 2,
                 max_epochs: int = 50,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        # Model
        self.model = MEGYOLOPhonemeModelLargeK(
            n_channels=input_dim,
            model_dim=model_dim,
            num_classes=39,  # Fixed for LibriBrain
            depth=depth,
            width=width,
            lstm_layers=lstm_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            share_lstm=share_lstm,
            pool=pool,
            use_attention=use_attention
        )
        
        # Loss with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # Metrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=39)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=39)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=39, average="macro")
        self.val_prec = torchmetrics.Precision(task="multiclass", num_classes=39, average="macro")
        self.val_rec = torchmetrics.Recall(task="multiclass", num_classes=39, average="macro")
        
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=39)
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=39, average="macro")
        self.test_prec = torchmetrics.Precision(task="multiclass", num_classes=39, average="macro")
        self.test_rec = torchmetrics.Recall(task="multiclass", num_classes=39, average="macro")
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        
        self.val_acc(preds, y)
        self.val_f1(preds, y)
        self.val_prec(preds, y)
        self.val_rec(preds, y)
        
        self.log_dict({
            'val_loss': loss,
            'val_acc': self.val_acc,
            'val_f1_macro': self.val_f1,
            'val_precision': self.val_prec,
            'val_recall': self.val_rec
        }, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        
        self.test_acc(preds, y)
        self.test_f1(preds, y)
        self.test_prec(preds, y)
        self.test_rec(preds, y)
        
        self.log_dict({
            'test_loss': loss,
            'test_acc': self.test_acc,
            'test_f1_macro': self.test_f1,
            'test_precision': self.test_prec,
            'test_recall': self.test_rec
        })
        
        return loss
    
    def configure_optimizers(self):
        # Separate parameter groups with different learning rates
        backbone_params = list(self.model.backbone.parameters())
        # lstm_params = list(self.model.lstm3.parameters()) + \
        #               list(self.model.lstm4.parameters()) + \
        #               list(self.model.lstm5.parameters())
        other_params = [p for p in self.parameters() 
                        if not any(p is bp for bp in backbone_params)]
        
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': self.hparams.lr},
            # {'params': lstm_params, 'lr': self.hparams.lr * 0.5},
            {'params': other_params, 'lr': self.hparams.lr}
        ], weight_decay=self.hparams.weight_decay)
        
        # Cosine annealing with warmup
        def lr_lambda(epoch):
            if epoch < self.hparams.warmup_epochs:
                return epoch / self.hparams.warmup_epochs
            else:
                progress = (epoch - self.hparams.warmup_epochs) / \
                          (self.hparams.max_epochs - self.hparams.warmup_epochs)
                return 0.5 * (1 + math.cos(math.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }


# ============================================
# Simple Baseline Model (for comparison)
# ============================================

class SimplePhonemeClassifier(L.LightningModule):
    """Simple baseline model matching the competition example."""
    
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(306, 128, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16000, 39)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.f1_macro = torchmetrics.F1Score(num_classes=39, average='macro', task="multiclass")
    
    def forward(self, x):
        return self.model(x)
    
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
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.0005)


# ============================================
# Training Script
# ============================================

if __name__ == "__main__":
    # Example usage matching the competition format
    
    # Initialize model with optimal hyperparameters
    model = MEGYOLOPhonemeClassifier(
        input_dim=306,        # MEG channels
        model_dim=512,        # Hidden dimension
        depth=0.5,           # Depth multiplier for YOLO blocks
        width=0.25,          # Width multiplier for channels
        lr=5e-4,
        weight_decay=0.01,
        dropout=0.3,
        label_smoothing=0.1,
        lstm_layers=2,
        bidirectional=True,
        share_lstm=False,
        pool="mean",
        use_attention=True,
        warmup_epochs=2,
        max_epochs=15       # Match competition example
    )
    
    # Test forward pass
    dummy_input = torch.randn(4, 306, 125)  # (batch, channels, time)
    output = model(dummy_input)
    print(f"Model output shape: {output.shape}")  # Should be (4, 39)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # For actual training, use the competition's data loading code:
    """
    from pnpl.datasets import LibriBrainPhoneme, GroupedDataset
    from torch.utils.data import DataLoader
    import lightning as L
    
    # Load datasets
    train_dataset = LibriBrainPhoneme(
        data_path="./libribrain/data/",
        include_run_keys=[("0", str(i), "Sherlock1", "1") for i in range(1, 10)],
        tmin=0.0,
        tmax=0.5
    )
    
    # Apply signal averaging
    averaged_train_dataset = GroupedDataset(train_dataset, grouped_samples=100)
    
    val_dataset = LibriBrainPhoneme(
        data_path="./libribrain/data/",
        partition="validation",
        tmin=0.0,
        tmax=0.5
    )
    
    # Create dataloaders
    train_loader = DataLoader(averaged_train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # Train
    trainer = L.Trainer(
        devices="auto",
        max_epochs=15,
        gradient_clip_val=1.0,
        enable_checkpointing=True,
    )
    
    trainer.fit(model, train_loader, val_loader)
    """