
"""
Enhanced Multi-Scale RNN Architecture with Piecewise Linear Approximation
Adds signal smoothing via piecewise linear approximation before processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import F1Score, Accuracy
import numpy as np
from typing import Optional, List, Dict, Tuple, Iterable
import math
from collections import defaultdict


# ============================================
# Data Augmentation (Enhanced)
# ============================================

class EnhancedMEGAugmentation(nn.Module):
    """Enhanced data augmentation for MEG signals."""
    
    def __init__(self,
                 noise_level: float = 0.05,
                 time_jitter: float = 0.02,
                 channel_dropout: float = 0.1,
                 time_masking: float = 0.1,
                 mixup_alpha: float = 0.2):
        super().__init__()
        self.noise_level = noise_level
        self.time_jitter = time_jitter
        self.channel_dropout = channel_dropout
        self.time_masking = time_masking
        self.mixup_alpha = mixup_alpha
    
    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None, 
                training: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply augmentation to MEG data."""
        if not training:
            return x, targets
        
        B, C, T = x.shape
        device = x.device
        
        # Gaussian noise
        if self.noise_level > 0 and torch.rand(1).item() < 0.5:
            noise = torch.randn_like(x) * self.noise_level
            x = x + noise
        
        # Channel dropout
        if self.channel_dropout > 0 and torch.rand(1).item() < 0.3:
            channel_mask = torch.rand(B, C, 1, device=device) > self.channel_dropout
            x = x * channel_mask.float()
        
        # Time masking
        if self.time_masking > 0 and torch.rand(1).item() < 0.3:
            mask_size = int(T * self.time_masking)
            if mask_size > 0:
                for b in range(B):
                    start_idx = torch.randint(0, T - mask_size + 1, (1,)).item()
                    x[b, :, start_idx:start_idx + mask_size] = 0
        
        # Mixup augmentation
        if self.mixup_alpha > 0 and targets is not None and torch.rand(1).item() < 0.3:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            index = torch.randperm(B, device=device)
            x = lam * x + (1 - lam) * x[index]
            if targets is not None:
                targets_a = targets
                targets_b = targets[index]
                return x, (targets_a, targets_b, lam)
        
        return x, targets

# ============================================
# MEG Conformer Layer with RNN Integration
# ============================================

class MEGConformerRNNLayer(nn.Module):
    """
    Hybrid Conformer-RNN layer combining DeBERTa attention with RNN processing.
    Based on demega's successful Conformer implementation.
    """
    
    def __init__(self, dim: int, rnn_hidden: int = None, num_heads: int = 4, 
                 ff_dim: int = None, kernel_size: int = 5, dropout: float = 0.1,
                 rnn_type: str = "LSTM", norm_type: str = "pre"):
        super().__init__()
        ff_dim = ff_dim or 2 * dim
        rnn_hidden = rnn_hidden or dim
        self.norm_type = norm_type
        
        # Depthwise separable convolution from demega
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, 1)
        )
        
        # Layer norms (pre-norm strategy from demega)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ln3 = nn.LayerNorm(dim)
        self.ln4 = nn.LayerNorm(dim)
        
        # RNN component for temporal modeling
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(dim, rnn_hidden, num_layers=1, 
                              batch_first=True, bidirectional=True)
        else:
            self.rnn = nn.GRU(dim, rnn_hidden, num_layers=1,
                             batch_first=True, bidirectional=True)
        
        # Project RNN output back to dim
        self.rnn_proj = nn.Linear(rnn_hidden * 2, dim)
        
        # FFN with SiLU activation (from demega)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Pre-layer norm implementation from demega
        if self.norm_type == "pre":
            # Convolution module with pre-norm
            res = x
            x_norm = self.ln1(x)
            x_conv = x_norm.transpose(1, 2)
            x_conv = self.conv(x_conv).transpose(1, 2)
            x = res + self.dropout(x_conv)
            
            # RNN module with pre-norm
            res = x
            x_norm = self.ln2(x)
            rnn_out, _ = self.rnn(x_norm)
            rnn_out = self.rnn_proj(rnn_out)
            x = res + self.dropout(rnn_out)
            
            # Feed-forward module with pre-norm
            res = x
            x_norm = self.ln3(x)
            ff_out = self.ffn(x_norm)
            x = res + ff_out
            
        return x


# ============================================
# Import DeBERTa Components from demega
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

# ============================================
# Balanced Phoneme Pretrainer from demega
# ============================================

class BalancedPhonemePretrainer(nn.Module):
    """
    Pre-training module with temperature-based reweighting from demega.
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
                   gamma: float = 2.0, alpha: torch.Tensor = None):
        """Focal loss to focus on hard-to-classify phonemes."""
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** gamma * ce_loss
        
        if alpha is not None:
            focal_loss = alpha[targets] * focal_loss
        
        return focal_loss.mean()

# ============================================
# Piecewise Linear Approximation Module
# ============================================

class PiecewiseLinearApproximation(nn.Module):
    """
    Performs piecewise linear approximation on MEG signals to reduce noise/spikes.
    Uses either fixed segments or adaptive breakpoints.
    """
    
    def __init__(self,
                 num_segments: int = 10,
                 method: str = "adaptive",  # "fixed", "adaptive", "ramer_douglas_peucker"
                 epsilon: float = 0.01,  # For Ramer-Douglas-Peucker algorithm
                 learnable: bool = False,  # Whether to learn optimal breakpoints
                 smooth_transitions: bool = True):
        super().__init__()
        self.num_segments = num_segments
        self.method = method
        self.epsilon = epsilon
        self.smooth_transitions = smooth_transitions
        
        if learnable and method == "adaptive":
            # Learnable breakpoint positions (as fractions of sequence length)
            self.breakpoint_logits = nn.Parameter(torch.randn(num_segments - 1))
        else:
            self.breakpoint_logits = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply piecewise linear approximation to input signals.
        x: (B, C, T) for MEG data or (B, T, C) for sequential data
        """
        if x.dim() == 3 and x.shape[1] > x.shape[2]:  # Likely (B, C, T) format
            return self._approximate_channels_last(x)
        else:  # (B, T, C) format
            return self._approximate_time_last(x)
    
    def _approximate_channels_last(self, x: torch.Tensor) -> torch.Tensor:
        """Process (B, C, T) format."""
        B, C, T = x.shape
        device = x.device
        
        if self.method == "fixed":
            # Fixed-width segments
            segment_size = T // self.num_segments
            approximated = torch.zeros_like(x)
            
            for seg in range(self.num_segments):
                start = seg * segment_size
                end = min((seg + 1) * segment_size, T)
                if start >= T:
                    break
                    
                # Fit linear approximation for each segment
                segment_data = x[:, :, start:end]
                approximated[:, :, start:end] = self._fit_linear_segment(
                    segment_data, start, end, T
                )
            
            # Handle remainder if T is not divisible by num_segments
            if end < T:
                segment_data = x[:, :, end:]
                approximated[:, :, end:] = self._fit_linear_segment(
                    segment_data, end, T, T
                )
        
        elif self.method == "adaptive":
            approximated = self._adaptive_approximation(x)
        
        elif self.method == "ramer_douglas_peucker":
            approximated = self._rdp_approximation(x)
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return approximated
    
    def _approximate_time_last(self, x: torch.Tensor) -> torch.Tensor:
        """Process (B, T, C) format."""
        # Transpose to (B, C, T), process, then transpose back
        x_transposed = x.transpose(1, 2)
        approximated = self._approximate_channels_last(x_transposed)
        return approximated.transpose(1, 2)
    
    def _fit_linear_segment(self, segment: torch.Tensor, start: int, end: int, T: int) -> torch.Tensor:
        """
        Fit a linear approximation to a segment using least squares.
        """
        B, C, seg_len = segment.shape
        
        if seg_len <= 1:
            return segment
        
        # Create time indices
        t = torch.arange(seg_len, dtype=torch.float32, device=segment.device)
        t_normalized = t / (seg_len - 1) if seg_len > 1 else t
        
        # Compute linear fit: y = a*t + b
        # Using least squares solution
        t_mean = t_normalized.mean()
        y_mean = segment.mean(dim=2, keepdim=True)
        
        # Compute slope (a) and intercept (b)
        numerator = ((t_normalized.unsqueeze(0).unsqueeze(0) - t_mean) * 
                    (segment - y_mean)).sum(dim=2)
        denominator = ((t_normalized - t_mean) ** 2).sum() + 1e-8
        
        a = numerator / denominator  # (B, C)
        b = y_mean.squeeze(2) - a * t_mean  # (B, C)
        
        # Generate approximated values
        approximated = a.unsqueeze(2) * t_normalized.unsqueeze(0).unsqueeze(0) + b.unsqueeze(2)
        
        if self.smooth_transitions and start > 0:
            # Smooth transition at segment boundaries
            blend_width = min(5, seg_len // 4)
            blend_weights = torch.linspace(0, 1, blend_width, device=segment.device)
            blend_weights = blend_weights.unsqueeze(0).unsqueeze(0)
            
            approximated[:, :, :blend_width] = (
                (1 - blend_weights) * segment[:, :, :blend_width] +
                blend_weights * approximated[:, :, :blend_width]
            )
        
        return approximated
    
    def _adaptive_approximation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Use adaptive breakpoints based on signal characteristics.
        """
        B, C, T = x.shape
        
        if self.breakpoint_logits is not None:
            # Use learnable breakpoints
            breakpoints = torch.sigmoid(self.breakpoint_logits).sort()[0]
            breakpoints = (breakpoints * T).long()
            breakpoints = torch.cat([
                torch.tensor([0], device=x.device),
                breakpoints,
                torch.tensor([T], device=x.device)
            ])
        else:
            # Find breakpoints based on signal variance
            breakpoints = self._find_variance_breakpoints(x)
        
        approximated = torch.zeros_like(x)
        
        for i in range(len(breakpoints) - 1):
            start = breakpoints[i].item()
            end = breakpoints[i + 1].item()
            if start >= end:
                continue
            
            segment_data = x[:, :, start:end]
            approximated[:, :, start:end] = self._fit_linear_segment(
                segment_data, start, end, T
            )
        
        return approximated
    
    def _find_variance_breakpoints(self, x: torch.Tensor) -> torch.Tensor:
        """
        Find breakpoints based on local signal variance.
        High variance regions get more segments.
        """
        B, C, T = x.shape
        
        # Compute local variance using sliding window
        window_size = max(3, T // (self.num_segments * 2))
        padding = window_size // 2
        
        # Pad signal
        x_padded = F.pad(x, (padding, padding), mode='reflect')
        
        # Compute local variance
        local_var = torch.zeros(B, C, T, device=x.device)
        for i in range(T):
            window = x_padded[:, :, i:i+window_size]
            local_var[:, :, i] = window.var(dim=2)
        
        # Average across batch and channels
        avg_var = local_var.mean(dim=[0, 1])
        
        # Find peaks in variance as potential breakpoints
        var_grad = torch.gradient(avg_var)[0]
        potential_breaks = torch.where(
            (var_grad[:-1] * var_grad[1:]) < 0  # Sign changes
        )[0] + 1
        
        # Select top num_segments-1 breakpoints
        if len(potential_breaks) >= self.num_segments - 1:
            _, indices = avg_var[potential_breaks].topk(self.num_segments - 1)
            breakpoints = potential_breaks[indices].sort()[0]
        else:
            # Fall back to uniform spacing
            breakpoints = torch.linspace(0, T, self.num_segments + 1, device=x.device).long()
            breakpoints = breakpoints[1:-1]
        
        # Add start and end points
        breakpoints = torch.cat([
            torch.tensor([0], device=x.device),
            breakpoints,
            torch.tensor([T], device=x.device)
        ])
        
        return breakpoints
    
    def _rdp_approximation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ramer-Douglas-Peucker algorithm for curve simplification.
        """
        B, C, T = x.shape
        approximated = torch.zeros_like(x)
        
        for b in range(B):
            for c in range(C):
                signal = x[b, c, :]
                # Get simplified points using RDP
                keep_indices = self._rdp_simplify(signal.cpu().numpy(), self.epsilon)
                keep_indices = torch.tensor(keep_indices, device=x.device)
                
                # Interpolate between kept points
                for i in range(len(keep_indices) - 1):
                    start_idx = keep_indices[i]
                    end_idx = keep_indices[i + 1]
                    
                    # Linear interpolation
                    num_points = end_idx - start_idx + 1
                    if num_points > 1:
                        interp_vals = torch.linspace(
                            signal[start_idx], 
                            signal[end_idx], 
                            num_points,
                            device=x.device
                        )
                        approximated[b, c, start_idx:end_idx+1] = interp_vals
                    else:
                        approximated[b, c, start_idx] = signal[start_idx]
        
        return approximated
    
    def _rdp_simplify(self, points: np.ndarray, epsilon: float) -> List[int]:
        """
        Ramer-Douglas-Peucker algorithm implementation.
        Returns indices of points to keep.
        """
        if len(points) < 3:
            return list(range(len(points)))
        
        # Find point with maximum distance from line between first and last
        dmax = 0
        index = 0
        
        for i in range(1, len(points) - 1):
            d = self._perpendicular_distance(
                i, points[i], 0, points[0], 
                len(points) - 1, points[-1]
            )
            if d > dmax:
                index = i
                dmax = d
        
        # If max distance is greater than epsilon, recursively simplify
        if dmax > epsilon:
            # Recursive call
            rec_results1 = self._rdp_simplify(points[:index+1], epsilon)
            rec_results2 = self._rdp_simplify(points[index:], epsilon)
            
            # Combine results
            result = rec_results1[:-1] + [i + index for i in rec_results2]
        else:
            # Return just the endpoints
            result = [0, len(points) - 1]
        
        return result
    
    def _perpendicular_distance(self, idx: int, point: float, 
                                idx1: int, point1: float, 
                                idx2: int, point2: float) -> float:
        """Calculate perpendicular distance from point to line."""
        if idx2 == idx1:
            return abs(point - point1)
        
        # Line equation: y = mx + b
        m = (point2 - point1) / (idx2 - idx1)
        b = point1 - m * idx1
        
        # Distance from point to line
        distance = abs(m * idx - point + b) / math.sqrt(m**2 + 1)
        return distance


# ============================================
# Signal Quality Enhancement Module
# ============================================

class SignalQualityEnhancer(nn.Module):
    """
    Combines multiple signal processing techniques to clean MEG data.
    """
    
    def __init__(self,
                 use_pla: bool = True,
                 pla_segments: int = 15,
                 pla_method: str = "adaptive",
                 use_wavelet_denoising: bool = False,
                 use_savitzky_golay: bool = False,
                 sg_window: int = 7,
                 sg_polyorder: int = 3):
        super().__init__()
        
        self.use_pla = use_pla
        self.use_wavelet_denoising = use_wavelet_denoising
        self.use_savitzky_golay = use_savitzky_golay
        
        if use_pla:
            self.pla = PiecewiseLinearApproximation(
                num_segments=pla_segments,
                method=pla_method,
                smooth_transitions=True
            )
        
        if use_savitzky_golay:
            # Create Savitzky-Golay filter coefficients
            self.sg_window = sg_window
            self.sg_polyorder = sg_polyorder
            self._init_sg_filter()
    
    def _init_sg_filter(self):
        """Initialize Savitzky-Golay filter coefficients."""
        from scipy.signal import savgol_coeffs
        self.sg_coeffs = torch.tensor(
            savgol_coeffs(self.sg_window, self.sg_polyorder),
            dtype=torch.float32
        )
    
    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Apply signal quality enhancement.
        x: (B, C, T) MEG data
        """
        # Store original for potential residual connections
        x_original = x.clone()
        
        # Apply piecewise linear approximation
        if self.use_pla:
            x = self.pla(x)
        
        # Apply Savitzky-Golay filtering
        if self.use_savitzky_golay:
            x = self._apply_sg_filter(x)
        
        # Optionally blend with original (helps preserve fine details)
        if training:
            # During training, randomly blend to improve robustness
            blend_factor = torch.rand(1, device=x.device) * 0.3  # 0-30% original
            x = (1 - blend_factor) * x + blend_factor * x_original
        
        return x
    
    def _apply_sg_filter(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Savitzky-Golay filter for additional smoothing."""
        B, C, T = x.shape
        
        # Apply 1D convolution with SG coefficients
        x_padded = F.pad(x, (self.sg_window//2, self.sg_window//2), mode='reflect')
        
        # Create conv weights
        weight = self.sg_coeffs.view(1, 1, -1).to(x.device)
        weight = weight.repeat(C, 1, 1)  # (C, 1, window)
        
        # Apply filter
        x_filtered = F.conv1d(x_padded, weight, groups=C)
        
        return x_filtered


# ============================================
# Enhanced Multi-Scale Temporal Processing
# ============================================

class EnhancedMultiScaleEncoder(nn.Module):
    """
    Multi-scale temporal encoder using Conformer-RNN hybrid layers.
    """
    
    def __init__(self,
                 input_dim: int,
                 temporal_scales: List[int],
                 scale_hidden_dims: List[int],
                 num_conformer_layers: int = 2,
                 dropout: float = 0.1,
                 norm_type: str = "pre"):
        super().__init__()
        assert len(temporal_scales) == len(scale_hidden_dims)
        
        self.temporal_scales = temporal_scales
        self.scale_hidden_dims = scale_hidden_dims
        self.num_scales = len(temporal_scales)
        
        # Input projection for each scale
        self.scale_projections = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim)
            for hidden_dim in scale_hidden_dims
        ])
        
        # Conformer-RNN layers for each scale
        self.scale_encoders = nn.ModuleList()
        for scale, hidden_dim in zip(temporal_scales, scale_hidden_dims):
            layers = nn.ModuleList([
                MEGConformerRNNLayer(
                    dim=hidden_dim,
                    rnn_hidden=hidden_dim // 2,
                    num_heads=4,
                    ff_dim=hidden_dim * 2,
                    kernel_size=5,
                    dropout=dropout,
                    norm_type=norm_type
                )
                for _ in range(num_conformer_layers)
            ])
            self.scale_encoders.append(layers)
        
        # Output dimension
        self.output_dim = sum(scale_hidden_dims)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Process input at multiple scales with Conformer-RNN layers.
        x: (batch, time, features)
        """
        B, T, F = x.shape
        scale_outputs = []
        
        for idx, (scale, projection, encoder_layers) in enumerate(
            zip(self.temporal_scales, self.scale_projections, self.scale_encoders)
        ):
            # Project input to scale-specific dimension
            x_scale = projection(x)  # (B, T, hidden_dim)
            
            # Apply Conformer-RNN layers
            for layer in encoder_layers:
                x_scale = layer(x_scale)
            
            # Pool based on scale
            if scale < T:
                # Adaptive pooling based on scale
                stride = max(1, scale // 4)
                pooled = x_scale[:, ::stride, :]
                scale_out = pooled.mean(dim=1)
            else:
                scale_out = x_scale.mean(dim=1)
            
            scale_outputs.append(scale_out)
        
        return scale_outputs

# ============================================
# Updated Main Classifier with PLA
# ============================================

class EnhancedMultiScaleRNNClassifier(L.LightningModule):
    """
    Enhanced Multi-scale RNN classifier with piecewise linear approximation.
    """
    
    def __init__(self,
                 # Core dimensions
                 meg_channels: int = 306,
                 time_points: int = 125,
                 vocab_size: int = 39,
                 hidden_dim: int = 64,
                 
                 # Signal processing parameters
                 use_signal_processing: bool = True,
                 pla_segments: int = 15,
                 pla_method: str = "adaptive",  # "fixed", "adaptive", "ramer_douglas_peucker"
                 use_savitzky_golay: bool = True,
                 sg_window: int = 7,
                 sg_polyorder: int = 3,
                 learnable_pla: bool = False,
                 
                 # Multi-scale parameters
                 temporal_scales: List[int] = [8, 16, 32, 64],
                 scale_hidden_dims: List[int] = [32, 48, 64, 80],
                 num_conformer_layers: int = 2,
                 
                 use_conformer: bool = True,
                 norm_type: str = "pre",
                 learning_rate: float = 0.0001,
                 classifier_lr_multiplier: float = 2.0,
                 loss_type: str = "focal",
                 focal_gamma: float = 2.0,
                 temperature: float = 2.0,
                 dropout_rate: float = 0.1,
                 label_smoothing: float = 0.1,
                 weight_decay: float = 0.01,
                 warmup_epochs: int = 5,
                 total_epochs: int = 100,
                 use_augmentation: bool = True,
                 augmentation_params: Optional[Dict] = None,
                 scale_weighting: str = "attention",
                 scale_attention_dim: int = 64,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        # Signal Quality Enhancement (NEW)
        if use_signal_processing:
            self.signal_enhancer = SignalQualityEnhancer(
                use_pla=True,
                pla_segments=pla_segments,
                pla_method=pla_method,
                use_savitzky_golay=use_savitzky_golay,
                sg_window=sg_window,
                sg_polyorder=sg_polyorder
            )
            
            # Optional: learnable PLA parameters
            if learnable_pla and pla_method == "adaptive":
                self.signal_enhancer.pla.breakpoint_logits = nn.Parameter(
                    torch.randn(pla_segments - 1)
                )
        else:
            self.signal_enhancer = None
        
        self.use_signal_processing = use_signal_processing
        
        # [Rest of initialization remains the same...]
        # Balanced pre-trainer from demega
        self.pretrainer = BalancedPhonemePretrainer(vocab_size, hidden_dim, temperature)
        
        # Data augmentation
        if use_augmentation:
            aug_params = augmentation_params or {}
            self.augmentation = EnhancedMEGAugmentation(
                noise_level=aug_params.get('noise_level', 0.05),
                time_jitter=aug_params.get('time_jitter', 0.02),
                channel_dropout=aug_params.get('channel_dropout', 0.1),
                time_masking=aug_params.get('time_masking', 0.1),
                mixup_alpha=aug_params.get('mixup_alpha', 0.2)
            )
        else:
            self.augmentation = None
        
        # [Continue with rest of initialization...]
        if use_conformer:
            self.input_projection = nn.Sequential(
                nn.Conv1d(meg_channels, hidden_dim, kernel_size=5, padding=2),
                nn.SiLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
            )
            self.input_skip = nn.Conv1d(meg_channels, hidden_dim, kernel_size=1)
            self.multi_scale_encoder = EnhancedMultiScaleEncoder(
                input_dim=hidden_dim,
                temporal_scales=temporal_scales,
                scale_hidden_dims=scale_hidden_dims,
                num_conformer_layers=num_conformer_layers,
                dropout=dropout_rate,
                norm_type=norm_type
            )
            encoder_output_dim = sum(scale_hidden_dims)
        else:
            self.input_projection = None
            self.input_skip = None
            self.multi_scale_encoder = None
            encoder_output_dim = meg_channels
        
        self.use_conformer = use_conformer
        
        # Scale weighting mechanism
        if scale_weighting == "attention":
            self.scale_attention = nn.MultiheadAttention(
                encoder_output_dim, num_heads=8, batch_first=True
            )
            self.scale_weight_proj = nn.Linear(encoder_output_dim, len(temporal_scales))
        else:
            self.scale_weights = nn.Parameter(torch.ones(len(temporal_scales)) / len(temporal_scales))
        
        self.scale_weighting_type = scale_weighting
        self.feature_norm = nn.LayerNorm(encoder_output_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(encoder_output_dim, 128),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, vocab_size)
        )
        
        # [Rest remains the same...]
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.classifier_lr_multiplier = classifier_lr_multiplier
        
        self.train_f1 = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
        self.val_f1 = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
        self.test_f1 = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
        
        self.phoneme_f1_scores = defaultdict(float)
        self.phoneme_counts = defaultdict(int)
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features with signal enhancement."""
        B, C, T = x.shape
        
        # Apply signal quality enhancement (NEW)
        if self.use_signal_processing and self.signal_enhancer is not None:
            x = self.signal_enhancer(x, training=self.training)
        
        # [Rest of feature extraction remains the same...]
        if self.use_conformer:
            features_main = self.input_projection(x)
            features_skip = self.input_skip(x)
            features = features_main + features_skip
            
            features = features.transpose(1, 2)
            scale_outputs = self.multi_scale_encoder(features)
            combined = torch.cat(scale_outputs, dim=-1)
            
            if self.scale_weighting_type == "attention":
                combined_exp = combined.unsqueeze(1)
                attended, _ = self.scale_attention(combined_exp, combined_exp, combined_exp)
                scale_weights = F.softmax(self.scale_weight_proj(attended.squeeze(1)), dim=-1)
                
                weighted_features = []
                start_idx = 0
                for i, scale_out in enumerate(scale_outputs):
                    weight = scale_weights[:, i:i+1]
                    weighted_features.append(scale_out * weight)
                
                features = torch.cat(weighted_features, dim=-1)
            else:
                features = combined
            
            features = self.feature_norm(features)
        else:
            features = x.mean(dim=2)
        
        return features
    
    # [All other methods remain exactly the same as before...]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits
    
    def compute_loss(self, logits, targets, mixup_targets=None):
        if mixup_targets is not None:
            targets_a, targets_b, lam = mixup_targets
            loss_a = self.compute_single_loss(logits, targets_a)
            loss_b = self.compute_single_loss(logits, targets_b)
            return lam * loss_a + (1 - lam) * loss_b
        else:
            return self.compute_single_loss(logits, targets)
    
    def compute_single_loss(self, logits, targets):
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
                
        else:
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
        
        mixup_targets = None
        if self.augmentation:
            x_aug, aug_targets = self.augmentation(x, y, training=True)
            x = x_aug
            if aug_targets is not None and isinstance(aug_targets, tuple) and len(aug_targets) == 3:
                mixup_targets = aug_targets
        
        logits = self(x)
        loss = self.compute_loss(logits, y, mixup_targets)
        
        if mixup_targets is None:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                acc = (preds == y).float().mean()
                f1 = self.train_f1(logits, y)
                
                for i in range(len(y)):
                    phoneme_id = y[i].item()
                    self.phoneme_counts[phoneme_id] += 1
                    if preds[i] == y[i]:
                        self.phoneme_f1_scores[phoneme_id] += 1
        else:
            acc = 0.0
            f1 = 0.0
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_f1_macro', f1, prog_bar=True)
        self.log('train_acc', acc)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        logits = self(x)
        loss = self.compute_loss(logits, y)
        
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()
        f1 = self.val_f1(logits, y)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_f1_macro', f1, prog_bar=True)
        self.log('val_acc', acc)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        
        logits = self(x)
        loss = self.compute_loss(logits, y)
        
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()
        f1 = self.test_f1(logits, y)
        
        self.log('test_loss', loss)
        self.log('test_f1_macro', f1)
        self.log('test_acc', acc)
        
        return loss
    
    def configure_optimizers(self):
        classifier_params = list(self.classifier.parameters())
        classifier_param_ids = {id(p) for p in classifier_params}
        other_params = [p for p in self.parameters() if id(p) not in classifier_param_ids]
        
        optimizer = torch.optim.AdamW([
            {'params': other_params, 'lr': self.learning_rate},
            {'params': classifier_params, 'lr': self.learning_rate * self.classifier_lr_multiplier}
        ], weight_decay=self.weight_decay)
        
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return float(epoch) / float(max(1, self.warmup_epochs))
            else:
                progress = float(epoch - self.warmup_epochs) / float(max(1, self.total_epochs - self.warmup_epochs))
                return 0.5 * (1.0 + math.cos(math.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }
