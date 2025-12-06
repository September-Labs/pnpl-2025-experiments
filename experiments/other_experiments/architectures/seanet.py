import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import F1Score
import typing as tp
import numpy as np
import einops


class ConvLayerNorm(nn.LayerNorm):
    """
    Convolution-friendly LayerNorm that moves channels to last dimensions
    before running the normalization and moves them back to original position right after.
    """
    def __init__(
        self, normalized_shape: tp.Union[int, tp.List[int], torch.Size], **kwargs
    ):
        super().__init__(normalized_shape, **kwargs)

    def forward(self, x):
        x = einops.rearrange(x, "b ... t -> b t ...")
        x = super().forward(x)
        x = einops.rearrange(x, "b t ... -> b ... t")
        return x


class SLSTM(nn.Module):
    """
    LSTM without worrying about the hidden state, nor the layout of the data.
    Expects input as convolutional layout.
    """
    def __init__(self, dimension: int, num_layers: int = 2, skip: bool = True):
        super().__init__()
        self.skip = skip
        self.lstm = nn.LSTM(dimension, dimension, num_layers)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        y, _ = self.lstm(x)
        if self.skip:
            y = y + x
        y = y.permute(1, 2, 0)
        return y


# Import the SConv1d and SConvTranspose1d from the provided architecture
# (These would need to be imported from the conv module in actual usage)
class SConv1d(nn.Module):
    """Simplified version - in practice, import from the conv module"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2)
    
    def forward(self, x):
        return self.conv(x)


class SConvTranspose1d(nn.Module):
    """Simplified version - in practice, import from the conv module"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, **kwargs):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2)
    
    def forward(self, x):
        return self.conv(x)


class SEANetResnetBlock(nn.Module):
    """Residual block from SEANet model."""

    def __init__(
        self,
        dim: int,
        kernel_sizes: tp.List[int] = [3, 1],
        dilations: tp.List[int] = [1, 1],
        activation: str = "ELU",
        activation_params: dict = {"alpha": 1.0},
        norm: str = "weight_norm",
        norm_params: tp.Dict[str, tp.Any] = {},
        causal: bool = False,
        pad_mode: str = "reflect",
        compress: int = 2,
        true_skip: bool = True,
    ):
        super().__init__()
        assert len(kernel_sizes) == len(
            dilations
        ), "Number of kernel sizes should match number of dilations"
        act = getattr(nn, activation)
        hidden = dim // compress
        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            block += [
                act(**activation_params),
                SConv1d(
                    in_chs,
                    out_chs,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    norm=norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    pad_mode=pad_mode,
                ),
            ]
        self.block = nn.Sequential(*block)
        self.shortcut: nn.Module
        if true_skip:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = SConv1d(
                dim,
                dim,
                kernel_size=1,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            )

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class SEANetBrainEncoder(nn.Module):
    """SEANet encoder for brain signals."""

    def __init__(
        self,
        channels: int = 306,
        conv_channels: tp.List[int] = [128, 256, 512, 1024],
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 1,
        ratios: tp.List[int] = [4, 2, 2],
        activation: str = "ELU",
        activation_params: dict = {"alpha": 1.0},
        norm: str = "weight_norm",
        norm_params: tp.Dict[str, tp.Any] = {},
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        causal: bool = False,
        pad_mode: str = "reflect",
        true_skip: bool = False,
        compress: int = 2,
        lstm: int = 2,
    ):
        super().__init__()
        self.channels = channels
        self.dimension = dimension
        self.n_filters = n_filters
        self.ratios = list(reversed(ratios))
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)

        act = getattr(nn, activation)
        model: tp.List[nn.Module] = [
            SConv1d(
                channels,
                conv_channels[0],
                kernel_size,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            )
        ]

        for ch_in, ch_out, ratio in zip(conv_channels, conv_channels[1:], self.ratios):
            for j in range(self.n_residual_layers):
                model += [
                    SEANetResnetBlock(
                        ch_in,
                        kernel_sizes=[residual_kernel_size, 1],
                        dilations=[dilation_base**j, 1],
                        norm=norm,
                        norm_params=norm_params,
                        activation=activation,
                        activation_params=activation_params,
                        causal=causal,
                        pad_mode=pad_mode,
                        compress=compress,
                        true_skip=true_skip,
                    )
                ]
            # Add downsampling layers
            model += [
                act(**activation_params),
                SConv1d(
                    ch_in,
                    ch_out,
                    kernel_size=ratio * 2,
                    stride=ratio,
                    norm=norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    pad_mode=pad_mode,
                ),
            ]

        if lstm:
            model += [SLSTM(conv_channels[-1], num_layers=lstm)]

        model += [
            act(**activation_params),
            SConv1d(
                conv_channels[-1],
                dimension,
                last_kernel_size,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            ),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class SEANetBrainDecoder(nn.Module):
    """SEANet decoder for brain signals."""

    def __init__(
        self,
        channels: int = 306,
        conv_channels: tp.List[int] = [128, 256, 512, 1024],
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 1,
        ratios: tp.List[int] = [4, 2, 2],
        activation: str = "ELU",
        activation_params: dict = {"alpha": 1.0},
        final_activation: tp.Optional[str] = None,
        final_activation_params: tp.Optional[dict] = None,
        norm: str = "weight_norm",
        norm_params: tp.Dict[str, tp.Any] = {},
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        causal: bool = False,
        pad_mode: str = "reflect",
        true_skip: bool = False,
        compress: int = 2,
        lstm: int = 2,
        trim_right_ratio: float = 1.0,
    ):
        super().__init__()
        self.dimension = dimension
        self.channels = channels
        self.n_filters = n_filters
        self.ratios = ratios
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)

        act = getattr(nn, activation)
        model: tp.List[nn.Module] = [
            SConv1d(
                dimension,
                conv_channels[-1],
                kernel_size,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            )
        ]

        if lstm:
            model += [SLSTM(conv_channels[-1], num_layers=lstm)]

        for ch_in, ch_out, ratio in zip(
            reversed(conv_channels), reversed(conv_channels[:-1]), self.ratios
        ):
            model += [
                act(**activation_params),
                SConvTranspose1d(
                    ch_in,
                    ch_out,
                    kernel_size=ratio * 2,
                    stride=ratio,
                    norm=norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    trim_right_ratio=trim_right_ratio,
                ),
            ]

            for j in range(self.n_residual_layers):
                model += [
                    SEANetResnetBlock(
                        ch_out,
                        kernel_sizes=[residual_kernel_size, 1],
                        dilations=[dilation_base**j, 1],
                        activation=activation,
                        activation_params=activation_params,
                        norm=norm,
                        norm_params=norm_params,
                        causal=causal,
                        pad_mode=pad_mode,
                        compress=compress,
                        true_skip=true_skip,
                    )
                ]

        # Add final layers
        model += [
            act(**activation_params),
            SConv1d(
                conv_channels[0],
                channels,
                last_kernel_size,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            ),
        ]
        # Add optional final activation to decoder (eg. tanh)
        if final_activation is not None:
            final_act = getattr(nn, final_activation)
            final_activation_params = final_activation_params or {}
            model += [final_act(**final_activation_params)]
        self.model = nn.Sequential(*model)

    def forward(self, z):
        y = self.model(z)
        return y


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

def t_ratio(x: torch.Tensor, eps: float = 1e-6, mode: str = "replicate") -> torch.Tensor:
    dim=-1
    median = x.median(dim=dim, keepdim=True)[0]
    q75 = x.quantile(0.75, dim=dim, keepdim=True)
    q25 = x.quantile(0.25, dim=dim, keepdim=True)
    iqr = q75 - q25
    y = (x - median) / (iqr + 1e-8)
    return y

class SEANetDualLossModel(L.LightningModule):
    """
    SEANet-based model with dual objectives: phoneme classification and signal reconstruction.
    
    Architecture:
    - SEANet encoder for feature extraction and temporal modeling
    - Classification head for phoneme prediction
    - SEANet decoder for signal reconstruction
    """
    
    def __init__(self, 
                 input_channels=306,
                 input_time_steps=125,  # Store original time dimension
                 encoder_conv_channels=[128, 256, 512, 1024],
                 decoder_conv_channels=None,  # Will default to same as encoder
                 dimension=128,
                 ratios=[4, 2, 2],
                 n_residual_layers=1,
                 lstm_layers=2,
                 activation="ELU",
                 activation_params={"alpha": 1.0},
                 norm="weight_norm",
                 dropout=0.3,
                 learning_rate=0.0005,
                 classification_weight=1.0,
                 reconstruction_weight=0.5,
                 num_classes=39,
                 loss_type = "focal",
                 contrastive_coef = 0.3,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        # Store original input dimensions
        self.input_time_steps = input_time_steps
        
        # Use same conv_channels for decoder if not specified
        if decoder_conv_channels is None:
            decoder_conv_channels = encoder_conv_channels
        
        # SEANet Encoder
        self.encoder = SEANetBrainEncoder(
            channels=input_channels,
            conv_channels=encoder_conv_channels,
            dimension=dimension,
            n_residual_layers=n_residual_layers,
            ratios=ratios,
            activation=activation,
            activation_params=activation_params,
            norm=norm,
            lstm=lstm_layers,
        )
        
        # SEANet Decoder
        self.decoder = SEANetBrainDecoder(
            channels=input_channels,
            conv_channels=decoder_conv_channels,
            dimension=dimension,
            n_residual_layers=n_residual_layers,
            ratios=ratios,
            activation=activation,
            activation_params=activation_params,
            norm=norm,
            lstm=lstm_layers,
        )
        
        # Classification head
        # The encoder outputs (batch, dimension, time), we need to pool over time
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Global average pooling over time dimension
            nn.Flatten(),  # (batch, dimension)
            nn.Linear(dimension, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.classifier = nn.Linear(128, num_classes)
        # Loss functions
        self.classification_criterion = nn.CrossEntropyLoss()
        self.reconstruction_criterion = nn.MSELoss()
        
        # Metrics
        vocab_size=num_classes
        self.train_metric = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
        self.val_metric = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
        self.test_metric = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
        self.metric_name = "f1_macro"        

    def encode(self, x):
        """Encode input through SEANet encoder"""
        # with torch.no_grad():
        #     x = t_ratio(x)
        # x shape: (batch, channels, time)
        encoded = self.encoder(x)  # (batch, dimension, encoded_time)
        return encoded
        
    def classify(self, encoded_features):
        """Classification from encoded features"""
        logits = self.classifier(encoded_features)  # (batch, num_classes)
        return logits
        
    def reconstruct(self, encoded_features):
        """Reconstruct signal from encoded features"""
        reconstructed = self.decoder(encoded_features)  # (batch, channels, time)
        
        # Ensure output matches input time dimension exactly
        current_time = reconstructed.shape[-1]
        target_time = self.input_time_steps
        
        if current_time != target_time:
            if current_time > target_time:
                # Trim excess time points
                reconstructed = reconstructed[..., :target_time]
            else:
                # Interpolate to match target size
                reconstructed = torch.nn.functional.interpolate(
                    reconstructed, 
                    size=target_time, 
                    mode='linear', 
                    align_corners=False
                )
        
        return reconstructed
    
    def train_forward(self, x):
        """Forward pass returning both classification and reconstruction"""
        # Encode
        encoded = self.encode(x)
        
        features = self.projection(encoded)

        # Classification
        classification_logits = self.classify(features)
        
        # Reconstruction
        reconstruction = self.reconstruct(encoded)
        
        return classification_logits, features, reconstruction
    
    def forward(self, x):
        """Forward pass returning both classification and reconstruction"""
        # Encode
        encoded = self.encode(x)
        
        features = self.projection(encoded)

        # Classification
        classification_logits = self.classify(features)
        
        return classification_logits

    def compute_loss(self, batch):
        """Compute combined loss"""
        x, y = batch
        classification_logits, encoded, reconstruction = self.train_forward(x)
        
        if self.hparams.loss_type == 'focal':
            classification_loss = focal_loss_mean_over_present_classes(classification_logits, y)
        else:
            # Classification loss
            classification_loss = self.classification_criterion(classification_logits, y)
        
        if self.hparams.contrastive_coef>0:
            contrastive_loss = supervised_nt_xent(encoded, y)
        else:
            contrastive_loss = 0.0

        # Reconstruction loss
        reconstruction_loss = self.reconstruction_criterion(reconstruction, x)
        
        # Combined loss
        total_loss = (self.hparams.classification_weight * classification_loss + 
                     self.hparams.reconstruction_weight * reconstruction_loss +
                     self.hparams.contrastive_coef * contrastive_loss)
        
        return total_loss, classification_loss, reconstruction_loss, classification_logits
    
    def training_step(self, batch, batch_idx):
        total_loss, cls_loss, recon_loss, y_hat = self.compute_loss(batch)
        _, y = batch
        
        f1_macro = self.train_metric(y_hat, y)
        
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('train_cls_loss', cls_loss)
        self.log('train_recon_loss', recon_loss)
        self.log('train_f1_macro', f1_macro)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        total_loss, cls_loss, recon_loss, y_hat = self.compute_loss(batch)
        _, y = batch
        
        f1_macro = self.val_metric(y_hat, y)
        
        self.log('val_loss', total_loss)
        self.log('val_cls_loss', cls_loss)
        self.log('val_recon_loss', recon_loss)
        self.log('val_f1_macro', f1_macro, prog_bar=True)
        
        return total_loss
    
    def test_step(self, batch, batch_idx):
        total_loss, cls_loss, recon_loss, y_hat = self.compute_loss(batch)
        _, y = batch
        
        f1_macro = self.test_metric(y_hat, y)
        
        self.log('test_loss', total_loss)
        self.log('test_cls_loss', cls_loss)
        self.log('test_recon_loss', recon_loss)
        self.log('test_f1_macro', f1_macro)
        
        return total_loss
    
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