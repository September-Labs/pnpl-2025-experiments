# multiscale_lstm.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import F1Score
import math
from typing import List, Optional, Tuple


class ChannelAttentionSelection(nn.Module):
    """
    Learned channel selection using attention mechanism.
    Selects the most informative channels for phoneme classification.
    """
    def __init__(self, n_channels: int, n_selected: int, hidden_dim: int = 128):
        super().__init__()
        self.n_channels = n_channels
        self.n_selected = n_selected
        
        self.attention = nn.Sequential(
            nn.Linear(n_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_channels),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch, channels, time)
        Returns:
            selected_x: Tensor with selected channels
            attention_weights: Channel attention weights
        """
        # Global average pooling across time
        x_pooled = x.mean(dim=2)  # (batch, channels)
        
        # Compute attention weights
        attention_weights = self.attention(x_pooled)  # (batch, channels)
        
        # Select top-k channels
        _, indices = torch.topk(attention_weights, self.n_selected, dim=1)
        
        # Gather selected channels
        batch_size = x.size(0)
        selected_x = torch.gather(
            x, 1, 
            indices.unsqueeze(-1).expand(batch_size, self.n_selected, x.size(2))
        )
        
        return selected_x, attention_weights


class ScaleWeightingModule(nn.Module):
    """
    Learns optimal weighting of different temporal scales.
    Adapted from the original implementation for multi-class phoneme classification.
    """
    def __init__(self, num_scales: int, embed_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.num_scales = num_scales
        
        self.weight_conv = nn.Sequential(
            nn.Conv1d(embed_dim * num_scales, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(hidden_dim // 2, num_scales, kernel_size=1)
        )
        
    def forward(self, scale_embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            scale_embeddings: List of tensors, each of shape (batch, embed_dim)
        Returns:
            weights: Tensor of shape (batch, num_scales)
        """
        # Concatenate embeddings
        concat_embeds = torch.cat(scale_embeddings, dim=1)  # (batch, embed_dim * num_scales)
        concat_embeds = concat_embeds.unsqueeze(-1)  # Add spatial dimension for Conv1d
        
        # Compute weights
        logits = self.weight_conv(concat_embeds).squeeze(-1)
        weights = F.softmax(logits, dim=1)
        
        return weights


class MultiScaleLSTMPhoneme(L.LightningModule):
    """
    Multi-scale LSTM model for MEG phoneme classification.
    
    Key features:
    - Processes MEG signals at multiple temporal scales
    - Scale-specific bidirectional LSTMs with different capacities
    - Learned scale weighting mechanism
    - Optional channel attention selection
    - Label smoothing for regularization
    """
    
    def __init__(
        self,
        time_points: int = 125,
        n_channels: int = 306,
        n_classes: int = 39,
        learning_rate: float = 0.0003,
        weight_decay: float = 0.01,
        label_smoothing: float = 0.0,
        # Multi-scale parameters
        scales: List[int] = [25, 50, 75, 100, 125],
        hidden_dims: List[int] = [256, 384, 512, 640, 768],
        # LSTM parameters
        num_layers: int = 2,
        dropout: float = 0.4,
        bidirectional: bool = True,
        # Scale weighting
        scale_weight_hidden_dim: int = 256,
        scale_weight_reg: float = 0.1,
        # Channel selection
        use_channel_selection: bool = True,
        n_selected_channels: int = 64,
        channel_selection_method: str = "attention",
        # Initialization
        use_custom_init: bool = True,
        init_method: str = "xavier"
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Store parameters
        self.scales = scales
        self.num_scales = len(scales)
        self.hidden_dims = hidden_dims
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.scale_weight_reg = scale_weight_reg
        self.time_points = time_points
        
        # Input channels (may be reduced by channel selection)
        input_channels = n_channels
        if use_channel_selection and channel_selection_method == "attention":
            self.channel_selector = ChannelAttentionSelection(
                n_channels, n_selected_channels, hidden_dim=128
            )
            input_channels = n_selected_channels
        else:
            self.channel_selector = None
        
        # Shared convolutional feature extractor
        self.shared_conv = nn.Conv1d(input_channels, hidden_dims[0], kernel_size=3, padding=1)
        self.shared_bn = nn.BatchNorm1d(hidden_dims[0])
        self.dropout = nn.Dropout(dropout)
        
        # Scale-specific LSTMs
        self.scale_lstms = nn.ModuleList()
        for i, (scale, hidden_dim) in enumerate(zip(scales, hidden_dims)):
            lstm = nn.LSTM(
                hidden_dims[0], hidden_dim, num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=bidirectional
            )
            self.scale_lstms.append(lstm)
        
        # Reference dimension for scale weighting
        reference_dim = hidden_dims[len(hidden_dims) // 2]
        
        # Scale weighting module
        self.scale_weighter = ScaleWeightingModule(
            num_scales=self.num_scales,
            embed_dim=reference_dim * self.num_directions,
            hidden_dim=scale_weight_hidden_dim
        )
        
        # Scale projections to common dimension
        self.scale_projections = nn.ModuleList()
        for hidden_dim in hidden_dims:
            proj = nn.Linear(
                hidden_dim * self.num_directions,
                reference_dim * self.num_directions
            )
            self.scale_projections.append(proj)
        
        # Final classifier
        classifier_input_dim = reference_dim * self.num_directions
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, classifier_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim // 2, n_classes)
        )
        
        # Loss and metrics
        if label_smoothing > 0:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        self.f1_macro = F1Score(num_classes=n_classes, average='macro', task="multiclass")
        self.f1_weighted = F1Score(num_classes=n_classes, average='weighted', task="multiclass")
        
        # Target scale weights for regularization
        self.register_buffer(
            'target_scale_weights',
            torch.tensor([1.0 / self.num_scales] * self.num_scales)
        )
        
        # Initialize weights
        if use_custom_init:
            self._init_weights(init_method)
    
    def _init_weights(self, method: str = "xavier"):
        """Custom weight initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                if method == "kaiming":
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
            elif isinstance(m, nn.Linear):
                if method == "kaiming":
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name:
                        if method == "orthogonal":
                            nn.init.orthogonal_(param)
                        else:
                            nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
                        # Set forget gate bias to 1
                        n = param.size(0)
                        param[n // 4 : n // 2].data.fill_(1.)
    
    def extract_multiscale_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract features at multiple temporal scales."""
        batch_size, channels, time_dim = x.shape
        multi_scale_data = []
        
        # Center the scales around the middle of the sequence
        center = time_dim // 2
        
        for scale in self.scales:
            if scale > time_dim:
                # If scale is larger than available time, use all time points
                scale_data = x
            else:
                # Extract centered window of the specified scale
                half_scale = scale // 2
                start = max(0, center - half_scale)
                end = min(time_dim, center + half_scale)
                
                scale_data = x[:, :, start:end]
                
                # Pad if necessary to maintain scale size
                if scale_data.size(2) < scale:
                    pad_size = scale - scale_data.size(2)
                    scale_data = F.pad(scale_data, (0, pad_size), mode='constant', value=0)
            
            multi_scale_data.append(scale_data)
        
        return multi_scale_data
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the multi-scale LSTM network.
        
        Args:
            x: Input tensor of shape (batch_size, n_channels, time_points)
        Returns:
            logits: Output tensor of shape (batch_size, n_classes)
        """
        # Optional channel selection
        if self.channel_selector is not None:
            x, channel_weights = self.channel_selector(x)
        
        # Extract multi-scale features
        multi_scale_inputs = self.extract_multiscale_features(x)
        
        scale_embeddings = []
        
        # Process each scale through its dedicated LSTM
        for i, (scale_input, lstm) in enumerate(zip(multi_scale_inputs, self.scale_lstms)):
            # Shared convolutional features
            h = self.shared_conv(scale_input)
            h = self.dropout(self.shared_bn(h))
            
            # Transpose for LSTM (batch, time, features)
            h = h.permute(0, 2, 1)
            
            # LSTM processing
            output, (h_n, _) = lstm(h)
            
            # Extract final hidden states
            if self.bidirectional:
                # Concatenate forward and backward hidden states
                hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
            else:
                hidden = h_n[-1]
            
            # Project to common dimension
            hidden = self.scale_projections[i](hidden)
            scale_embeddings.append(hidden)
        
        # Compute scale weights
        scale_weights = self.scale_weighter(scale_embeddings)
        
        # Weighted combination of scale embeddings
        weighted_embedding = torch.zeros_like(scale_embeddings[0])
        for i, embedding in enumerate(scale_embeddings):
            weighted_embedding += scale_weights[:, i:i+1] * embedding
        
        # Final classification
        logits = self.classifier(self.dropout(weighted_embedding))
        
        # Store scale weights for logging
        self.current_scale_weights = scale_weights
        
        return logits
    
    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute loss including scale weight regularization."""
        # Classification loss
        cls_loss = self.criterion(logits, targets)
        
        # Scale weight regularization
        if hasattr(self, 'current_scale_weights') and self.scale_weight_reg > 0:
            scale_weights_mean = self.current_scale_weights.mean(dim=0)
            scale_reg_loss = F.mse_loss(scale_weights_mean, self.target_scale_weights)
            total_loss = cls_loss + self.scale_weight_reg * scale_reg_loss
        else:
            total_loss = cls_loss
            scale_reg_loss = torch.tensor(0.0)
        
        return total_loss, cls_loss, scale_reg_loss
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        total_loss, cls_loss, scale_reg_loss = self.compute_loss(logits, y)
        
        # Compute metrics
        f1_macro = self.f1_macro(logits, y)
        f1_weighted = self.f1_weighted(logits, y)
        
        # Logging
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('train_cls_loss', cls_loss)
        self.log('train_scale_reg_loss', scale_reg_loss)
        self.log('train_f1_macro', f1_macro, prog_bar=True)
        self.log('train_f1_weighted', f1_weighted)
        
        # Log scale weights
        if hasattr(self, 'current_scale_weights'):
            for i in range(self.num_scales):
                self.log(f'train_scale_{i}_weight', 
                        self.current_scale_weights[:, i].mean())
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        total_loss, cls_loss, scale_reg_loss = self.compute_loss(logits, y)
        
        # Compute metrics
        f1_macro = self.f1_macro(logits, y)
        f1_weighted = self.f1_weighted(logits, y)
        
        # Logging
        self.log('val_loss', total_loss, prog_bar=True)
        self.log('val_cls_loss', cls_loss)
        self.log('val_f1_macro', f1_macro, prog_bar=True)
        self.log('val_f1_weighted', f1_weighted)
        
        # Log scale weights
        if hasattr(self, 'current_scale_weights'):
            for i in range(self.num_scales):
                self.log(f'val_scale_{i}_weight', 
                        self.current_scale_weights[:, i].mean())
        
        return total_loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        total_loss, cls_loss, _ = self.compute_loss(logits, y)
        
        # Compute metrics
        f1_macro = self.f1_macro(logits, y)
        f1_weighted = self.f1_weighted(logits, y)
        
        # Logging
        self.log('test_loss', total_loss)
        self.log('test_f1_macro', f1_macro)
        self.log('test_f1_weighted', f1_weighted)
        
        return total_loss
    
    def configure_optimizers(self):
        """Configure optimizer with weight decay."""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.named_parameters():
            if 'bias' in name or 'bn' in name or 'norm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': self.hparams.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], lr=self.hparams.learning_rate)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50, eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }