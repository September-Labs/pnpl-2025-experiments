"""
MEGNet architecture with Spectral Density Feature Extraction
Modular preprocessing component for MEG Phoneme Classification
Lightning module compatible with config-based training system
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import F1Score, Accuracy
import numpy as np
from scipy import signal
from typing import Optional, Tuple, Dict, Any


class SpectralDensityExtractor(nn.Module):
    """
    Modular spectral density feature extractor for MEG signals.
    Can be easily integrated into any architecture.
    """
    def __init__(
        self,
        sampling_rate: int = 250,
        freq_bands: Optional[Dict[str, Tuple[float, float]]] = None,
        nperseg: Optional[int] = None,
        noverlap: Optional[int] = None,
        use_learnable_filters: bool = False,
        output_mode: str = 'concatenate'  # 'concatenate', 'add', 'multiply', 'separate'
    ):
        """
        Args:
            sampling_rate: Sampling frequency of MEG signals
            freq_bands: Dictionary of frequency bands to extract
            nperseg: Length of each segment for Welch's method
            noverlap: Number of points to overlap between segments
            use_learnable_filters: Whether to use learnable frequency filters
            output_mode: How to combine spectral features with original signal
        """
        super().__init__()
        
        self.sampling_rate = sampling_rate
        self.nperseg = nperseg or sampling_rate // 2
        self.noverlap = noverlap or self.nperseg // 2
        self.output_mode = output_mode
        
        # Default frequency bands for MEG analysis
        if freq_bands is None:
            self.freq_bands = {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 100)
            }
        else:
            self.freq_bands = freq_bands
        
        self.n_bands = len(self.freq_bands)
        
        # Learnable frequency band weights if enabled
        if use_learnable_filters:
            self.band_weights = nn.Parameter(torch.ones(self.n_bands))
            self.channel_weights = nn.Parameter(torch.ones(1, 1, 1))
        else:
            self.register_buffer('band_weights', torch.ones(self.n_bands))
            self.register_buffer('channel_weights', torch.ones(1, 1, 1))
    
    def extract_spectral_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract spectral density features from MEG signals.
        
        Args:
            x: Input tensor of shape (batch, channels, time)
        
        Returns:
            Spectral features tensor
        """
        batch_size, n_channels, n_samples = x.shape
        device = x.device
        
        # Move to CPU for scipy processing
        x_cpu = x.detach().cpu().numpy()
        
        # Initialize output tensor for spectral features
        spectral_features = np.zeros((batch_size, n_channels, self.n_bands, n_samples))
        
        for b in range(batch_size):
            for c in range(n_channels):
                # Compute power spectral density using Welch's method
                frequencies, psd = signal.welch(
                    x_cpu[b, c, :],
                    fs=self.sampling_rate,
                    nperseg=min(self.nperseg, n_samples),
                    noverlap=min(self.noverlap, n_samples - 1) if n_samples > 1 else 0
                )
                
                # Extract power in each frequency band
                for band_idx, (band_name, (low_freq, high_freq)) in enumerate(self.freq_bands.items()):
                    # Find frequency indices
                    freq_mask = (frequencies >= low_freq) & (frequencies <= high_freq)
                    
                    # Compute mean power in this band
                    band_power = np.mean(psd[freq_mask]) if freq_mask.any() else 0
                    
                    # Create time-distributed feature
                    spectral_features[b, c, band_idx, :] = band_power
        
        # Convert back to tensor
        spectral_features = torch.from_numpy(spectral_features).float().to(device)
        
        # Apply learnable weights
        spectral_features = spectral_features * self.band_weights.view(1, 1, -1, 1)
        
        return spectral_features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining original signal with spectral features.
        
        Args:
            x: Input tensor of shape (batch, channels, time)
        
        Returns:
            Combined features based on output_mode
        """
        # Extract spectral features
        spectral_features = self.extract_spectral_features(x)
        
        if self.output_mode == 'concatenate':
            # Reshape spectral features to (batch, channels * n_bands, time)
            batch_size, n_channels, n_bands, n_samples = spectral_features.shape
            spectral_features = spectral_features.permute(0, 2, 1, 3)
            spectral_features = spectral_features.reshape(batch_size, n_channels * n_bands, n_samples)
            
            # Concatenate along channel dimension
            output = torch.cat([x, spectral_features], dim=1)
            
        elif self.output_mode == 'add':
            # Average spectral features across bands and add to signal
            spectral_avg = spectral_features.mean(dim=2)
            output = x + spectral_avg * self.channel_weights
            
        elif self.output_mode == 'multiply':
            # Use spectral features as attention weights
            spectral_attention = torch.sigmoid(spectral_features.mean(dim=2))
            output = x * (1 + spectral_attention)
            
        elif self.output_mode == 'separate':
            # Return both separately (for dual-stream architectures)
            output = (x, spectral_features)
        
        else:
            raise ValueError(f"Unknown output_mode: {self.output_mode}")
        
        return output


class SpectralMEGNetBackbone(nn.Module):
    """
    Enhanced MEGNet with spectral density preprocessing
    """
    def __init__(
        self,
        num_classes=39,
        Chans=306,
        Samples=125,
        dropoutRate=0.5,
        kernLength=25,
        F1=16,
        D=2,
        F2=32,
        norm_rate=0.25,
        # Spectral processing parameters
        use_spectral=True,
        sampling_rate=250,
        freq_bands=None,
        spectral_output_mode='concatenate',
        use_learnable_filters=True
    ):
        super().__init__()
        
        self.use_spectral = use_spectral
        self.F2 = F2
        self.num_classes = num_classes
        
        # Initialize spectral extractor if enabled
        if use_spectral:
            self.spectral_extractor = SpectralDensityExtractor(
                sampling_rate=sampling_rate,
                freq_bands=freq_bands,
                nperseg=sampling_rate // 2,
                noverlap=sampling_rate // 4,
                use_learnable_filters=use_learnable_filters,
                output_mode=spectral_output_mode
            )
            
            # Adjust input channels based on spectral mode
            if spectral_output_mode == 'concatenate':
                n_spectral_bands = len(self.spectral_extractor.freq_bands)
                input_channels = Chans + (Chans * n_spectral_bands)
            else:
                input_channels = Chans
        else:
            input_channels = Chans
        
        # Block 1
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding=(0, kernLength // 2), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1, eps=1e-5, momentum=0.1)
        
        # Depthwise convolution across channels (adjusted for spectral features)
        self.depthwiseConv = nn.Conv2d(F1, F1 * D, (input_channels, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D, eps=1e-5, momentum=0.1)
        self.elu = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropoutRate)
        
        # Block 2
        self.separableConv = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        )
        self.batchnorm3 = nn.BatchNorm2d(F2, eps=1e-5, momentum=0.1)
        self.avgpool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropoutRate)
        
        # Adaptive pooling for fixed size output
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 4))
        
        # Handle separate spectral stream if needed
        if use_spectral and spectral_output_mode == 'separate':
            self.spectral_processor = nn.Sequential(
                nn.Conv2d(1, 16, (Chans, 1)),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 4))
            )
            self.flatten_size = (F2 * 4) + (16 * 4)
        else:
            self.flatten_size = F2 * 4
        
        # Classification head
        self.classify = nn.Linear(self.flatten_size, num_classes)
    
    def forward(self, x):
        # Add channel dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Remove unnecessary channel dimension for spectral processing
        x_squeezed = x.squeeze(1)
        
        # Apply spectral preprocessing if enabled
        if self.use_spectral:
            spectral_output = self.spectral_extractor(x_squeezed)
            
            if isinstance(spectral_output, tuple):
                # Separate processing for dual-stream
                x_main, x_spectral = spectral_output
                x_main = x_main.unsqueeze(1)
                
                # Process spectral features separately
                batch_size = x_spectral.shape[0]
                x_spectral = x_spectral.mean(dim=2, keepdim=True).unsqueeze(1)
                x_spectral = self.spectral_processor(x_spectral)
                x_spectral = x_spectral.view(batch_size, -1)
            else:
                x = spectral_output.unsqueeze(1)
                x_spectral = None
        else:
            x_spectral = None
        
        # Main processing stream
        if self.use_spectral and isinstance(spectral_output, tuple):
            x = x_main
        
        # Block 1
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwiseConv(x)
        x = self.batchnorm2(x)
        x = self.elu(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = self.separableConv(x)
        x = self.batchnorm3(x)
        x = self.elu(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)
        
        # Adaptive pooling
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Combine with spectral stream if separate
        if x_spectral is not None:
            x = torch.cat([x, x_spectral], dim=1)
        
        # Classification
        x = self.classify(x)
        
        return x


class SpectralMEGNet(L.LightningModule):
    """
    Lightning wrapper for SpectralMEGNet model with spectral density preprocessing
    """
    def __init__(
        self,
        time_points=125,
        learning_rate=1e-4,
        weight_decay=1e-5,
        num_classes=39,
        channels=306,
        dropout_rate=0.1,
        kern_length=25,
        F1=16,
        D=2,
        F2=32,
        norm_rate=0.25,
        label_smoothing=0.0,
        # Spectral parameters
        use_spectral=True,
        sampling_rate=250,
        freq_bands=None,
        spectral_output_mode='concatenate',
        use_learnable_filters=True,
        # Scheduler parameters
        use_scheduler='onecycle',
        scheduler_params=None,
        optimizer='adamw',
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize model with spectral preprocessing
        self.model = SpectralMEGNetBackbone(
            num_classes=num_classes,
            Chans=channels,
            Samples=time_points,
            dropoutRate=dropout_rate,
            kernLength=kern_length,
            F1=F1,
            D=D,
            F2=F2,
            norm_rate=norm_rate,
            use_spectral=use_spectral,
            sampling_rate=sampling_rate,
            freq_bands=freq_bands,
            spectral_output_mode=spectral_output_mode,
            use_learnable_filters=use_learnable_filters
        )
        
        # Loss and metrics
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.train_f1 = F1Score(num_classes=num_classes, average='macro', task='multiclass')
        self.val_f1 = F1Score(num_classes=num_classes, average='macro', task='multiclass')
        self.test_f1 = F1Score(num_classes=num_classes, average='macro', task='multiclass')
        self.train_acc = Accuracy(num_classes=num_classes, task='multiclass')
        self.val_acc = Accuracy(num_classes=num_classes, task='multiclass')
        self.test_acc = Accuracy(num_classes=num_classes, task='multiclass')
        
        # For tracking best metrics
        self.best_val_f1 = 0.0
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Calculate metrics
        f1 = self.train_f1(y_hat, y)
        acc = self.train_acc(y_hat, y)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_f1_macro', f1, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Calculate metrics
        f1 = self.val_f1(y_hat, y)
        acc = self.val_acc(y_hat, y)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_f1_macro', f1, prog_bar=True)
        self.log('val_acc', acc)
        
        # Track best F1
        if f1 > self.best_val_f1:
            self.best_val_f1 = f1
            self.log('best_val_f1', self.best_val_f1)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Calculate metrics
        f1 = self.test_f1(y_hat, y)
        acc = self.test_acc(y_hat, y)
        
        # Log metrics
        self.log('test_loss', loss)
        self.log('test_f1_macro', f1)
        self.log('test_acc', acc)
        
        return loss
    
    def configure_optimizers(self):
        # Select optimizer
        if self.hparams.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                betas=(0.9, 0.999)
            )
        elif self.hparams.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                momentum=0.9,
                nesterov=True
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.hparams.optimizer}")
        
        # Configure scheduler
        if self.hparams.use_scheduler == 'none':
            return optimizer
        
        scheduler_params = self.hparams.scheduler_params or {}
        
        if self.hparams.use_scheduler == 'onecycle':
            # OneCycleLR needs total steps
            if hasattr(self.trainer, 'estimated_stepping_batches'):
                total_steps = self.trainer.estimated_stepping_batches
            else:
                # Fallback estimation
                total_steps = 1000 * self.hparams.get('max_epochs', 100)
            
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.learning_rate,
                total_steps=total_steps,
                pct_start=scheduler_params.get('pct_start', 0.1),
                anneal_strategy=scheduler_params.get('anneal_strategy', 'cos')
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
            }
        
        elif self.hparams.use_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_params.get('T_max', 50),
                eta_min=scheduler_params.get('eta_min', 1e-6)
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch'
                }
            }
        
        elif self.hparams.use_scheduler == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=scheduler_params.get('factor', 0.5),
                patience=scheduler_params.get('patience', 5),
                min_lr=scheduler_params.get('min_lr', 1e-7)
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_f1_macro',
                    'interval': 'epoch'
                }
            }
        
        return optimizer
    
    def on_train_epoch_end(self):
        # Reset metrics for next epoch
        self.train_f1.reset()
        self.train_acc.reset()
    
    def on_validation_epoch_end(self):
        # Reset metrics for next epoch
        self.val_f1.reset()
        self.val_acc.reset()
    
    def on_test_epoch_end(self):
        # Reset metrics
        self.test_f1.reset()
        self.test_acc.reset()


# Standalone function to easily add spectral preprocessing to any model
def add_spectral_preprocessing(
    model_class,
    sampling_rate=250,
    freq_bands=None,
    output_mode='concatenate',
    use_learnable_filters=True
):
    """
    Wrapper function to add spectral preprocessing to any model class.
    
    Usage:
        EnhancedModel = add_spectral_preprocessing(OriginalModel)
        model = EnhancedModel(**model_params)
    """
    class SpectralEnhancedModel(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.spectral_extractor = SpectralDensityExtractor(
                sampling_rate=sampling_rate,
                freq_bands=freq_bands,
                output_mode=output_mode,
                use_learnable_filters=use_learnable_filters
            )
            
            # Adjust channels parameter if needed
            if output_mode == 'concatenate' and 'Chans' in kwargs:
                n_bands = len(self.spectral_extractor.freq_bands)
                kwargs['Chans'] = kwargs['Chans'] * (1 + n_bands)
            
            self.base_model = model_class(*args, **kwargs)
        
        def forward(self, x):
            # Apply spectral preprocessing
            x_enhanced = self.spectral_extractor(x)
            
            # Handle separate streams
            if isinstance(x_enhanced, tuple):
                # For models that can handle dual inputs
                return self.base_model(*x_enhanced)
            else:
                return self.base_model(x_enhanced)
    
    return SpectralEnhancedModel

