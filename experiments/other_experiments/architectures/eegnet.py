"""
Fixed Integrated PySeizure-based Phoneme Classification Model
Resolves device placement issues for metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np
import math
from torchmetrics import F1Score
from typing import Optional, Dict, List, Tuple, Any

# ============================================
# Fixed Feature Extraction
# ============================================

class ConfigurableFeatureExtractor(nn.Module):
    """Fixed feature extraction from MEG data"""
    
    def __init__(self, 
                 n_channels=306,
                 sampling_rate=250,
                 temporal_features=None,
                 frequency_features=None,
                 correlation_features=None,
                 frequency_bands=None):
        super().__init__()
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        
        # Default feature sets
        self.temporal_features = temporal_features or [
            'mean', 'variance', 'std', 'skewness', 'kurtosis',
            'peak_to_peak', 'zero_crossings', 'hjorth_mobility'
        ]
        
        self.frequency_features = frequency_features or [
            'band_power', 'spectral_centroid', 'peak_frequency'
        ]
        
        self.correlation_features = correlation_features or [
            'mean_correlation', 'max_correlation', 'clustering_coefficient'
        ]
        
        # Default frequency bands
        self.frequency_bands = frequency_bands or {
            'delta': [1, 4],
            'theta': [4, 8],
            'alpha': [8, 12],
            'beta': [12, 30],
            'gamma': [30, 45],
            'high_gamma': [65, 80]
        }
        
    def extract_temporal_features(self, x):
        """Extract selected temporal domain features"""
        B, C, T = x.shape
        features = []
        
        if 'mean' in self.temporal_features:
            features.append(x.mean(dim=-1))
        
        if 'variance' in self.temporal_features:
            features.append(x.var(dim=-1))
        
        if 'std' in self.temporal_features:
            features.append(x.std(dim=-1))
        
        if 'skewness' in self.temporal_features:
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True) + 1e-8
            skewness = ((x - mean) ** 3).mean(dim=-1) / (std.squeeze(-1) ** 3)
            features.append(skewness)
        
        if 'kurtosis' in self.temporal_features:
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True) + 1e-8
            kurtosis = ((x - mean) ** 4).mean(dim=-1) / (std.squeeze(-1) ** 4) - 3
            features.append(kurtosis)
        
        if 'peak_to_peak' in self.temporal_features:
            features.append(x.max(dim=-1)[0] - x.min(dim=-1)[0])
        
        if 'zero_crossings' in self.temporal_features:
            zero_cross = (x[:, :, 1:] * x[:, :, :-1] < 0).float().sum(dim=-1)
            features.append(zero_cross)
        
        if 'hjorth_mobility' in self.temporal_features:
            diff = x[:, :, 1:] - x[:, :, :-1]
            mobility = (diff.var(dim=-1) / (x.var(dim=-1) + 1e-8)).sqrt()
            features.append(mobility)
        
        if 'hjorth_complexity' in self.temporal_features:
            diff1 = x[:, :, 1:] - x[:, :, :-1]
            diff2 = diff1[:, :, 1:] - diff1[:, :, :-1]
            mobility1 = (diff1.var(dim=-1) / (x.var(dim=-1) + 1e-8)).sqrt()
            mobility2 = (diff2.var(dim=-1) / (diff1.var(dim=-1) + 1e-8)).sqrt()
            complexity = mobility2 / (mobility1 + 1e-8)
            features.append(complexity)
        
        if 'min' in self.temporal_features:
            features.append(x.min(dim=-1)[0])
        
        if 'max' in self.temporal_features:
            features.append(x.max(dim=-1)[0])
        
        if 'absolute_area' in self.temporal_features:
            features.append(x.abs().sum(dim=-1))
        
        if 'total_energy' in self.temporal_features:
            features.append((x ** 2).sum(dim=-1))
        
        return torch.stack(features, dim=-1) if features else torch.zeros(B, C, 0, device=x.device)
    
    def extract_frequency_features(self, x):
        """Extract selected frequency domain features - FIXED"""
        B, C, T = x.shape
        features = []
        
        # Compute FFT
        fft = torch.fft.rfft(x, dim=-1)
        power = torch.abs(fft) ** 2
        # Ensure freqs is on the same device as x
        freqs = torch.fft.rfftfreq(T, d=1/self.sampling_rate).to(x.device)
        n_freqs = len(freqs)
        
        if 'band_power' in self.frequency_features:
            for band_name, (low, high) in self.frequency_bands.items():
                mask = (freqs >= low) & (freqs < high)
                if mask.any():
                    band_power = power[:, :, mask].mean(dim=-1)
                    features.append(band_power)
        
        if 'spectral_centroid' in self.frequency_features:
            total_power = power.sum(dim=-1, keepdim=True) + 1e-8
            freqs_expanded = freqs.unsqueeze(0).unsqueeze(0).expand(B, C, -1)
            spectral_centroid = (power * freqs_expanded).sum(dim=-1) / total_power.squeeze(-1)
            features.append(spectral_centroid)
        
        if 'peak_frequency' in self.frequency_features:
            peak_freq_idx = power.argmax(dim=-1)
            # Clamp indices to valid range
            peak_freq_idx = peak_freq_idx.clamp(0, n_freqs - 1)
            # Gather frequencies using advanced indexing to maintain device
            peak_freq = freqs[peak_freq_idx.flatten()].reshape(B, C)
            features.append(peak_freq)
        
        if 'spectral_entropy' in self.frequency_features:
            norm_power = power / (power.sum(dim=-1, keepdim=True) + 1e-8)
            spectral_entropy = -(norm_power * (norm_power + 1e-8).log()).sum(dim=-1)
            features.append(spectral_entropy)
        
        if 'spectral_rolloff' in self.frequency_features:
            cumsum_power = power.cumsum(dim=-1)
            threshold = 0.85 * power.sum(dim=-1, keepdim=True)
            rolloff_idx = (cumsum_power <= threshold).sum(dim=-1)
            # Clamp indices to valid range (0 to n_freqs-1)
            rolloff_idx = rolloff_idx.clamp(0, n_freqs - 1)
            # Gather frequencies using advanced indexing to maintain device
            rolloff_freq = freqs[rolloff_idx.flatten()].reshape(B, C)
            features.append(rolloff_freq)
        
        return torch.stack(features, dim=-1) if features else torch.zeros(B, C, 0, device=x.device)
    
    def extract_correlation_features(self, x):
        """Extract selected inter-channel correlation features"""
        B, C, T = x.shape
        features = []
        
        # Compute correlation matrix
        x_centered = x - x.mean(dim=-1, keepdim=True)
        x_normalized = x_centered / (x_centered.std(dim=-1, keepdim=True) + 1e-8)
        corr_matrix = torch.bmm(x_normalized, x_normalized.transpose(1, 2)) / T
        
        if 'mean_correlation' in self.correlation_features:
            mean_corr = corr_matrix.mean(dim=-1)
            features.append(mean_corr)
        
        if 'max_correlation' in self.correlation_features:
            mask = torch.eye(C, device=x.device).bool()
            corr_matrix_masked = corr_matrix.clone()
            corr_matrix_masked[:, mask] = -1
            max_corr = corr_matrix_masked.max(dim=-1)[0]
            features.append(max_corr)
        
        if 'clustering_coefficient' in self.correlation_features:
            threshold = 0.5
            adj_matrix = (corr_matrix.abs() > threshold).float()
            clustering = torch.zeros_like(adj_matrix[:, :, 0])
            
            # Simplified clustering coefficient calculation
            for i in range(min(C, 10)):  # Limit iterations for efficiency
                neighbors = adj_matrix[:, i] > 0
                n_neighbors = neighbors.sum(dim=-1)
                
                # For each batch element
                for b in range(B):
                    if n_neighbors[b] > 1:
                        neighbor_mask = neighbors[b]
                        neighbor_indices = torch.where(neighbor_mask)[0]
                        if len(neighbor_indices) > 1:
                            # Get submatrix for this batch and channel
                            submatrix = adj_matrix[b, neighbor_indices][:, neighbor_indices]
                            n_connections = (submatrix.sum() - len(neighbor_indices)) / 2
                            n_possible = len(neighbor_indices) * (len(neighbor_indices) - 1) / 2
                            clustering[b, i] = n_connections / (n_possible + 1e-8)
            
            features.append(clustering)
        
        if 'global_efficiency' in self.correlation_features:
            # Simplified global efficiency
            inv_dist = 1 / (1 - corr_matrix.abs() + 1e-8)
            mask = torch.eye(C, device=x.device).bool()
            inv_dist[:, mask] = 0
            # Don't use keepdim to get shape [B], then expand to [B, C]
            global_eff = inv_dist.mean(dim=(1,2)).unsqueeze(1).expand(B, C)
            features.append(global_eff)
        
        return torch.stack(features, dim=-1) if features else torch.zeros(B, C, 0, device=x.device)
    
    def forward(self, x):
        """Extract all configured features"""
        temporal_features = self.extract_temporal_features(x)
        frequency_features = self.extract_frequency_features(x)
        correlation_features = self.extract_correlation_features(x)
        
        # Concatenate all features
        all_features = []
        if temporal_features.shape[-1] > 0:
            all_features.append(temporal_features)
        if frequency_features.shape[-1] > 0:
            all_features.append(frequency_features)
        if correlation_features.shape[-1] > 0:
            all_features.append(correlation_features)
        
        if all_features:
            return torch.cat(all_features, dim=-1)
        else:
            return torch.zeros(x.shape[0], x.shape[1], 0, device=x.device)

# ============================================
# Rest of the components remain the same
# ============================================

class ConfigurableEEGNet(nn.Module):
    """EEGNet architecture component"""
    
    def __init__(self,
                 n_channels=306,
                 time_points=125,
                 n_classes=39,
                 temporal_filters=8,
                 temporal_kernel_size=64,
                 depthwise_multiplier=2,
                 pointwise_filters=16,
                 separable_kernel_size=16,
                 dropout_rate=0.25,
                 pool_sizes=(4, 8),
                 activation='elu'):
        super().__init__()
        
        activation_fn = {
            'elu': nn.ELU(),
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'gelu': nn.GELU()
        }.get(activation.lower(), nn.ELU())
        
        self.temporal_conv = nn.Conv2d(1, temporal_filters, 
                                      (1, temporal_kernel_size), 
                                      padding=(0, temporal_kernel_size//2), 
                                      bias=False)
        self.bn1 = nn.BatchNorm2d(temporal_filters)
        
        depthwise_filters = temporal_filters * depthwise_multiplier
        self.depthwise_conv = nn.Conv2d(temporal_filters, depthwise_filters, 
                                       (n_channels, 1), 
                                       groups=temporal_filters, 
                                       bias=False)
        self.bn2 = nn.BatchNorm2d(depthwise_filters)
        self.activation1 = activation_fn
        self.avg_pool1 = nn.AvgPool2d((1, pool_sizes[0]))
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.separable_conv = nn.Conv2d(depthwise_filters, pointwise_filters, 
                                       (1, separable_kernel_size), 
                                       padding=(0, separable_kernel_size//2), 
                                       bias=False)
        self.pointwise_conv = nn.Conv2d(pointwise_filters, pointwise_filters, 
                                       (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(pointwise_filters)
        self.activation2 = activation_fn
        self.avg_pool2 = nn.AvgPool2d((1, pool_sizes[1]))
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Calculate flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, n_channels, time_points)
            dummy_output = self._forward_features(dummy_input)
            self.flatten_size = dummy_output.shape[1]
        
        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 2),
            nn.Linear(128, n_classes)
        )
    
    def _forward_features(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        x = self.temporal_conv(x)
        x = self.bn1(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.activation1(x)
        x = self.avg_pool1(x)
        x = self.dropout1(x)
        x = self.separable_conv(x)
        x = self.pointwise_conv(x)
        x = self.bn3(x)
        x = self.activation2(x)
        x = self.avg_pool2(x)
        x = self.dropout2(x)
        
        return x.flatten(1)
    
    def forward(self, x):
        features = self._forward_features(x)
        return self.classifier(features)

class ConfigurableConvTransformer(nn.Module):
    """ConvTransformer architecture component"""
    
    def __init__(self,
                 n_channels=306,
                 time_points=125,
                 n_classes=39,
                 conv_channels=(256, 128, 64),
                 conv_kernels=(5, 3, 3),
                 conv_pool_sizes=(2, 2, 1),
                 conv_dropout=0.25,
                 d_model=64,
                 n_heads=4,
                 n_layers=2,
                 feedforward_dim=256,
                 transformer_dropout=0.1,
                 classifier_hidden=(256,),
                 classifier_dropout=0.5):
        super().__init__()
        
        # Build convolutional layers
        conv_layers = []
        in_channels = n_channels
        current_time = time_points
        
        for i, (out_ch, kernel, pool) in enumerate(zip(conv_channels, conv_kernels, conv_pool_sizes)):
            conv_layers.extend([
                nn.Conv1d(in_channels, out_ch, kernel_size=kernel, padding=kernel//2),
                nn.ReLU(),
                nn.BatchNorm1d(out_ch)
            ])
            
            if pool > 1:
                conv_layers.append(nn.MaxPool1d(pool))
                current_time = current_time // pool
            
            if conv_dropout > 0:
                conv_layers.append(nn.Dropout(conv_dropout))
            
            in_channels = out_ch
        
        self.conv_layers = nn.Sequential(*conv_layers)
        self.seq_len = current_time
        
        if conv_channels[-1] != d_model:
            self.projection = nn.Linear(conv_channels[-1], d_model)
        else:
            self.projection = None
        
        self.d_model = d_model
        self.positional_encoding = PositionalEncoding(self.d_model, self.seq_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=feedforward_dim,
            dropout=transformer_dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Build classifier
        classifier_layers = []
        in_features = self.d_model * self.seq_len
        
        for hidden_dim in classifier_hidden:
            classifier_layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(classifier_dropout)
            ])
            in_features = hidden_dim
        
        classifier_layers.append(nn.Linear(in_features, n_classes))
        self.classifier = nn.Sequential(*classifier_layers)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.transpose(1, 2)
        
        if self.projection is not None:
            x = self.projection(x)
        
        x = self.positional_encoding(x)
        x = self.transformer(x)
        x = x.flatten(1)
        
        return self.classifier(x)

class ConfigurableFeatureClassifier(nn.Module):
    """MLP classifier for extracted features"""
    
    def __init__(self,
                 n_features,
                 n_classes=39,
                 hidden_layers=(512, 256, 128),
                 dropout_rates=None,
                 activation='relu',
                 use_batch_norm=True):
        super().__init__()
        
        activation_fn = {
            'relu': nn.ReLU,
            'leaky_relu': nn.LeakyReLU,
            'elu': nn.ELU,
            'gelu': nn.GELU,
            'tanh': nn.Tanh
        }.get(activation.lower(), nn.ReLU)
        
        if dropout_rates is None:
            dropout_rates = [0.5] + [0.3] * (len(hidden_layers) - 1)
        
        layers = []
        in_features = n_features
        
        for i, (hidden_dim, dropout_rate) in enumerate(zip(hidden_layers, dropout_rates)):
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(activation_fn())
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            in_features = hidden_dim
        
        layers.append(nn.Linear(in_features, n_classes))
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.classifier(x)

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# ============================================
# Main Integrated Model - WITH FIX FOR DEVICE ISSUE
# ============================================

class IntegratedPySeizurePhonemeClassifier(L.LightningModule):
    """
    Fully configurable integrated model - FIXED device placement for metrics
    """
    
    def __init__(self,
                 meg_channels=306,
                 time_points=125,
                 vocab_size=39,
                 sampling_rate=250,
                 channel_selection=None,
                 feature_extraction=None,
                 ensemble_config=None,
                 training_config=None,
                 optimizer_config=None,
                 **kwargs):
        
        super().__init__()
        self.save_hyperparameters()
        
        # Process configurations with defaults
        self._process_channel_selection(channel_selection, meg_channels)
        self._process_feature_extraction(feature_extraction, sampling_rate)
        self._process_ensemble_config(ensemble_config)
        self._process_training_config(training_config)
        self._process_optimizer_config(optimizer_config)
        
        # Initialize components
        self._initialize_feature_extractor()
        self._initialize_models()
        self._initialize_metrics()
    
    def _process_channel_selection(self, config, meg_channels):
        """Process channel selection configuration"""
        default_config = {
            'enabled': False,
            'method': 'variance',
            'n_channels': meg_channels,
            'custom_indices': None
        }
        
        self.channel_config = {**default_config, **(config or {})}
        
        if self.channel_config['enabled']:
            if self.channel_config['custom_indices']:
                self.channel_indices = self.channel_config['custom_indices']
            else:
                self.channel_indices = self._select_channels(
                    meg_channels,
                    self.channel_config['n_channels'],
                    self.channel_config['method']
                )
        else:
            self.channel_indices = list(range(meg_channels))
        
        self.n_channels = len(self.channel_indices)
    
    def _process_feature_extraction(self, config, sampling_rate):
        """Process feature extraction configuration"""
        default_config = {
            'enabled': True,
            'temporal_features': ['mean', 'variance', 'std'],
            'frequency_features': ['band_power', 'spectral_centroid'],
            'correlation_features': ['mean_correlation'],
            'frequency_bands': {
                'delta': [1, 4],
                'theta': [4, 8],
                'alpha': [8, 12],
                'beta': [12, 30],
                'gamma': [30, 45]
            }
        }
        
        self.feature_config = {**default_config, **(config or {})}
        self.feature_config['sampling_rate'] = sampling_rate
    
    def _process_ensemble_config(self, config):
        """Process ensemble model configuration"""
        default_config = {
            'use_eegnet': True,
            'use_transformer': False,
            'use_feature_classifier': True,
            'ensemble_weights': {'eegnet': 1.0, 'feature_classifier': 1.0},
            
            'eegnet_config': {
                'temporal_filters': 8,
                'temporal_kernel_size': 64,
                'depthwise_multiplier': 2,
                'pointwise_filters': 16,
                'separable_kernel_size': 16,
                'dropout_rate': 0.25,
                'pool_sizes': [4, 8],
                'activation': 'elu'
            },
            
            'transformer_config': {
                'conv_channels': [256, 128, 64],
                'conv_kernels': [5, 3, 3],
                'conv_pool_sizes': [2, 2, 1],
                'conv_dropout': 0.25,
                'd_model': 64,
                'n_heads': 4,
                'n_layers': 2,
                'feedforward_dim': 256,
                'transformer_dropout': 0.1,
                'classifier_hidden': [256],
                'classifier_dropout': 0.5
            },
            
            'feature_classifier_config': {
                'hidden_layers': [256, 128],
                'dropout_rates': [0.3, 0.2],
                'activation': 'relu',
                'use_batch_norm': True
            }
        }
        
        self.ensemble_config = self._deep_update(default_config, config or {})
    
    def _process_training_config(self, config):
        """Process training configuration"""
        default_config = {
            'label_smoothing': 0.0,
            'auxiliary_loss_weight': 0.1,
            'gradient_clip_val': 1.0
        }
        
        self.training_config = {**default_config, **(config or {})}
    
    def _process_optimizer_config(self, config):
        """Process optimizer configuration"""
        default_config = {
            'optimizer': 'adamw',
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'scheduler': 'cosine_annealing_warm_restarts',
            'scheduler_config': {
                'T_0': 10,
                'T_mult': 2,
                'eta_min': 0.000001
            },
            'differential_lr': {
                'enabled': True,
                'feature_extractor_lr_factor': 0.1
            }
        }
        
        self.optimizer_config = self._deep_update(default_config, config or {})
    
    def _deep_update(self, base_dict, update_dict):
        """Recursively update nested dictionaries"""
        result = base_dict.copy()
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                result[key] = self._deep_update(result[key], value)
            else:
                result[key] = value
        return result
    
    def _select_channels(self, total_channels, n_select, method):
        """Select subset of channels based on specified method"""
        if n_select >= total_channels:
            return list(range(total_channels))
        
        if method == 'variance':
            temporal_channels = []
            for ch in range(50, min(100, total_channels)):
                temporal_channels.append(ch)
            for ch in range(200, min(250, total_channels)):
                temporal_channels.append(ch)
            
            other_channels = [ch for ch in range(total_channels) 
                            if ch not in temporal_channels]
            
            selected = temporal_channels[:n_select]
            if len(selected) < n_select:
                remaining = n_select - len(selected)
                step = max(1, len(other_channels) // remaining)
                for i in range(0, len(other_channels), step):
                    if len(selected) < n_select:
                        selected.append(other_channels[i])
            
            return selected[:n_select]
        
        elif method == 'uniform':
            step = total_channels // n_select
            return list(range(0, total_channels, step))[:n_select]
        
        else:  # random
            import random
            channels = list(range(total_channels))
            random.shuffle(channels)
            return channels[:n_select]
    
    def _initialize_feature_extractor(self):
        """Initialize feature extractor if enabled"""
        if self.feature_config['enabled']:
            self.feature_extractor = ConfigurableFeatureExtractor(
                n_channels=self.n_channels,
                sampling_rate=self.feature_config['sampling_rate'],
                temporal_features=self.feature_config['temporal_features'],
                frequency_features=self.feature_config['frequency_features'],
                correlation_features=self.feature_config['correlation_features'],
                frequency_bands=self.feature_config['frequency_bands']
            )
            
            # Calculate number of features safely
            with torch.no_grad():
                dummy_input = torch.zeros(1, self.n_channels, self.hparams.time_points)
                try:
                    dummy_features = self.feature_extractor(dummy_input)
                    self.n_features = dummy_features.flatten(1).shape[1]
                except Exception as e:
                    print(f"Warning: Feature extraction failed during initialization: {e}")
                    self.n_features = self.n_channels * 10  # Fallback estimate
        else:
            self.feature_extractor = None
            self.n_features = 0
    
    def _initialize_models(self):
        """Initialize ensemble models based on configuration"""
        self.models = nn.ModuleDict()
        self.model_weights = {}
        
        if self.ensemble_config['use_eegnet']:
            config = self.ensemble_config['eegnet_config']
            self.models['eegnet'] = ConfigurableEEGNet(
                n_channels=self.n_channels,
                time_points=self.hparams.time_points,
                n_classes=self.hparams.vocab_size,
                **config
            )
            self.model_weights['eegnet'] = self.ensemble_config['ensemble_weights'].get('eegnet', 1.0)
        
        if self.ensemble_config['use_transformer']:
            config = self.ensemble_config['transformer_config']
            self.models['transformer'] = ConfigurableConvTransformer(
                n_channels=self.n_channels,
                time_points=self.hparams.time_points,
                n_classes=self.hparams.vocab_size,
                **config
            )
            self.model_weights['transformer'] = self.ensemble_config['ensemble_weights'].get('transformer', 1.0)
        
        if self.ensemble_config['use_feature_classifier'] and self.feature_config['enabled'] and self.n_features > 0:
            config = self.ensemble_config['feature_classifier_config']
            self.models['feature_classifier'] = ConfigurableFeatureClassifier(
                n_features=self.n_features,
                n_classes=self.hparams.vocab_size,
                **config
            )
            self.model_weights['feature_classifier'] = self.ensemble_config['ensemble_weights'].get('feature_classifier', 1.0)
        
        # Normalize weights
        if self.model_weights:
            total_weight = sum(self.model_weights.values())
            for k in self.model_weights:
                self.model_weights[k] /= total_weight
    
    def _initialize_metrics(self):
        """Initialize metrics - FIXED to properly register as modules"""
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=self.training_config['label_smoothing']
        )
        
        # Register main F1 metric as a module
        self.f1_macro = F1Score(
            num_classes=self.hparams.vocab_size,
            average='macro',
            task="multiclass"
        )
        
        # Register model-specific metrics as a ModuleDict for proper device handling
        self.model_metrics = nn.ModuleDict({
            name: F1Score(num_classes=self.hparams.vocab_size, average='macro', task="multiclass")
            for name in self.models.keys()
        })
    
    def _forward_full(self, x):
        """Internal forward pass that always returns both ensemble and individual logits"""
        B, C, T = x.shape
        
        # Select channels if needed
        if len(self.channel_indices) < C:
            x_selected = x[:, self.channel_indices, :]
        else:
            x_selected = x
        
        # Collect predictions from each model
        all_logits = {}
        
        if 'eegnet' in self.models:
            all_logits['eegnet'] = self.models['eegnet'](x_selected)
        
        if 'transformer' in self.models:
            all_logits['transformer'] = self.models['transformer'](x_selected)
        
        if 'feature_classifier' in self.models and self.feature_extractor is not None:
            try:
                features = self.feature_extractor(x_selected)
                features_flat = features.flatten(1)
                all_logits['feature_classifier'] = self.models['feature_classifier'](features_flat)
            except Exception as e:
                print(f"Warning: Feature classifier failed: {e}")
        
        # Weighted ensemble
        if len(all_logits) == 1:
            ensemble_logits = list(all_logits.values())[0]
        elif len(all_logits) > 1:
            ensemble_logits = torch.zeros_like(list(all_logits.values())[0])
            for name, logits in all_logits.items():
                weight = self.model_weights.get(name, 1.0)
                ensemble_logits += weight * logits
        else:
            # Fallback: return zeros if no models produced output
            ensemble_logits = torch.zeros(B, self.hparams.vocab_size, device=x.device)
        
        return ensemble_logits, all_logits
    
    def forward(self, x):
        """Forward pass - returns tuple during training, single tensor during inference"""
        ensemble_logits, all_logits = self._forward_full(x)
        
        # Return tuple during training, single tensor during inference
        if self.training:
            return ensemble_logits, all_logits
        else:
            return ensemble_logits
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        # During training, forward returns tuple
        ensemble_logits, individual_logits = self(x)
        
        loss = self.criterion(ensemble_logits, y)
        
        if self.training_config['auxiliary_loss_weight'] > 0:
            for name, logits in individual_logits.items():
                individual_loss = self.criterion(logits, y)
                loss += self.training_config['auxiliary_loss_weight'] * individual_loss
                
                # Metrics are now properly on device
                self.model_metrics[name].update(logits, y)
                self.log(f'train_{name}_f1', self.model_metrics[name], prog_bar=False)
        
        f1_macro = self.f1_macro(ensemble_logits, y)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_f1_macro', f1_macro, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        # Use internal method to get both ensemble and individual logits
        ensemble_logits, individual_logits = self._forward_full(x)
        
        loss = self.criterion(ensemble_logits, y)
        f1_macro = self.f1_macro(ensemble_logits, y)
        
        # Now the metrics are properly on the same device as the model
        for name, logits in individual_logits.items():
            ind_f1 = self.model_metrics[name](logits, y)
            self.log(f'val_{name}_f1', ind_f1, prog_bar=False)
        
        self.log('val_loss', loss)
        self.log('val_f1_macro', f1_macro, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer based on settings"""
        param_groups = []
        
        if self.optimizer_config['differential_lr']['enabled']:
            if self.feature_extractor is not None:
                lr_factor = self.optimizer_config['differential_lr']['feature_extractor_lr_factor']
                param_groups.append({
                    'params': self.feature_extractor.parameters(),
                    'lr': self.optimizer_config['learning_rate'] * lr_factor
                })
            
            for model in self.models.values():
                param_groups.append({
                    'params': model.parameters(),
                    'lr': self.optimizer_config['learning_rate']
                })
        else:
            param_groups = [{
                'params': self.parameters(),
                'lr': self.optimizer_config['learning_rate']
            }]
        
        optimizer_class = {
            'adam': torch.optim.Adam,
            'adamw': torch.optim.AdamW,
            'sgd': torch.optim.SGD,
            'rmsprop': torch.optim.RMSprop
        }.get(self.optimizer_config['optimizer'].lower(), torch.optim.AdamW)
        
        optimizer = optimizer_class(
            param_groups,
            weight_decay=self.optimizer_config['weight_decay']
        )
        
        scheduler_type = self.optimizer_config['scheduler']
        scheduler_config = self.optimizer_config['scheduler_config']
        
        if scheduler_type == 'cosine_annealing_warm_restarts':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, **scheduler_config
            )
        elif scheduler_type == 'cosine_annealing':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, **scheduler_config
            )
        else:
            scheduler = None
        
        if scheduler:
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch'
                }
            }
        else:
            return optimizer
    
    def on_train_epoch_end(self):
        """Reset individual model metrics at epoch end"""
        for metric in self.model_metrics.values():
            metric.reset()