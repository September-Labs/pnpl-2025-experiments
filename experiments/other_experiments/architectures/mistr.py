"""
MiSTR-adapted architecture for MEG-based Phoneme Classification
Integrates wavelet-based encoding, neural compression, and temporal attention
from the MiSTR speech synthesis model, adapted for discriminative phoneme classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np
from torchmetrics import F1Score
import pywt
from typing import Optional, Dict, Tuple

# ============================================
# Wavelet-based Neural Signal Encoder
# ============================================

class WaveletMEGEncoder(nn.Module):
    """
    Wavelet-based feature extraction for MEG signals.
    Adapted from MiSTR's neural_signal_encoder for MEG characteristics.
    """
    
    def __init__(self, 
                 meg_channels: int = 306,
                 time_points: int = 125,
                 wavelet: str = 'db4',
                 decomposition_level: int = 4,
                 output_dim: int = 256):
        super().__init__()
        self.meg_channels = meg_channels
        self.time_points = time_points
        self.wavelet = wavelet
        self.decomposition_level = decomposition_level
        
        # Calculate wavelet feature dimension
        # Each decomposition level produces detail coefficients
        # The size depends on the wavelet and input length
        self.wavelet_features_per_channel = decomposition_level * 2  # Simplified estimation
        
        # Projection layers for wavelet features
        self.wavelet_projection = nn.Sequential(
            nn.Linear(meg_channels * self.wavelet_features_per_channel, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim)
        )
        
        # High-gamma feature extraction (70-170 Hz)
        self.high_gamma_conv = nn.Conv1d(meg_channels, 64, kernel_size=5, padding=2)
        self.high_gamma_pool = nn.AdaptiveAvgPool1d(8)
        
        # Low-frequency feature extraction (4-30 Hz)
        self.low_freq_conv = nn.Conv1d(meg_channels, 32, kernel_size=11, padding=5)
        self.low_freq_pool = nn.AdaptiveAvgPool1d(8)
        
        # Cross-frequency coupling module
        self.cfc_attention = nn.MultiheadAttention(
            embed_dim=96,  # 64 + 32
            num_heads=4,
            batch_first=True
        )
        
        # Combine all features
        self.feature_fusion = nn.Sequential(
            nn.Linear(output_dim + 96 * 8, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(output_dim * 2, output_dim)
        )
    
    def extract_wavelet_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract wavelet coefficients from MEG signals.
        
        Args:
            x: (B, channels, time_points)
        Returns:
            (B, channels * wavelet_features)
        """
        B, C, T = x.shape
        wavelet_features = []
        
        # Process each sample in the batch
        for b in range(B):
            channel_features = []
            for c in range(C):
                signal = x[b, c, :].cpu().numpy()
                
                try:
                    # Perform wavelet decomposition
                    coeffs = pywt.wavedec(signal, wavelet=self.wavelet, level=self.decomposition_level)
                    
                    # Extract energy from detail coefficients
                    energies = []
                    for level_coeffs in coeffs[1:]:  # Skip approximation coefficients
                        energy = np.sum(np.abs(level_coeffs) ** 2)
                        energies.append(energy)
                        # Also include max amplitude
                        max_amp = np.max(np.abs(level_coeffs))
                        energies.append(max_amp)
                    
                    channel_features.extend(energies[:self.wavelet_features_per_channel])
                except:
                    # Fallback if wavelet decomposition fails
                    channel_features.extend([0.0] * self.wavelet_features_per_channel)
            
            # Pad if necessary
            while len(channel_features) < C * self.wavelet_features_per_channel:
                channel_features.append(0.0)
            
            wavelet_features.append(channel_features[:C * self.wavelet_features_per_channel])
        
        return torch.tensor(wavelet_features, dtype=torch.float32, device=x.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through wavelet encoder.
        
        Args:
            x: (B, channels, time_points)
        Returns:
            (B, output_dim)
        """
        B, C, T = x.shape
        
        # Extract wavelet features
        wavelet_feats = self.extract_wavelet_features(x)
        wavelet_encoded = self.wavelet_projection(wavelet_feats)  # (B, output_dim)
        
        # Extract high-gamma features
        high_gamma = F.relu(self.high_gamma_conv(x))  # (B, 64, T)
        high_gamma = self.high_gamma_pool(high_gamma)  # (B, 64, 8)
        
        # Extract low-frequency features
        low_freq = F.relu(self.low_freq_conv(x))  # (B, 32, T)
        low_freq = self.low_freq_pool(low_freq)  # (B, 32, 8)
        
        # Combine frequency features for cross-frequency coupling
        freq_features = torch.cat([high_gamma, low_freq], dim=1)  # (B, 96, 8)
        freq_features = freq_features.transpose(1, 2)  # (B, 8, 96)
        
        # Apply cross-frequency attention
        cfc_features, _ = self.cfc_attention(freq_features, freq_features, freq_features)
        cfc_features = cfc_features.reshape(B, -1)  # (B, 96 * 8)
        
        # Fuse all features
        combined = torch.cat([wavelet_encoded, cfc_features], dim=-1)
        output = self.feature_fusion(combined)
        
        return output

# ============================================
# Neural Compressor (Autoencoder)
# ============================================

class NeuralCompressor(nn.Module):
    """
    Neural compression module adapted from MiSTR for MEG feature reduction.
    Uses autoencoder architecture with phoneme-aware latent space.
    """
    
    def __init__(self, 
                 input_dim: int,
                 latent_dim: int = 64,
                 num_phonemes: int = 39):
        super().__init__()
        
        # Encoder with residual connections
        self.encoder = nn.ModuleList([
            nn.Linear(input_dim, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, latent_dim)
        ])
        
        self.encoder_norms = nn.ModuleList([
            nn.LayerNorm(512),
            nn.LayerNorm(256),
            nn.LayerNorm(128),
            nn.LayerNorm(latent_dim)
        ])
        
        # Decoder (for reconstruction loss)
        self.decoder = nn.ModuleList([
            nn.Linear(latent_dim, 128),
            nn.Linear(128, 256),
            nn.Linear(256, 512),
            nn.Linear(512, input_dim)
        ])
        
        self.decoder_norms = nn.ModuleList([
            nn.LayerNorm(128),
            nn.LayerNorm(256),
            nn.LayerNorm(512),
            nn.LayerNorm(input_dim)
        ])
        
        # Phoneme-specific projection heads for contrastive learning
        self.phoneme_projector = nn.Linear(latent_dim, num_phonemes)
        
        self.dropout = nn.Dropout(0.1)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        for i, (layer, norm) in enumerate(zip(self.encoder, self.encoder_norms)):
            x = layer(x)
            x = norm(x)
            if i < len(self.encoder) - 1:  # Don't apply ReLU to final layer
                x = F.relu(x)
                x = self.dropout(x)
        return x
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation."""
        for i, (layer, norm) in enumerate(zip(self.decoder, self.decoder_norms)):
            z = layer(z)
            z = norm(z)
            if i < len(self.decoder) - 1:
                z = F.relu(z)
                z = self.dropout(z)
        return z
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through compressor.
        
        Returns:
            latent: Compressed representation
            reconstructed: Reconstructed input
            phoneme_logits: Phoneme prediction from latent space
        """
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        phoneme_logits = self.phoneme_projector(latent)
        
        return latent, reconstructed, phoneme_logits

# ============================================
# Temporal Attention Network (Transformer)
# ============================================

class TemporalAttentionNetwork(nn.Module):
    """
    Transformer-based temporal modeling adapted from MiSTR.
    Captures long-range dependencies in MEG signals for phoneme recognition.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.1,
                 max_seq_len: int = 125):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Learnable positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(1, max_seq_len, hidden_dim) * 0.1
        )
        
        # Multi-scale temporal processing
        self.local_attention = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads // 2,  # Fewer heads for local patterns
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        
        self.global_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # Temporal pooling strategies
        self.temporal_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.attention_pool = nn.MultiheadAttention(
            hidden_dim, num_heads=4, batch_first=True
        )
        
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        """
        Process temporal sequence through transformer.
        
        Args:
            x: (B, T, D) or (B, D) input features
            seq_len: Optional sequence length for reshaping
            
        Returns:
            (B, hidden_dim) aggregated features
        """
        # Handle both sequential and flattened inputs
        if len(x.shape) == 2 and seq_len is not None:
            B, D = x.shape
            T = seq_len
            x = x.view(B, T, D // T)
        elif len(x.shape) == 3:
            B, T, D = x.shape
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        # Project to hidden dimension
        x = self.input_projection(x)  # (B, T, hidden_dim)
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :T, :]
        
        # Local attention processing
        local_features = self.local_attention(x)  # (B, T, hidden_dim)
        
        # Global transformer processing
        global_features = self.global_transformer(x + local_features)  # (B, T, hidden_dim)
        
        # Apply temporal convolution
        conv_features = self.temporal_conv(global_features.transpose(1, 2))  # (B, hidden_dim, T)
        conv_features = conv_features.transpose(1, 2)  # (B, T, hidden_dim)
        
        # Combine features
        combined = global_features + conv_features
        
        # Attention pooling for final representation
        # Use learned query for pooling
        query = combined.mean(dim=1, keepdim=True)  # (B, 1, hidden_dim)
        pooled, _ = self.attention_pool(query, combined, combined)  # (B, 1, hidden_dim)
        
        return pooled.squeeze(1)  # (B, hidden_dim)

# ============================================
# Main MiSTR Model for Phoneme Classification
# ============================================

class MiSTRPhonemeClassifier(L.LightningModule):
    """
    Main MiSTR-adapted model for MEG phoneme classification.
    Combines wavelet encoding, neural compression, and temporal attention.
    
    This is the main class that will be instantiated by train.py
    """
    
    def __init__(self,
                 time_points: int = 125,  # Required by train.py
                 meg_channels: int = 306,
                 vocab_size: int = 39,
                 hidden_dim: int = 256,
                 latent_dim: int = 64,
                 num_transformer_layers: int = 4,
                 num_attention_heads: int = 8,
                 learning_rate: float = 1e-4,
                 use_reconstruction_loss: bool = True,
                 reconstruction_weight: float = 0.1,
                 use_contrastive_loss: bool = True,
                 contrastive_weight: float = 0.1,
                 use_wavelet: bool = True,
                 wavelet_type: str = 'db4',
                 wavelet_level: int = 4,
                 dropout: float = 0.2,
                 **kwargs):  # Accept additional kwargs for compatibility
        super().__init__()
        self.save_hyperparameters()
        
        # Store time_points for compatibility
        self.time_points = time_points
        
        # Wavelet-based MEG encoder (optional)
        if use_wavelet:
            self.wavelet_encoder = WaveletMEGEncoder(
                meg_channels=meg_channels,
                time_points=time_points,
                wavelet=wavelet_type,
                decomposition_level=wavelet_level,
                output_dim=hidden_dim
            )
        else:
            self.wavelet_encoder = None
        
        # Spatial-temporal feature extraction
        self.spatial_encoder = nn.Sequential(
            nn.Conv1d(meg_channels, hidden_dim, kernel_size=1),  # Channel mixing
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Neural compressor for dimensionality reduction
        compressor_input_dim = hidden_dim * 2 if use_wavelet else hidden_dim
        self.compressor = NeuralCompressor(
            input_dim=compressor_input_dim,
            latent_dim=latent_dim,
            num_phonemes=vocab_size
        )
        
        # Temporal attention network
        self.temporal_attention = TemporalAttentionNetwork(
            input_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_heads=num_attention_heads,
            num_layers=num_transformer_layers,
            dropout=dropout,
            max_seq_len=time_points
        )
        
        # Multi-scale temporal aggregation
        self.multi_scale_pools = nn.ModuleList([
            nn.AdaptiveAvgPool1d(output_size=size)
            for size in [8, 16, 32]
        ])
        
        # Final classification head with skip connections
        classifier_input_dim = hidden_dim + latent_dim + len(self.multi_scale_pools) * hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size)
        )
        
        # Loss functions
        self.criterion = nn.CrossEntropyLoss()
        self.reconstruction_loss = nn.MSELoss()
        self.f1_macro = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
        
        # Contrastive loss temperature
        self.temperature = nn.Parameter(torch.tensor(0.07))
        
    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract multi-level features from MEG input.
        
        Args:
            x: (B, channels, time_points)
            
        Returns:
            Dictionary of features at different levels
        """
        B, C, T = x.shape
        
        # Wavelet-based features (optional)
        if self.wavelet_encoder is not None:
            wavelet_features = self.wavelet_encoder(x)  # (B, hidden_dim)
        else:
            wavelet_features = torch.zeros(B, self.hparams.hidden_dim, device=x.device)
        
        # Spatial-temporal features
        spatial_features = self.spatial_encoder(x)  # (B, hidden_dim, T)
        
        # Multi-scale temporal pooling
        multi_scale_features = []
        for pool in self.multi_scale_pools:
            pooled = pool(spatial_features)  # (B, hidden_dim, pool_size)
            pooled = pooled.mean(dim=-1)  # (B, hidden_dim)
            multi_scale_features.append(pooled)
        
        # Global spatial features
        global_spatial = spatial_features.mean(dim=-1)  # (B, hidden_dim)
        
        return {
            'wavelet': wavelet_features,
            'spatial': global_spatial,
            'multi_scale': multi_scale_features,
            'spatial_temporal': spatial_features
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: (B, channels, time_points)
            
        Returns:
            (B, vocab_size) logits
        """
        # Extract features
        features = self.extract_features(x)
        
        # Combine wavelet and spatial features
        if self.hparams.use_wavelet:
            combined_features = torch.cat([
                features['wavelet'],
                features['spatial']
            ], dim=-1)  # (B, hidden_dim * 2)
        else:
            combined_features = features['spatial']  # (B, hidden_dim)
        
        # Neural compression
        latent, reconstructed, phoneme_logits_aux = self.compressor(combined_features)
        
        # Temporal attention processing
        # Expand latent to sequence for temporal processing
        B = x.size(0)
        latent_seq = latent.unsqueeze(1).expand(-1, self.time_points, -1)  # (B, T, latent_dim)
        temporal_features = self.temporal_attention(latent_seq)  # (B, hidden_dim)
        
        # Combine all features for final classification
        final_features = torch.cat([
            temporal_features,
            latent,
            *features['multi_scale']
        ], dim=-1)
        
        # Final classification
        logits = self.classifier(final_features)
        
        # Store intermediate outputs for loss computation
        self.last_reconstructed = reconstructed
        self.last_combined_features = combined_features
        self.last_phoneme_logits_aux = phoneme_logits_aux
        self.last_latent = latent
        
        return logits
    
    def compute_contrastive_loss(self, latent: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss for phoneme-aware latent space.
        """
        # Normalize latent representations
        latent_norm = F.normalize(latent, p=2, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(latent_norm, latent_norm.T) / self.temperature
        
        # Create mask for same-class pairs
        mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask.fill_diagonal_(False)
        
        # Compute contrastive loss
        if mask.sum() > 0:
            pos_similarity = similarity[mask].mean()
            neg_similarity = similarity[~mask].mean()
            loss = -pos_similarity + neg_similarity + 1.0  # Margin of 1.0
            return F.relu(loss)
        else:
            return torch.tensor(0.0, device=latent.device)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # Forward pass
        y_hat = self(x)
        
        # Primary classification loss
        loss = self.criterion(y_hat, y)
        
        # Reconstruction loss
        if self.hparams.use_reconstruction_loss:
            recon_loss = self.reconstruction_loss(
                self.last_reconstructed, 
                self.last_combined_features
            )
            loss += self.hparams.reconstruction_weight * recon_loss
            self.log('train_recon_loss', recon_loss, prog_bar=False)
        
        # Auxiliary phoneme prediction loss from compressor
        aux_loss = self.criterion(self.last_phoneme_logits_aux, y)
        loss += 0.1 * aux_loss
        self.log('train_aux_loss', aux_loss, prog_bar=False)
        
        # Contrastive loss
        if self.hparams.use_contrastive_loss:
            contrastive_loss = self.compute_contrastive_loss(self.last_latent, y)
            loss += self.hparams.contrastive_weight * contrastive_loss
            self.log('train_contrastive_loss', contrastive_loss, prog_bar=False)
        
        # Metrics
        f1_macro = self.f1_macro(y_hat, y)
        
        # Logging
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
        
        # Log per-class accuracy periodically
        if batch_idx == 0:
            predictions = y_hat.argmax(dim=-1)
            for phoneme_id in range(self.hparams.vocab_size):
                mask = y == phoneme_id
                if mask.sum() > 0:
                    accuracy = (predictions[mask] == phoneme_id).float().mean()
                    self.log(f'val_acc_phoneme_{phoneme_id}', accuracy)
        
        return loss
    
    def configure_optimizers(self):
        # Separate parameter groups with different learning rates
        params = []
        
        if self.wavelet_encoder is not None:
            params.append({'params': self.wavelet_encoder.parameters(), 'lr': self.hparams.learning_rate * 0.5})
        
        params.extend([
            {'params': self.spatial_encoder.parameters(), 'lr': self.hparams.learning_rate},
            {'params': self.compressor.parameters(), 'lr': self.hparams.learning_rate},
            {'params': self.temporal_attention.parameters(), 'lr': self.hparams.learning_rate},
            {'params': self.classifier.parameters(), 'lr': self.hparams.learning_rate * 2}
        ])
        
        optimizer = torch.optim.AdamW(params, weight_decay=0.01)
        
        # Cosine annealing with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }