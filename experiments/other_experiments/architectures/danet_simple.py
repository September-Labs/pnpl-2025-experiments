"""
Simple DAnet-based MEG Phoneme Classification Model with Automatic Pre-training
Handles pre-training automatically on first run, then uses cached weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np
from torchmetrics import F1Score
from pathlib import Path
import pickle
from typing import Optional, Dict
from torch.utils.data import DataLoader, ConcatDataset

# ============================================
# DAnet Components (from original code)
# ============================================

class Decomposer(nn.Module):
    """Single-channel decomposer from DAnet"""
    def __init__(self, n_components, kernel_size=125, n_mid_channels=4, n_mid_layers=2):
        super().__init__()
        self.n_components = n_components
        
        # Build detector and atom networks for each component
        self.detectors = nn.ModuleList()
        self.atoms = nn.ModuleList()
        
        for _ in range(n_components):
            # Detector network
            detector_layers = [
                nn.Conv1d(1, n_mid_channels, kernel_size, padding='same'),
                nn.ReLU(),
            ]
            for _ in range(n_mid_layers):
                detector_layers.extend([
                    nn.Conv1d(n_mid_channels, n_mid_channels, kernel_size, padding='same'),
                    nn.ReLU(),
                ])
            detector_layers.extend([
                nn.Conv1d(n_mid_channels, 1, kernel_size, padding='same'),
                nn.ReLU(),
            ])
            self.detectors.append(nn.Sequential(*detector_layers))
            
            # Atom network (single convolution)
            self.atoms.append(
                nn.Conv1d(1, 1, kernel_size, bias=False, padding='same')
            )
    
    def forward(self, x, return_detector_outputs=False):
        """
        Args:
            x: (B, T) single channel input
            return_detector_outputs: If True, return detector outputs instead of reconstructions
        Returns:
            (B, n_components, T) decomposed signals or detector outputs
        """
        B, T = x.shape
        x = x.unsqueeze(1)  # (B, 1, T)
        
        outputs = []
        for detector, atom in zip(self.detectors, self.atoms):
            det_out = detector(x)  # (B, 1, T)
            
            if return_detector_outputs:
                outputs.append(det_out.squeeze(1))  # (B, T)
            else:
                atom_out = atom(det_out)  # (B, 1, T)
                outputs.append(atom_out.squeeze(1))  # (B, T)
        
        return torch.stack(outputs, dim=1)  # (B, n_components, T)

# ============================================
# Simple DAnet-based Phoneme Classifier with Auto Pre-training
# ============================================

class SimpleDAnetPhonemeClassifier(L.LightningModule):
    """
    Minimalist DAnet-based classifier for MEG phoneme classification.
    Automatically handles pre-training on first run.
    """
    
    def __init__(self,
                 meg_channels=306,
                 time_points=125,
                 vocab_size=39,
                 # DAnet parameters
                 n_components=8,
                 n_selected_channels=30,
                 danet_kernel_size=125,
                 danet_pretrained_dir="./danet_pretrained",
                 danet_pretrain_epochs=50,
                 danet_pretrain_batch_size=64,
                 freeze_danet=True,
                 force_retrain_danet=False,
                 # Training parameters
                 learning_rate=1e-3,
                 label_smoothing=0.0,
                 dropout=0.3,
                 # Data paths for pre-training
                 data_path=None,
                 data_tmin=0.0,
                 data_tmax=0.5):
        super().__init__()
        self.save_hyperparameters()
        
        # Channel selection
        self.channel_indices = self._get_auditory_channels(meg_channels, n_selected_channels)
        self.n_channels = len(self.channel_indices)
        
        # Initialize DAnet decomposers
        self.decomposers = nn.ModuleList([
            Decomposer(
                n_components=n_components,
                kernel_size=danet_kernel_size,
                n_mid_channels=4,
                n_mid_layers=2
            )
            for _ in range(self.n_channels)
        ])
        
        # Flag to track if pre-training is needed
        self.pretrained_loaded = False
        
        # Try to load pre-trained weights (will be done in setup if not here)
        self._try_load_pretrained()
        
        # Classification head
        feature_dim = self.n_channels * n_components * 2  # *2 for mean and max pooling
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, vocab_size)
        )
        
        # Loss and metrics
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.f1_macro = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
        
        # For pre-training
        self.pretrain_reconstruction_loss = nn.MSELoss()
        self.pretrain_alpha_l1 = 1e-5
    
    def _get_auditory_channels(self, total_channels, n_select):
        """Select channels most relevant for auditory processing."""
        if n_select >= total_channels:
            return list(range(total_channels))
        
        # For MEG, prioritize temporal channels
        temporal_priority = []
        
        # Add temporal region channels first (approximate locations)
        for start, end in [(50, 100), (200, 250)]:
            for ch in range(start, min(end, total_channels)):
                if len(temporal_priority) < n_select:
                    temporal_priority.append(ch)
        
        # Fill remaining with evenly spaced channels
        remaining = n_select - len(temporal_priority)
        if remaining > 0:
            other_channels = [ch for ch in range(total_channels) if ch not in temporal_priority]
            step = max(1, len(other_channels) // remaining)
            for i in range(0, len(other_channels), step):
                if len(temporal_priority) < n_select:
                    temporal_priority.append(other_channels[i])
        
        return temporal_priority[:n_select]
    
    def _try_load_pretrained(self):
        """Try to load pre-trained weights if they exist."""
        if self.pretrained_loaded:
            return True
            
        pretrained_dir = Path(self.hparams.danet_pretrained_dir)
        pretrained_path = pretrained_dir / "danet_pretrained.pt"
        metadata_path = pretrained_dir / "danet_metadata.pkl"
        
        if pretrained_path.exists() and not self.hparams.force_retrain_danet:
            print(f"Loading pre-trained DAnet from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # Load metadata
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                print(f"Pre-trained with: {metadata}")
            
            # Load weights for each decomposer
            if 'decomposers' in checkpoint:
                for i, state_dict in enumerate(checkpoint['decomposers']):
                    if i < len(self.decomposers):
                        self.decomposers[i].load_state_dict(state_dict)
                print(f"Successfully loaded pre-trained weights for {len(checkpoint['decomposers'])} decomposers")
                self.pretrained_loaded = True
                
                # Freeze DAnet if specified
                if self.hparams.freeze_danet:
                    for decomposer in self.decomposers:
                        for param in decomposer.parameters():
                            param.requires_grad = False
                
                return True
            else:
                print("Warning: Checkpoint format not recognized")
                return False
        else:
            if self.hparams.force_retrain_danet:
                print("Force retrain flag set - will retrain DAnet")
            else:
                print(f"Pre-trained weights not found at {pretrained_path}")
            return False
    
    def _run_pretraining(self):
        """Run unsupervised pre-training on all available data."""
        print("\n" + "="*50)
        print("Starting automatic DAnet pre-training...")
        print("="*50 + "\n")
        
        # Check if we should use preprocessed data
        preprocessed_dir = Path(self.hparams.data_path)
        
        # Look for preprocessed H5 files
        train_h5 = preprocessed_dir / 'train_grouped.h5'
        val_h5 = preprocessed_dir / 'validation_grouped.h5'
        test_h5 = preprocessed_dir / 'test_grouped.h5'
        
        if train_h5.exists() and val_h5.exists() and test_h5.exists():
            print("Using preprocessed grouped H5 data for pre-training...")
            
            # Import GroupedDataset for loading preprocessed data
            from pnpl.datasets import GroupedDataset
            
            # Load preprocessed datasets
            train_dataset = GroupedDataset(
                preprocessed_path=train_h5,
                load_to_memory=True  # Load to memory for faster training
            )
            val_dataset = GroupedDataset(
                preprocessed_path=val_h5,
                load_to_memory=True
            )
            test_dataset = GroupedDataset(
                preprocessed_path=test_h5,
                load_to_memory=True
            )
            
            print(f"  Loaded train: {len(train_dataset)} grouped samples")
            print(f"  Loaded val: {len(val_dataset)} grouped samples")
            print(f"  Loaded test: {len(test_dataset)} grouped samples")
            
            # Combine datasets
            from torch.utils.data import ConcatDataset
            combined_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])
            
        else:
            # Fall back to loading raw data
            print("Preprocessed H5 files not found, loading raw data...")
            from pnpl.datasets import LibriBrainPhoneme
            
            all_datasets = []
            for partition in ['train', 'validation', 'test']:
                try:
                    dataset = LibriBrainPhoneme(
                        data_path=self.hparams.data_path,
                        partition=partition,
                        tmin=self.hparams.data_tmin,
                        tmax=self.hparams.data_tmax,
                        standardize=True
                    )
                    all_datasets.append(dataset)
                    print(f"  Loaded {partition}: {len(dataset)} samples")
                except Exception as e:
                    print(f"  Could not load {partition}: {e}")
            
            if not all_datasets:
                raise ValueError("No datasets could be loaded for pre-training")
            
            combined_dataset = ConcatDataset(all_datasets)
        
        print(f"Total samples for pre-training: {len(combined_dataset)}")
        
        # Create dataloader
        pretrain_loader = DataLoader(
            combined_dataset,
            batch_size=self.hparams.danet_pretrain_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True if 4 > 0 else False
        )
        
        # Rest of the pre-training code remains the same...
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        
        # Create optimizer for pre-training
        pretrain_optimizer = torch.optim.Adam(
            [p for d in self.decomposers for p in d.parameters()],
            lr=1e-4
        )
        
        # Training loop
        for epoch in range(self.hparams.danet_pretrain_epochs):
            epoch_loss = 0
            n_batches = 0
            
            for batch_idx, (x, _) in enumerate(pretrain_loader):
                x = x.to(device)
                B, C, T = x.shape
                
                # Select subset of channels
                total_loss = 0
                total_l1_loss = 0
                
                for idx, ch_idx in enumerate(self.channel_indices):
                    if ch_idx >= C:
                        continue
                    
                    ch_data = x[:, ch_idx, :]  # (B, T)
                    decomposer = self.decomposers[idx]
                    
                    # Get decomposed signals
                    decomposed = decomposer(ch_data, return_detector_outputs=False)
                    detector_outputs = decomposer(ch_data, return_detector_outputs=True)
                    
                    # Reconstruct
                    reconstructed = decomposed.sum(dim=1)
                    
                    # Losses
                    rec_loss = self.pretrain_reconstruction_loss(reconstructed, ch_data)
                    l1_loss = detector_outputs.abs().mean()
                    
                    total_loss += rec_loss
                    total_l1_loss += l1_loss
                
                # Average losses
                n_channels_used = min(len(self.channel_indices), C)
                avg_loss = total_loss / n_channels_used
                avg_l1_loss = total_l1_loss / n_channels_used
                
                # Combined loss
                loss = avg_loss + self.pretrain_alpha_l1 * avg_l1_loss
                
                # Optimize
                pretrain_optimizer.zero_grad()
                loss.backward()
                pretrain_optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
                
                if batch_idx % 10 == 0:
                    print(f"  Epoch {epoch+1}/{self.hparams.danet_pretrain_epochs}, "
                        f"Batch {batch_idx}/{len(pretrain_loader)}, "
                        f"Loss: {loss.item():.4f}")
            
            avg_epoch_loss = epoch_loss / n_batches
            print(f"Epoch {epoch+1} complete. Average loss: {avg_epoch_loss:.4f}")
    
        # Save pre-trained weights
        pretrained_dir = Path(self.hparams.danet_pretrained_dir)
        pretrained_dir.mkdir(parents=True, exist_ok=True)
        pretrained_path = pretrained_dir / "danet_pretrained.pt"
        metadata_path = pretrained_dir / "danet_metadata.pkl"
        
        torch.save({
            'decomposers': [d.state_dict() for d in self.decomposers]
        }, pretrained_path)
        
        # Save metadata
        metadata = {
            'n_components': self.hparams.n_components,
            'n_channels': self.n_channels,
            'kernel_size': self.hparams.danet_kernel_size,
            'pretrain_epochs': self.hparams.danet_pretrain_epochs,
            'status': 'trained',
            'training_samples': len(combined_dataset),
            'final_loss': avg_epoch_loss
        }
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"\nPre-training complete!")
        print(f"Weights saved to {pretrained_path}")
        print(f"Metadata saved to {metadata_path}")
        
        self.pretrained_loaded = True
        
        # Freeze DAnet after pre-training if specified
        if self.hparams.freeze_danet:
            for decomposer in self.decomposers:
                for param in decomposer.parameters():
                    param.requires_grad = False
            print("DAnet parameters frozen")
        
        print("\n" + "="*50)
        print("Continuing with main training...")
        print("="*50 + "\n")
    
    def on_fit_start(self):
        """Called at the very beginning of fit, before any training."""
        # Check if we need to run pre-training
        if not self.pretrained_loaded:
            self._run_pretraining()
        else:
            print("Using existing pre-trained DAnet weights")
            
        # Ensure freezing is applied if needed
        if self.hparams.freeze_danet:
            for decomposer in self.decomposers:
                for param in decomposer.parameters():
                    param.requires_grad = False
    
    def extract_features(self, x):
        """Extract features using DAnet decomposers."""
        B, C, T = x.shape
        features = []
        
        # Process selected channels
        for ch_idx, decomposer in zip(self.channel_indices, self.decomposers):
            if ch_idx >= C:
                continue
                
            # Get single channel data
            ch_data = x[:, ch_idx, :]  # (B, T)
            
            # Get detector outputs (pattern presence indicators)
            detector_outputs = decomposer(ch_data, return_detector_outputs=True)  # (B, n_components, T)
            
            # Temporal aggregation
            mean_pool = detector_outputs.mean(dim=-1)  # (B, n_components)
            max_pool = detector_outputs.max(dim=-1)[0]  # (B, n_components)
            
            features.append(mean_pool)
            features.append(max_pool)
        
        # Concatenate all features
        features = torch.cat(features, dim=-1)
        
        # Pad with zeros if we have fewer channels than expected
        expected_dim = self.n_channels * self.hparams.n_components * 2
        if features.shape[1] < expected_dim:
            padding = torch.zeros(B, expected_dim - features.shape[1], device=features.device)
            features = torch.cat([features, padding], dim=1)
        
        return features
    
    def forward(self, x):
        """Forward pass for classification."""
        features = self.extract_features(x)
        logits = self.classifier(features)
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
    
    def configure_optimizers(self):
        # Only optimize classifier parameters if DAnet is frozen
        if self.hparams.freeze_danet:
            params = self.classifier.parameters()
        else:
            params = self.parameters()
        
        optimizer = torch.optim.AdamW(
            params,
            lr=self.hparams.learning_rate,
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=50,
            eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }