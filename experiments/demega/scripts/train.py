#!/usr/bin/env python3
"""
Training script for DeMEGa Phoneme Classifier

This script trains the MEG phoneme classification model using the pnpl library
for data loading and Lightning for training orchestration.
"""

import os
import sys
import argparse
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, StochasticWeightAveraging
from pnpl.datasets import LibriBrainPhoneme, GroupedDataset
import importlib

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from meg_classifier.models import SingleStageMEGClassifier


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(config):
    """Create training and validation dataloaders."""
    data_config = config['data']
    training_config = config['training']
    
    print("Loading datasets...")
    
    # Training dataset
    train_dataset = LibriBrainPhoneme(
        data_path=data_config['data_path'],
        partition="train",
        tmin=data_config['tmin'],
        tmax=data_config['tmax'],
        standardize=False,
    )
    
    print("Extracting standardization parameters from training dataset...")
    channel_means = train_dataset.channel_means
    channel_stds = train_dataset.channel_stds
    
    # Apply signal averaging if configured
    if data_config.get('use_signal_averaging', True):
        print("Applying signal averaging to training dataset...")
        train_dataset = GroupedDataset(
            train_dataset, 
            grouped_samples=data_config.get('grouped_samples', 100),
            shuffle_seed=data_config.get('shuffle_seed', 777)
        )
    
    # Validation dataset
    val_dataset = LibriBrainPhoneme(
        data_path=data_config['data_path'],
        partition="validation",
        tmin=data_config['tmin'],
        tmax=data_config['tmax'],
        standardize=True,
        channel_means=channel_means,
        channel_stds=channel_stds,
    )
    
    if data_config.get('use_signal_averaging', True):
        print("Applying signal averaging to validation dataset...")
        val_dataset = GroupedDataset(
            val_dataset,
            grouped_samples=data_config.get('grouped_samples', 100),
            shuffle_seed=data_config.get('shuffle_seed', 777)
        )
    
    # Test dataset (combined with validation for more data)
    test_dataset = LibriBrainPhoneme(
        data_path=data_config['data_path'],
        partition="test",
        tmin=data_config['tmin'],
        tmax=data_config['tmax'],
        standardize=True,
        channel_means=channel_means,
        channel_stds=channel_stds
    )
    
    if data_config.get('use_signal_averaging', True):
        print("Applying signal averaging to test dataset...")
        test_dataset = GroupedDataset(
            test_dataset,
            grouped_samples=data_config.get('grouped_samples', 100),
            shuffle_seed=data_config.get('shuffle_seed', 777)
        )
    
    # Combine validation and test for more validation data
    from torch.utils.data import ConcatDataset
    val_dataset = ConcatDataset([val_dataset, test_dataset])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=training_config['num_workers'],
        pin_memory=True,
        persistent_workers=True if training_config['num_workers'] > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=training_config['num_workers'],
        pin_memory=True,
        persistent_workers=True if training_config['num_workers'] > 0 else False
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader

def create_model(config):
    """Create model instance from configuration."""
    model_config = config['model']
    
    # Calculate time points based on configuration
    tmin = config['data']['tmin']
    tmax = config['data']['tmax']
    sampling_rate = 250  # Hz
    time_points = int((tmax - tmin) * sampling_rate)
    
    # Get model parameters
    params = model_config.get('params', {})
    params['time_points'] = time_points
    
    print(f"Creating model with time_points={time_points} (tmin={tmin}, tmax={tmax})")
    print(f"Model parameters: {params}")
    
    model = SingleStageMEGClassifier(**params)
    
    return model


def main(args):
    """Main training function."""
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        # Try relative to script directory
        config_path = Path(__file__).parent.parent / "configs" / args.config
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {args.config}")
    
    config = load_config(config_path)
    
    # Override data path if provided
    if args.data_path:
        config['data']['data_path'] = args.data_path
    
    # Check data path exists
    data_path = Path(config['data']['data_path'])
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data path not found: {data_path}\n"
            f"Please specify the path to your MEG data using --data_path"
        )
    
    # Set seed for reproducibility
    L.seed_everything(config['training'].get('seed', 42))
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)
    
    # Create model
    model = create_model(config)
    
    # Setup save directory
    save_dir = Path(config['training'].get('save_dir', './experiments'))
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_dir = save_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    loggers = []
    
    # CSV logger
    csv_logger = CSVLogger(
        save_dir=log_dir,
        name="csv_logs",
        version=None
    )
    loggers.append(csv_logger)
    
    # TensorBoard logger
    if config['logging'].get('use_tensorboard', True):
        tb_logger = TensorBoardLogger(
            save_dir=log_dir,
            name="tensorboard_logs",
            version=None
        )
        loggers.append(tb_logger)
    
    # WandB logger (optional)
    if config['logging'].get('wandb', {}).get('project'):
        wandb_config = config['logging']['wandb']
        wandb_logger = WandbLogger(
            project=wandb_config['project'],
            entity=wandb_config.get('entity'),
            name=wandb_config.get('name', 'experiment'),
            tags=wandb_config.get('tags', []),
            notes=wandb_config.get('notes', ''),
            save_dir=log_dir,
            log_model=False
        )
        loggers.append(wandb_logger)
    
    # Setup callbacks
    callbacks = []
    
    # Model checkpoint
    checkpoint_dir = save_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    monitor_metric = "val_f1_macro"
    filename_template = "demega-{epoch:02d}-{val_f1_macro:.2f}"
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=filename_template,
        monitor=monitor_metric,
        mode="max",
        save_top_k=3,
        save_last=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    if config['training'].get('use_early_stopping', False):
        early_stop_callback = EarlyStopping(
            monitor=monitor_metric,
            mode="max",
            patience=config['training'].get('early_stopping_patience', 10),
            verbose=True
        )
        callbacks.append(early_stop_callback)
    
    # Stochastic Weight Averaging
    if config['training'].get('use_swa', False):
        swa_lrs = config['training'].get('swa_lrs', 1e-3)
        callbacks.append(StochasticWeightAveraging(swa_lrs=swa_lrs))
    
    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=config['training']['max_epochs'],
        devices=config['training'].get('devices', 1),
        accelerator=config['training'].get('accelerator', 'auto'),
        strategy=config['training'].get('strategy', 'auto'),
        logger=loggers,
        callbacks=callbacks,
        enable_checkpointing=True,
        gradient_clip_val=config['training'].get('gradient_clip_val', 1.0),
        accumulate_grad_batches=config['training'].get('accumulate_grad_batches', 1)
    )
    
    # Train the model
    print(f"Starting training with config: {config_path}")
    print(f"Save directory: {save_dir}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Monitoring metric: {monitor_metric}")
    
    trainer.fit(model, train_loader, val_loader)
    
    print(f"Training completed. Best model saved at: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeMEGa Phoneme Classifier")
    
    parser.add_argument(
        "--config",
        type=str,
        default="default_config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to MEG data (overrides config)"
    )
    
    args = parser.parse_args()
    main(args)
