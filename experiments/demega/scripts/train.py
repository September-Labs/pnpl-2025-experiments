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


def create_dataloaders(config, use_huggingface=False, grouping_level=100,
                       local_dir="./data", force_download=False):
    """Create training and validation dataloaders.

    Args:
        config: Configuration dictionary
        use_huggingface: If True, use HuggingFace h5 files instead of raw data
        grouping_level: Grouping level for h5 files (5, 10, 15, 20, 25, 30, 35, 45, 50, 55, 60, 100)
        local_dir: Directory for HuggingFace h5 files
        force_download: Force download even if files exist
    """
    data_config = config['data']
    training_config = config['training']

    if use_huggingface:
        print(f"Loading preprocessed h5 files (grouped_{grouping_level})...")

        # Build paths to h5 files
        h5_dir = Path(local_dir) / f"grouped_{grouping_level}"
        train_h5 = h5_dir / "train_grouped.h5"
        val_h5 = h5_dir / "validation_grouped.h5"
        test_h5 = h5_dir / "test_grouped.h5"

        # Check if files exist
        files_exist = all([train_h5.exists(), val_h5.exists(), test_h5.exists()])

        if not files_exist or force_download:
            if force_download:
                print("Force download enabled - downloading from HuggingFace...")
            else:
                print("H5 files not found locally - downloading from HuggingFace...")

            try:
                from huggingface_hub import snapshot_download

                print(f"Downloading grouped_{grouping_level} from wordcab/libribrain-meg-preprocessed...")
                print(f"This may take a while depending on your internet speed.")

                snapshot_download(
                    repo_id="wordcab/libribrain-meg-preprocessed",
                    repo_type="dataset",
                    allow_patterns=[f"data/grouped_{grouping_level}/**"],
                    local_dir=local_dir
                )

                print(f"Download complete! Files saved to {h5_dir}")

            except Exception as e:
                raise RuntimeError(
                    f"Failed to download from HuggingFace: {e}\n"
                    f"You can manually download using: python download_data.py --grouping_level {grouping_level}"
                )
        else:
            print(f"Using existing h5 files from {h5_dir}")

        # Load datasets from h5 files
        load_to_memory = data_config.get('load_to_memory', True)

        train_dataset = GroupedDataset(
            preprocessed_path=train_h5,
            load_to_memory=load_to_memory
        )

        val_dataset = GroupedDataset(
            preprocessed_path=val_h5,
            load_to_memory=load_to_memory
        )

        test_dataset = GroupedDataset(
            preprocessed_path=test_h5,
            load_to_memory=load_to_memory
        )

        print(f"Loaded h5 datasets:")
        print(f"  Train: {len(train_dataset)} groups")
        print(f"  Validation: {len(val_dataset)} groups")
        print(f"  Test: {len(test_dataset)} groups")

    else:
        # Legacy mode - load raw data
        print("Loading raw MEG data...")

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

        # Test dataset
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

    # Check data path exists (only needed for raw data mode)
    if not args.use_huggingface:
        data_path = Path(config['data']['data_path'])
        if not data_path.exists():
            raise FileNotFoundError(
                f"Data path not found: {data_path}\n"
                f"Please specify the path to your MEG data using --data_path\n"
                f"Or use --use_huggingface to load preprocessed h5 files"
            )
    
    # Set seed for reproducibility
    L.seed_everything(config['training'].get('seed', 42))

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        config,
        use_huggingface=args.use_huggingface,
        grouping_level=args.grouping_level,
        local_dir=args.local_dir,
        force_download=args.download
    )
    
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

    # HuggingFace h5 file loading options
    parser.add_argument(
        "--use_huggingface",
        action="store_true",
        help="Load preprocessed h5 files from HuggingFace instead of raw data"
    )

    parser.add_argument(
        "--grouping_level",
        type=int,
        default=100,
        choices=[5, 10, 15, 20, 25, 30, 35, 45, 50, 55, 60, 100],
        help="Grouping level for HuggingFace h5 files (default: 100)"
    )

    parser.add_argument(
        "--local_dir",
        type=str,
        default="./data",
        help="Local directory for HuggingFace h5 files (default: ./data)"
    )

    parser.add_argument(
        "--download",
        action="store_true",
        help="Force download from HuggingFace even if files exist locally"
    )

    args = parser.parse_args()
    main(args)
