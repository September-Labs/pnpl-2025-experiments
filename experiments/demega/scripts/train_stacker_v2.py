#!/usr/bin/env python3
"""
Multi-Stage Stacking Training Script for DeMEGa

This script implements a 3-stage training process:
1.  **Stage 1:** Train multiple "base" SingleStageMEGClassifier models, each on
    a different preprocessed dataset (e.g., different averaging groups).
2.  **Stage 2:** Use the trained base models to generate predictions (logits)
    for the full train, validation, and test sets. These logits are
    concatenated to form a new "meta-dataset".
3.  **Stage 3:** Train a "main" meta-model (StackingClassifier) on this
    meta-dataset. This stage is repeated N times for robustness.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torchmetrics
import lightning as L
# Modified import to include ConcatDataset, which is required by the imported pseudo-labeling functions
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any
from torchmetrics import F1Score

# from .preprocess_data.get_preprocess_dataloader import create_dataloaders_multi_preprocessed # Old import
from .preprocess_data.get_weighted_data import create_weighted_sampler, calculate_class_weights, get_dataset_labels
from .preprocess_data.get_preprocess_dataloader import PseudoLabeledDataset, load_holdout_with_pseudolabels, create_dataloaders_preprocessed, create_dataloaders_multi_preprocessed

from .train import load_config
# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from meg_classifier.models import SingleStageMEGClassifier

# --- Global Helper Function ---

def get_base_model_from_config(config: Dict[str, Any]) -> SingleStageMEGClassifier:
    """Creates the SingleStageMEGClassifier instance from the config dict."""
    model_config = config['model']
    
    # Calculate time points (copied from your train.py)
    tmin = config['data']['tmin']
    tmax = config['data']['tmax']
    sampling_rate = 250  # Hz
    time_points = int((tmax - tmin) * sampling_rate)
    
    params = model_config.get('params', {})
    params['time_points'] = time_points
    
    # Ensure vocab_size is present
    if 'vocab_size' not in params:
        print("Warning: 'vocab_size' not in config, defaulting to 39.")
        params['vocab_size'] = 39
        
    model = SingleStageMEGClassifier(**params)
    return model

# --- Stage 1: Prediction Step Patch ---

def patched_predict_step(self, batch, batch_idx, dataloader_idx=0):
    """
    Patched predict_step for SingleStageMEGClassifier.
    This function will be added to the class dynamically.
    """
    # Batch contains (data, labels). We only need data for prediction.
    x, _ = batch
    # Call the forward pass with return_all=False to get logits
    logits = self(x, return_all=False)
    return logits


# --- Stage 2: Meta-Model Definition ---

class StackingClassifier(L.LightningModule):
    """
    This is the 'main' meta-model (Stage 3).
    It learns from the concatenated logits of the base models.
    """
    def __init__(self, 
                 num_inputs: int, 
                 num_classes: int, 
                 hidden_dim: int = 512, 
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-4,
                 metric_type: str = "f1_macro"):
        super().__init__()
        self.save_hyperparameters()
        
        self.stacker_head = nn.Sequential(
            nn.BatchNorm1d(self.hparams.num_inputs),
            nn.Linear(self.hparams.num_inputs, self.hparams.hidden_dim),
            nn.SiLU(),
            nn.BatchNorm1d(self.hparams.hidden_dim),
            nn.Dropout(0.5),
            nn.Linear(self.hparams.hidden_dim, self.hparams.num_classes)
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup metrics (mirroring SingleStageMEGClassifier)
        self.metric_name = "f1_macro" if metric_type == "f1_macro" else "balanced_acc"
        metric_class = F1Score if metric_type == "f1_macro" else torchmetrics.Accuracy
        
        self.train_metric = metric_class(
            num_classes=num_classes, average='macro', task="multiclass"
        )
        self.val_metric = metric_class(
            num_classes=num_classes, average='macro', task="multiclass"
        )

    def forward(self, x):
        # x will have shape [batch, num_models * num_classes]
        return self.stacker_head(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        self.log('train_loss', loss, prog_bar=True)
        self.train_metric.update(logits, y)
        self.log(f'train_{self.metric_name}', self.train_metric, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        self.val_metric.update(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log(f'val_{self.metric_name}', self.val_metric, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        # Use a simple scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

class LogitDataset(Dataset):
    """A simple dataset for the meta-model."""
    def __init__(self, stacked_logits, labels):
        self.stacked_logits = stacked_logits
        self.labels = labels
        assert len(self.stacked_logits) == len(self.labels), "Logits and labels must have the same length"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.stacked_logits[idx], self.labels[idx]


# --- Stage 1: Base Model Training Function ---

def train_base_model(
    base_config: Dict[str, Any],
    preprocessed_dir: str,
    preprocessed_root: Path,
    save_dir: Path
) -> str:
    """
    Trains a single base model on a specific preprocessed directory.
    
    Returns:
        Path to the best model checkpoint.
    """
    print("\n" + "="*80)
    print(f"🚀 STAGE 1: Training Base Model for: {preprocessed_dir}")
    print(f"Save Directory: {save_dir}")
    print("="*80)
    
    # --- 1. Load Config and Override ---
    # Create a deep copy to avoid modifying the original config
    config = yaml.safe_load(yaml.dump(base_config))
    
    # Override data paths
    config['data']['preprocessed_dirs'] = [preprocessed_dir]
    
    # Override save directory
    config['training']['save_dir'] = str(save_dir)
    
    # Ensure weighted sampling is enabled if specified (it's in the dataloader fn)
    config['training']['use_weighted_sampling'] = base_config['training'].get('use_weighted_sampling', True)

    # --- 2. Create Dataloaders ---
    # create_dataloaders_multi_preprocessed(config, preprocessed_root, preprocessed_dirs)
    print(f"Loading data from root: {preprocessed_root}, dir: {[preprocessed_dir]}")
    # This call will now automatically use pseudo-labeling if config['holdout']['use_holdout'] is True
    train_loader, val_loader, _ = create_dataloaders_multi_preprocessed(
        config,
        preprocessed_root,
        [preprocessed_dir]
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # --- 3. Create Model ---
    model = get_base_model_from_config(config)

    # --- 4. Setup Trainer & Callbacks ---
    log_dir = save_dir / 'logs'
    checkpoint_dir = save_dir / "checkpoints"
    
    csv_logger = CSVLogger(save_dir=log_dir, name="csv_logs")
    tb_logger = TensorBoardLogger(save_dir=log_dir, name="tensorboard_logs")
    
    monitor_metric = f"val_{model.metric_name}"
    print(f"Monitoring metric: {monitor_metric}")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"base_model-{{epoch:02d}}-{{{monitor_metric}:.3f}}",
        monitor=monitor_metric,
        mode="max",
        save_top_k=1,
    )
    
    early_stop_callback = EarlyStopping(
        monitor=monitor_metric,
        mode="max",
        patience=config['training'].get('early_stopping_patience', 15),
        verbose=True
    )
    
    trainer = L.Trainer(
        max_epochs=config['training']['max_epochs'],
        devices=config['training'].get('devices', 1),
        accelerator=config['training'].get('accelerator', 'auto'),
        logger=[csv_logger, tb_logger],
        callbacks=[checkpoint_callback, early_stop_callback],
        gradient_clip_val=config['training'].get('gradient_clip_val', 1.0),
        accumulate_grad_batches=config['training'].get('accumulate_grad_batches', 1),
        precision="16-mixed" if config['training'].get('accelerator', 'auto') == 'gpu' else 32
    )

    # --- 5. Train ---
    trainer.fit(model, train_loader, val_loader)
    
    print(f"✅ Stage 1 Complete for: {preprocessed_dir}")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    
    return checkpoint_callback.best_model_path


def main(args):
    """Main stacking pipeline."""
    
    # --- 0. Setup ---
    L.seed_everything(42, workers=True)
    config = load_config(args.config)
    
    # Define experiment directories
    stacking_experiments_dir = Path(config['training'].get('save_dir', './experiments')) / "stacking_run"
    base_model_dir = stacking_experiments_dir / "base_models"
    meta_model_dir = stacking_experiments_dir / "meta_model"
    os.makedirs(base_model_dir, exist_ok=True)
    os.makedirs(meta_model_dir, exist_ok=True)

    preprocessed_root = Path(args.preprocessed_root)
    base_data_dirs = args.base_data_dirs

    print(f"Stacking experiment starting...")
    print(f"Config: {args.config}")
    print(f"Preprocessed Root: {preprocessed_root}")
    print(f"Base Data Dirs: {base_data_dirs}")
    print(f"Meta-Model Runs: {args.n_runs}")
    
    all_base_model_checkpoints = []

    # --- STAGE 1: Train Base Models ---
    for dir_name in base_data_dirs:
        # Create a unique save directory for this base model
        model_save_dir = base_model_dir / Path(dir_name).name
        
        # Check if a checkpoint already exists
        checkpoint = list(model_save_dir.glob("checkpoints/*.ckpt"))
        if checkpoint and not args.force_retrain_base:
            print(f"Found existing checkpoint for {dir_name}, skipping training.")
            all_base_model_checkpoints.append(str(checkpoint[0]))
        else:
            ckpt_path = train_base_model(
                base_config=config,
                preprocessed_dir=dir_name,
                preprocessed_root=preprocessed_root,
                save_dir=model_save_dir
            )
            all_base_model_checkpoints.append(ckpt_path)

    print("\n" + "="*80)
    print(f"✅ STAGE 1 (All Base Models) Complete.")
    print(f"Models to use for stacking: {all_base_model_checkpoints}")
    print("="*80)

    # --- STAGE 2: Create Meta-Dataset ---
    print(f"🚀 STAGE 2: Generating Meta-Dataset...")

    # !! Patch the model class with the predict_step !!
    SingleStageMEGClassifier.predict_step = patched_predict_step
    
    # Load the FULL dataloaders for prediction
    # We need unshuffled dataloaders for all splits to ensure order
    
    # 1. Create a config for prediction (no sampling/shuffling)
    pred_config = yaml.safe_load(yaml.dump(config))
    pred_config['training']['use_weighted_sampling'] = False
    
    # 2. Get the dataloaders (which will be shuffled by default)
    # This call will also use pseudo-labeling if config['holdout']['use_holdout'] is True
    train_loader_sampled, val_loader, test_loader = create_dataloaders_multi_preprocessed(
        pred_config,
        preprocessed_root,
        base_data_dirs
    )
    
    # 3. Re-create DataLoaders for prediction (unshuffled)
    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']
    
    train_pred_loader = DataLoader(
        train_loader_sampled.dataset,
        batch_size=batch_size * 2, # Speed up prediction
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    # val_loader and test_loader are already unshuffled
    val_pred_loader = DataLoader(
        val_loader.dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_pred_loader = DataLoader(
        test_loader.dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Prediction loaders created:")
    print(f"Train: {len(train_pred_loader.dataset)} samples")
    print(f"Val:   {len(val_pred_loader.dataset)} samples")
    print(f"Test:  {len(test_pred_loader.dataset)} samples")

    # 4. Get predictions from each base model
    all_train_logits = []
    all_val_logits = []
    all_test_logits = []
    
    pred_trainer = L.Trainer(
        devices=1, 
        accelerator=config['training'].get('accelerator', 'auto'), 
        logger=False,
        precision="16-mixed" if config['training'].get('accelerator', 'auto') == 'gpu' else 32
    )
    
    for ckpt_path in all_base_model_checkpoints:
        print(f"Getting predictions from: {Path(ckpt_path).name}...")
        model = SingleStageMEGClassifier.load_from_checkpoint(ckpt_path)
        
        train_logits = pred_trainer.predict(model, train_pred_loader)
        val_logits = pred_trainer.predict(model, val_pred_loader)
        test_logits = pred_trainer.predict(model, test_pred_loader)
        
        all_train_logits.append(torch.cat(train_logits, dim=0))
        all_val_logits.append(torch.cat(val_logits, dim=0))
        all_test_logits.append(torch.cat(test_logits, dim=0))
        
        del model # Free memory

    # 5. Concatenate logits to create meta-features
    # Shape: [num_samples, num_models * num_classes]
    meta_train_X = torch.cat(all_train_logits, dim=1)
    meta_val_X = torch.cat(all_val_logits, dim=1)
    meta_test_X = torch.cat(all_test_logits, dim=1)

    print(f"Meta-feature shape (Train): {meta_train_X.shape}")
    
    # 6. Get labels (now safe because loaders were unshuffled)
    print("Extracting labels...")
    meta_train_y = torch.cat([y for _, y in train_pred_loader], dim=0)
    meta_val_y = torch.cat([y for _, y in val_pred_loader], dim=0)
    meta_test_y = torch.cat([y for _, y in test_pred_loader], dim=0)
    
    # 7. Create final meta-datasets and dataloaders
    meta_train_ds = LogitDataset(meta_train_X, meta_train_y)
    meta_val_ds = LogitDataset(meta_val_X, meta_val_y)
    meta_test_ds = LogitDataset(meta_test_X, meta_test_y)
    
    meta_train_loader = DataLoader(
        meta_train_ds,
        batch_size=batch_size,
        shuffle=True, # Shuffle for training the meta-model
        num_workers=num_workers,
        pin_memory=True
    )
    meta_val_loader = DataLoader(
        meta_val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"✅ STAGE 2 Complete. Meta-dataset created.")
    print("="*80)

    # --- STAGE 3 & 4: Train Meta-Model N Times ---
    print(f"🚀 STAGE 3/4: Training Meta-Model for {args.n_runs} runs...")
    
    num_inputs = meta_train_X.shape[1]
    num_classes = config['model']['params'].get('vocab_size', 39)
    metric_type = config['model']['params'].get('metric_type', 'f1_macro')
    
    for i in range(args.n_runs):
        run_num = i + 1
        print("\n" + "-"*80)
        print(f"Meta-Model Run {run_num}/{args.n_runs}")
        print("-"*80)
        
        # Seed each run differently for variance
        L.seed_everything(42 + i, workers=True)
        
        run_dir = meta_model_dir / f"run_{run_num}"
        
        stacking_model = StackingClassifier(
            num_inputs=num_inputs,
            num_classes=num_classes,
            metric_type=metric_type,
            learning_rate=config['training'].get('learning_rate', 1e-4)
        )
        
        # Setup Trainer
        run_log_dir = run_dir / 'logs'
        run_checkpoint_dir = run_dir / "checkpoints"
        
        run_csv_logger = CSVLogger(save_dir=run_log_dir, name="csv_logs")
        run_tb_logger = TensorBoardLogger(save_dir=run_log_dir, name="tensorboard_logs")
        
        monitor_metric = f"val_{stacking_model.metric_name}"
        
        run_checkpoint_cb = ModelCheckpoint(
            dirpath=run_checkpoint_dir,
            filename=f"stacker-{{epoch:02d}}-{{{monitor_metric}:.3f}}",
            monitor=monitor_metric,
            mode="max",
            save_top_k=1,
        )
        
        run_early_stop_cb = EarlyStopping(
            monitor=monitor_metric,
            mode="max",
            patience=config['training'].get('early_stopping_patience', 15),
            verbose=True
        )
        
        meta_trainer = L.Trainer(
            max_epochs=config['training']['max_epochs'],
            devices=config['training'].get('devices', 1),
            accelerator=config['training'].get('accelerator', 'auto'),
            logger=[run_csv_logger, run_tb_logger],
            callbacks=[run_checkpoint_cb, run_early_stop_cb],
            gradient_clip_val=config['training'].get('gradient_clip_val', 1.0),
            accumulate_grad_batches=config['training'].get('accumulate_grad_batches', 1),
            precision="16-mixed" if config['training'].get('accelerator', 'auto') == 'gpu' else 32
        )
        
        meta_trainer.fit(stacking_model, meta_train_loader, meta_val_loader)
        
        print(f"✅ Meta-Model Run {run_num} complete.")
        print(f"Best model saved at: {run_checkpoint_cb.best_model_path}")

    print("\n" + "="*80)
    print("🎉 Stacking Pipeline Finished! 🎉")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Stage Stacking Training for DeMEGa")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to the main configuration file (used for all models)"
    )
    
    parser.add_argument(
        "--preprocessed_root",
        type=str,
        default="./preprocessed_data",
        help="Root directory where preprocessed data folders are located"
    )
    
    parser.add_argument(
        "--base_data_dirs",
        type=str,
        nargs='+',
        required=True,
        help="List of preprocessed directory names (e.g., 'grouped_5' 'grouped_100')"
    )
    
    parser.add_argument(
        "--n_runs",
        type=int,
        default=5,
        help="Number of times to train the final meta-model (Stage 3)"
    )
    
    parser.add_argument(
        "--force_retrain_base",
        action="store_true",
        help="Force retraining of base models even if checkpoints exist"
    )

    args = parser.parse_args()
    main(args)
