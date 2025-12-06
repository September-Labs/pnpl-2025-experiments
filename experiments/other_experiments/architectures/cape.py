#!/usr/bin/env python3
"""
CAPE: Context-Aware Phoneme Embeddings for LibriBrain Competition
Properly handles session boundaries and sequential context
"""

import argparse
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import h5py
from torch.utils.data import Dataset, DataLoader, Sampler
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.strategies import DDPStrategy
from torchmetrics import F1Score, Accuracy

import wandb
from lightning.pytorch.loggers import WandbLogger, CSVLogger, TensorBoardLogger
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.panel import Panel
from tqdm import tqdm

from pnpl.datasets import LibriBrainPhoneme, GroupedDataset, LibriBrainCompetitionHoldout

console = Console()

"""
python models/architectures/cape.py --data-path <DATA_ROOT>/libribrain/data/     --batch-size 64     --epochs 50     --use-context     --grouped-samples 1     --lr 1e-4     --wandb-name "CAPE_082925v0"     --gpus 1     --seed 42     --holdout-path <DATA_ROOT>/libribrain/data/COMPETITION_HOLDOUT/derivatives/serialised/sub-0_ses-2025_task-COMPETITION_HOLDOUT_run-1_proc-bads+headpos+sss+notch+bp+ds_meg.h5      --output-dir <MODEL_ROOT>/models/cape_082925v0     --submission-dir <DATA_ROOT>/submissions
"""


class SessionAwarePhonemeDataset(Dataset):
    """Dataset that properly handles session boundaries for context-aware training"""
    
    def __init__(self, 
                 data_path: str,
                 partition: str = 'train',
                 use_context: bool = False,
                 context_type: str = 'diphone',  # 'diphone' or 'triphone'
                 grouped_samples: int = 1,
                 tmin: float = 0.0,
                 tmax: float = 0.5):
        
        self.data_path = Path(data_path)
        self.partition = partition
        self.use_context = use_context and grouped_samples == 1
        self.context_type = context_type
        self.grouped_samples = grouped_samples
        
        # Load base dataset
        self.base_dataset = LibriBrainPhoneme(
            data_path=data_path,
            partition=partition,
            tmin=tmin,
            tmax=tmax
        )
        
        # Apply grouping if needed
        if grouped_samples > 1:
            self.dataset = GroupedDataset(
                self.base_dataset, 
                grouped_samples=grouped_samples,
                average_grouped_samples=True
            )
            self.use_context = False  # Disable context for grouped data
        else:
            self.dataset = self.base_dataset
        
        # Build session boundaries from the samples attribute
        self.session_info = self._build_session_info()
        
        # Build valid indices for context mode
        if self.use_context:
            self.valid_indices = self._build_valid_context_indices()
        else:
            self.valid_indices = list(range(len(self.dataset)))

    def _build_session_info(self):
        """Build a mapping of which samples belong to which session"""
        session_info = {}
        current_session = None
        session_start = 0
        
        # The base dataset stores samples as (subject, session, task, run, onset, label)
        for i, sample in enumerate(self.dataset.samples):
            session_key = (sample[0], sample[1], sample[2], sample[3])  # (subject, session, task, run)
            
            if session_key != current_session:
                if current_session is not None:
                    # Store the end of the previous session
                    session_info[current_session]['end'] = i
                
                # Start new session
                current_session = session_key
                session_info[session_key] = {
                    'start': i,
                    'end': None
                }
        
        # Don't forget the last session
        if current_session is not None:
            session_info[current_session]['end'] = len(self.dataset.samples)
        
        return session_info
    
    def _build_session_boundaries(self):
        """Identify where sessions start and end"""
        # This would need actual implementation based on dataset structure
        # For now, assume continuous data
        boundaries = [0, len(self.dataset)]
        return boundaries
    
    def _get_session_for_index(self, idx):
        """Find which session an index belongs to"""
        for session_key, bounds in self.session_info.items():
            if bounds['start'] <= idx < bounds['end']:
                return session_key
        return None
    
    def _crosses_boundary(self, start_idx, end_idx):
        """Check if indices span multiple sessions"""
        start_session = self._get_session_for_index(start_idx)
        end_session = self._get_session_for_index(end_idx)
        return start_session != end_session
    
    def _build_valid_context_indices(self):
        """Build list of indices that can be used as context centers"""
        valid = []
        
        if self.context_type == 'diphone':
            # Can't use last index of each session
            for session_key, bounds in self.session_info.items():
                # All indices except the last one in the session
                for i in range(bounds['start'], bounds['end'] - 1):
                    valid.append(i)
        
        elif self.context_type == 'triphone':
            # Can't use first or last index of each session
            for session_key, bounds in self.session_info.items():
                # Skip first and last in each session
                for i in range(bounds['start'] + 1, bounds['end'] - 1):
                    valid.append(i)
        
        return valid
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        
        if not self.use_context:
            # Single sample mode
            data, label = self.dataset[actual_idx]
            return {'meg': data, 'label': label}
        
        if self.context_type == 'diphone':
            # Current and next phoneme (guaranteed to be in same session)
            curr_data, curr_label = self.dataset[actual_idx]
            next_data, next_label = self.dataset[actual_idx + 1]
            
            return {
                'curr_meg': curr_data,
                'next_meg': next_data,
                'curr_label': curr_label,
                'next_label': next_label,
                'diphone_label': curr_label * 39 + next_label
            }
        
        elif self.context_type == 'triphone':
            # Previous, current, and next (guaranteed to be in same session)
            prev_data, prev_label = self.dataset[actual_idx - 1]
            curr_data, curr_label = self.dataset[actual_idx]
            next_data, next_label = self.dataset[actual_idx + 1]
            
            return {
                'prev_meg': prev_data,
                'curr_meg': curr_data,
                'next_meg': next_data,
                'prev_label': prev_label,
                'curr_label': curr_label,
                'next_label': next_label
            }

class SequentialBatchSampler(Sampler):
    """Sampler that ensures sequential batches for context learning"""
    
    def __init__(self, data_source, batch_size, shuffle_chunks=True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle_chunks = shuffle_chunks
        
    def __iter__(self):
        # Create chunks of sequential data
        n_samples = len(self.data_source)
        chunk_size = self.batch_size * 10  # Process in larger sequential chunks
        
        chunks = []
        for i in range(0, n_samples, chunk_size):
            chunk = list(range(i, min(i + chunk_size, n_samples)))
            chunks.append(chunk)
        
        # Optionally shuffle chunks (but keep sequences within chunks)
        if self.shuffle_chunks:
            np.random.shuffle(chunks)
        
        # Yield batches from chunks
        for chunk in chunks:
            for i in range(0, len(chunk), self.batch_size):
                batch = chunk[i:i + self.batch_size]
                if len(batch) == self.batch_size:  # Only full batches
                    yield batch
    
    def __len__(self):
        return len(self.data_source) // self.batch_size


class PhonemeEncoder(nn.Module):
    """Simplified encoder that actually works"""
    
    def __init__(self, n_channels=306, embedding_dim=256, hidden_dim=512,
                 n_layers=4, dropout=0.2):
        super().__init__()
        
        # Initial projection
        self.input_conv = nn.Sequential(
            nn.Conv1d(n_channels, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            self._make_residual_block(hidden_dim, dropout)
            for _ in range(n_layers)
        ])
        
        # Temporal aggregation
        self.temporal_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=dropout, batch_first=False
        )
        
        # Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def _make_residual_block(self, channels, dropout):
        return nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.BatchNorm1d(channels)
        )
    
    def forward(self, x):
        # x: (batch, channels, time)
        x = self.input_conv(x)
        
        # Residual blocks
        for block in self.blocks:
            residual = x
            x = F.relu(block(x) + residual)
        
        # Self-attention over time
        x_att = x.permute(2, 0, 1)  # (time, batch, channels)
        x_att, _ = self.temporal_attention(x_att, x_att, x_att)
        x = x + x_att.permute(1, 2, 0)
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)
        
        # Output projection
        embedding = self.output_proj(x)
        
        return F.normalize(embedding, dim=-1)


class CAPE(L.LightningModule):
    """Context-Aware Phoneme Embeddings model"""
    
    def __init__(self, 
                 n_channels=306,
                 n_classes=39,
                 n_diphones=1521,  # 39*39
                 embedding_dim=256,
                 hidden_dim=512,
                 n_layers=4,
                 dropout=0.2,
                 temperature=0.07,
                 context_weight=0.3,
                 lr=1e-4,
                 weight_decay=1e-5,
                 warmup_epochs=5,
                 max_epochs=100,
                 use_context=False,
                 context_type='diphone'):
        
        super().__init__()
        self.save_hyperparameters()
        
        # Encoder
        self.encoder = PhonemeEncoder(
            n_channels=n_channels,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout
        )
        
        # Classification heads
        self.phoneme_classifier = nn.Linear(embedding_dim, n_classes)
        
        if use_context and context_type == 'diphone':
            self.diphone_classifier = nn.Linear(embedding_dim * 2, n_diphones)
        
        # Losses
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Metrics
        self.train_f1 = F1Score(task='multiclass', num_classes=n_classes, average='macro')
        self.val_f1 = F1Score(task='multiclass', num_classes=n_classes, average='macro')
        
    def forward(self, batch):
        if 'curr_meg' in batch and 'next_meg' in batch and self.hparams.context_type == 'diphone':
            # Diphone mode
            curr_emb = self.encoder(batch['curr_meg'])
            next_emb = self.encoder(batch['next_meg'])
            
            # Classify individual phonemes
            curr_logits = self.phoneme_classifier(curr_emb)
            next_logits = self.phoneme_classifier(next_emb)
            
            # Classify diphone
            diphone_emb = torch.cat([curr_emb, next_emb], dim=1)
            diphone_logits = self.diphone_classifier(diphone_emb)
            
            return {
                'curr_logits': curr_logits,
                'next_logits': next_logits,
                'diphone_logits': diphone_logits,
                'embeddings': curr_emb
            }
        
        elif 'prev_meg' in batch and 'curr_meg' in batch and 'next_meg' in batch:
            # Triphone mode
            prev_emb = self.encoder(batch['prev_meg'])
            curr_emb = self.encoder(batch['curr_meg'])
            next_emb = self.encoder(batch['next_meg'])
            
            # Context-aware embedding
            context_emb = (prev_emb + curr_emb + next_emb) / 3
            
            logits = self.phoneme_classifier(context_emb)
            
            return {
                'logits': logits,
                'embeddings': context_emb
            }
        
        else:
            # Single mode
            meg = batch.get('meg', batch)
            embeddings = self.encoder(meg)
            logits = self.phoneme_classifier(embeddings)
            
            return {
                'logits': logits,
                'embeddings': embeddings
            }
    
    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        
        if 'diphone_logits' in outputs:
            # Diphone training
            curr_loss = self.ce_loss(outputs['curr_logits'], batch['curr_label'])
            next_loss = self.ce_loss(outputs['next_logits'], batch['next_label'])
            diphone_loss = self.ce_loss(outputs['diphone_logits'], batch['diphone_label'])
            
            loss = (curr_loss + next_loss) / 2 + self.hparams.context_weight * diphone_loss
            
            # Track metrics on current phoneme
            preds = torch.argmax(outputs['curr_logits'], dim=1)
            self.train_f1(preds, batch['curr_label'])
            
        else:
            # Standard training
            labels = batch.get('curr_label', batch.get('label'))
            loss = self.ce_loss(outputs['logits'], labels)
            
            preds = torch.argmax(outputs['logits'], dim=1)
            self.train_f1(preds, labels)
        
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/f1_macro', self.train_f1, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        labels = batch['label']
        
        loss = self.ce_loss(outputs['logits'], labels)
        preds = torch.argmax(outputs['logits'], dim=1)
        
        self.val_f1(preds, labels)
        
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/f1_macro', self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.hparams.warmup_epochs / self.hparams.max_epochs,
            anneal_strategy='cos'
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }


def evaluate_on_holdout(model, holdout_path, output_dir, batch_size=32, device='cuda'):
    """Evaluate model on competition holdout"""
    console.print(Panel("[bold cyan]Evaluating on Competition Holdout[/bold cyan]"))
    
    # Load holdout dataset
    data_path = str(Path(holdout_path).parent.parent.parent.parent)
    holdout_dataset = LibriBrainCompetitionHoldout(
        data_path=data_path,
        task="phoneme"
    )
    
    dataloader = DataLoader(
        holdout_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    model = model.to(device)
    model.eval()
    
    predictions = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Processing holdout...", total=len(dataloader))
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(device)
                outputs = model({'meg': batch})
                probs = torch.softmax(outputs['logits'], dim=1)
                
                for prob in probs.cpu():
                    predictions.append(prob)
                
                progress.update(task, advance=1)
    
    # Generate submission
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_file = Path(output_dir) / f"cape_submission_{timestamp}.csv"
    holdout_dataset.generate_submission_in_csv(predictions, str(submission_file))
    
    console.print(f"[green]✓[/green] Submission saved to {submission_file}")
    console.print(f"[green]✓[/green] Total predictions: {len(predictions)}")


def main():
    parser = argparse.ArgumentParser(description='CAPE: Context-Aware Phoneme Embeddings')
    
    # Data arguments
    parser.add_argument('--data-path', type=str, required=True, help='Path to LibriBrain data')
    parser.add_argument('--holdout-path', type=str, 
                       default='<DATA_ROOT>/libribrain/data/COMPETITION_HOLDOUT/derivatives/serialised/sub-0_ses-2025_task-COMPETITION_HOLDOUT_run-1_proc-bads+headpos+sss+notch+bp+ds_meg.h5',
                       help='Path to holdout MEG file')
    
    # Training strategy
    parser.add_argument('--use-context', action='store_true', help='Use context-aware training')
    parser.add_argument('--context-type', choices=['diphone', 'triphone'], default='diphone')
    parser.add_argument('--grouped-samples', type=int, default=1, 
                       help='Number of samples to average (1=no grouping, 100=match holdout)')
    
    # Output directories
    parser.add_argument('--output-dir', type=str, default='./outputs')
    parser.add_argument('--checkpoint-dir', type=str, default=None)
    parser.add_argument('--log-dir', type=str, default=None)
    parser.add_argument('--submission-dir', type=str, default=None)
    
    # Model arguments
    parser.add_argument('--embedding-dim', type=int, default=256)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--context-weight', type=float, default=0.3)
    
    # Other arguments
    parser.add_argument('--checkpoint', type=str, help='Resume from checkpoint')
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpus', type=int, default=1)
    
    # W&B arguments
    parser.add_argument('--wandb-project', type=str, default='libribrain-phoneme-classification')
    parser.add_argument('--wandb-entity', type=str, default='september-labs')
    parser.add_argument('--wandb-name', type=str, default='CAPE_082925v0')
    parser.add_argument('--no-wandb', action='store_true')
    
    args = parser.parse_args()
    
    # Set seed
    L.seed_everything(args.seed)
    
    # Setup directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else output_dir / 'checkpoints'
    log_dir = Path(args.log_dir) if args.log_dir else output_dir / 'logs'
    submission_dir = Path(args.submission_dir) if args.submission_dir else output_dir / 'submissions'
    
    for dir_path in [checkpoint_dir, log_dir, submission_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Display configuration
    console.print(Panel("[bold green]CAPE Configuration[/bold green]"))
    
    config_table = Table(title="Settings")
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="magenta")
    
    config_table.add_row("Context Mode", "Enabled" if args.use_context else "Disabled")
    config_table.add_row("Context Type", args.context_type if args.use_context else "N/A")
    config_table.add_row("Grouped Samples", str(args.grouped_samples))
    config_table.add_row("Batch Size", str(args.batch_size))
    config_table.add_row("Learning Rate", str(args.lr))
    config_table.add_row("Epochs", str(args.epochs))
    config_table.add_row("Output Dir", str(output_dir))
    
    console.print(config_table)
    
    # Initialize model
    model = CAPE(
        n_classes=39,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
        context_weight=args.context_weight,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
        use_context=args.use_context,
        context_type=args.context_type
    )
    
    if args.checkpoint:
        model = CAPE.load_from_checkpoint(args.checkpoint)
    
    if args.evaluate_only:
        evaluate_on_holdout(model, args.holdout_path, submission_dir)
        return
    
    # Prepare datasets
    console.print("[cyan]Loading datasets...[/cyan]")
    
    train_dataset = SessionAwarePhonemeDataset(
        data_path=args.data_path,
        partition='train',
        use_context=args.use_context,
        context_type=args.context_type,
        grouped_samples=args.grouped_samples
    )
    
    val_dataset = SessionAwarePhonemeDataset(
        data_path=args.data_path,
        partition='validation',
        use_context=False,  # Always single mode for validation
        grouped_samples=args.grouped_samples
    )
    
    # Create data loaders
    if args.use_context and args.grouped_samples == 1:
        # Use sequential sampling for context mode
        train_sampler = SequentialBatchSampler(
            train_dataset, 
            args.batch_size,
            shuffle_chunks=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=8
        )
    else:
        # Standard random sampling
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=4
    )
    
    console.print(f"[green]Train samples: {len(train_dataset)}[/green]")
    console.print(f"[green]Val samples: {len(val_dataset)}[/green]")
    
    # Setup loggers
    loggers = []
    
    if not args.no_wandb:
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            save_dir=str(log_dir)
        )
        loggers.append(wandb_logger)
    
    csv_logger = CSVLogger(
        save_dir=str(log_dir),
        name='cape'
    )
    loggers.append(csv_logger)
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename='cape-{epoch:02d}-{val/f1_macro:.3f}',
            monitor='val/f1_macro',
            mode='max',
            save_top_k=3,
            save_last=True
        ),
        EarlyStopping(
            monitor='val/f1_macro',
            mode='max',
            patience=15,
            verbose=True
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    
    # Setup strategy for multi-GPU
    if args.gpus > 1:
        strategy = DDPStrategy(find_unused_parameters=True)
    else:
        strategy = 'auto'
    
    # Train
    trainer = L.Trainer(
        max_epochs=args.epochs,
        devices=args.gpus,
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        strategy=strategy,
        precision=16,
        gradient_clip_val=1.0,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=50,
        val_check_interval=0.25,
        default_root_dir=str(output_dir)
    )
    
    console.print(Panel("[bold green]Starting Training[/bold green]"))
    trainer.fit(model, train_loader, val_loader)
    
    # Evaluate on holdout
    console.print(Panel("[bold cyan]Final Evaluation[/bold cyan]"))
    best_model_path = trainer.checkpoint_callback.best_model_path
    model = CAPE.load_from_checkpoint(best_model_path)
    evaluate_on_holdout(model, args.holdout_path, submission_dir)
    
    console.print(Panel("[bold green]Training Complete![/bold green]"))


if __name__ == "__main__":
    main()