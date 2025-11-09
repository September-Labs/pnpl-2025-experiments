#!/usr/bin/env python3
"""
Preprocess LibriBrain datasets for fast training.
Run this once to create preprocessed grouped datasets.
"""

import os
import argparse
import yaml
from pathlib import Path
import time
import random
import numpy as np

from pnpl.datasets import LibriBrainPhoneme, GroupedDataset
from pnpl.datasets.libribrain2025.constants import RUN_KEYS


def split_run_keys(train_ratio=80, val_ratio=10, test_ratio=10, seed=42):
    """
    Randomly split run keys into train/val/test sets.
    
    Args:
        train_ratio: Percentage of run keys for training
        val_ratio: Percentage of run keys for validation
        test_ratio: Percentage of run keys for testing
        seed: Random seed for reproducibility
        
    Returns:
        dict with 'train', 'validation', and 'test' run keys
    """
    if train_ratio + val_ratio + test_ratio != 100:
        raise ValueError("Split ratios must sum to 100")
    
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Get all run keys and shuffle them
    all_run_keys = list(RUN_KEYS)
    random.shuffle(all_run_keys)
    
    # Calculate split indices
    n_total = len(all_run_keys)
    n_train = int(n_total * train_ratio / 100)
    n_val = int(n_total * val_ratio / 100)
    
    # Split the run keys
    train_keys = all_run_keys[:n_train]
    val_keys = all_run_keys[n_train:n_train + n_val]
    test_keys = all_run_keys[n_train + n_val:]
    
    print(f"\nRun key split (total: {n_total}):")
    print(f"  Train: {len(train_keys)} ({len(train_keys)/n_total*100:.1f}%)")
    print(f"  Validation: {len(val_keys)} ({len(val_keys)/n_total*100:.1f}%)")
    print(f"  Test: {len(test_keys)} ({len(test_keys)/n_total*100:.1f}%)")
    
    return {
        'train': train_keys,
        'validation': val_keys,
        'test': test_keys
    }


def preprocess_partition(config, partition, channel_means=None, channel_stds=None, custom_run_keys=None, averaging_group: int=5):
    """Preprocess a single partition and save to HDF5."""
    
    print(f"\n{'='*60}")
    print(f"Processing {partition} partition")
    print(f"{'='*60}")
    
    data_config = config['data']
    
    # Create base dataset
    print(f"Loading {partition} dataset...")
    start_time = time.time()
    
    if custom_run_keys is not None:
        # Use custom run keys for this partition
        base_dataset = LibriBrainPhoneme(
            data_path=data_config['data_path'],
            include_run_keys=custom_run_keys[partition],
            tmin=data_config['tmin'],
            tmax=data_config['tmax'],
            standardize=True,
            channel_means=channel_means,
            channel_stds=channel_stds,
        )
    elif partition == 'train' and data_config.get('use_subset', False):
        # Use subset for training if specified
        base_dataset = LibriBrainPhoneme(
            data_path=data_config['data_path'],
            include_run_keys=[("0", str(i), "Sherlock1", "1") for i in range(1, 10)],
            tmin=data_config['tmin'],
            tmax=data_config['tmax'],
            standardize=True,
        )
    else:
        # Use default partition
        base_dataset = LibriBrainPhoneme(
            data_path=data_config['data_path'],
            partition=partition,
            tmin=data_config['tmin'],
            tmax=data_config['tmax'],
            standardize=True,
            channel_means=channel_means,
            channel_stds=channel_stds,
        )
    
    # Extract standardization params from training set
    if partition == 'train' and channel_means is None:
        channel_means = base_dataset.channel_means
        channel_stds = base_dataset.channel_stds
        print(f"Extracted standardization parameters from training set")
    
    print(f"Base dataset size: {len(base_dataset)} samples")
    print(f"Time to load: {time.time() - start_time:.2f}s")
    
    # Determine output path
    output_dir = Path(data_config.get('preprocessed_dir', './preprocessed_data'))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{partition}_grouped.h5"
    
    # Preprocess and save
    GroupedDataset.preprocess_and_save(
        base_dataset,
        output_path,
        grouped_samples=data_config.get('grouped_samples', averaging_group), # replace with 100, 95, etc.
        shuffle=True if partition == 'train' else False,
        average_grouped_samples=data_config.get('use_signal_averaging', True),
        shuffle_seed=data_config.get('shuffle_seed', 777),
        batch_size=256,  # Increased from 16 for better performance
        num_workers=40  # Use parallel processing with 40 workers
    )
    
    return channel_means, channel_stds, output_path


def main(args):
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override output directory if provided via CLI
    if args.output_dir:
        config['data']['preprocessed_dir'] = args.output_dir
    
    print("Preprocessing configuration:")
    data_config = config['data']
    print(f"  Data path: {data_config['data_path']}")
    print(f"  Time window: {data_config['tmin']} - {data_config['tmax']}s")
    print(f"  Grouped samples: {data_config['grouped_samples']}")
    print(f"  Output dir: {data_config.get('preprocessed_dir', './preprocessed_data')}")
    
    # Determine if using custom splits
    custom_run_keys = None
    if args.custom_split:
        print(f"\nUsing custom split ratios: {args.train_ratio}/{args.val_ratio}/{args.test_ratio}")
        print(f"Random seed: {args.split_seed}")
        custom_run_keys = split_run_keys(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.split_seed
        )
        
        # Save split info for reproducibility
        split_info_file = Path(data_config.get('preprocessed_dir', './preprocessed_data')) / 'split_info.yaml'
        split_info = {
            'train_ratio': args.train_ratio,
            'val_ratio': args.val_ratio,
            'test_ratio': args.test_ratio,
            'seed': args.split_seed,
            'train_keys': [list(k) for k in custom_run_keys['train']],
            'val_keys': [list(k) for k in custom_run_keys['validation']],
            'test_keys': [list(k) for k in custom_run_keys['test']]
        }
        split_info_file.parent.mkdir(parents=True, exist_ok=True)
        with open(split_info_file, 'w') as f:
            yaml.dump(split_info, f)
        print(f"\nSplit info saved to: {split_info_file}")
    
    # Process partitions
    if args.partition:
        # Process single partition
        if args.partition == 'train':
            channel_means, channel_stds, _ = preprocess_partition(config, 'train', custom_run_keys=custom_run_keys, averaging_group=args.mean_group_window)
        else:
            # Need to get standardization from train first
            print("Loading training dataset to get standardization parameters...")
            if custom_run_keys:
                train_dataset = LibriBrainPhoneme(
                    data_path=data_config['data_path'],
                    include_run_keys=custom_run_keys['train'],
                    tmin=data_config['tmin'],
                    tmax=data_config['tmax'],
                    standardize=True,
                )
            else:
                train_dataset = LibriBrainPhoneme(
                    data_path=data_config['data_path'],
                    partition='train',
                    tmin=data_config['tmin'],
                    tmax=data_config['tmax'],
                    standardize=True,
                )
            channel_means = train_dataset.channel_means
            channel_stds = train_dataset.channel_stds
            del train_dataset
            
            preprocess_partition(config, args.partition, channel_means, channel_stds, custom_run_keys, args.mean_group_window)
    else:
        # Process all partitions
        print("\nProcessing all partitions...")
        
        # Train first to get standardization
        channel_means, channel_stds, train_path = preprocess_partition(config, 'train', custom_run_keys=custom_run_keys)
        
        # Then validation and test with train standardization
        val_path = preprocess_partition(config, 'validation', channel_means, channel_stds, custom_run_keys, args.mean_group_window)
        test_path = preprocess_partition(config, 'test', channel_means, channel_stds, custom_run_keys, args.mean_group_window)
        
        # Save paths for easy reference
        paths = {
            'train': str(train_path),
            'validation': str(val_path[2]),
            'test': str(test_path[2])
        }
        
        paths_file = Path(data_config.get('preprocessed_dir', './preprocessed_data')) / 'paths.yaml'
        with open(paths_file, 'w') as f:
            yaml.dump(paths, f)
        
        print(f"\nDataset paths saved to: {paths_file}")
    
    print("\nPreprocessing complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess LibriBrain datasets')
    parser.add_argument('config', type=str, help='Path to config.yaml')
    parser.add_argument('--partition', type=str, choices=['train', 'validation', 'test'],
                       help='Process only specific partition (default: all)')
    
    # Custom split arguments
    parser.add_argument('--custom-split', action='store_true',
                       help='Use custom train/val/test split ratios instead of default')
    parser.add_argument('--train-ratio', type=int, default=80,
                       help='Percentage of run keys for training (default: 80)')
    parser.add_argument('--val-ratio', type=int, default=10,
                       help='Percentage of run keys for validation (default: 10)')
    parser.add_argument('--test-ratio', type=int, default=10,
                       help='Percentage of run keys for testing (default: 10)')
    parser.add_argument('--split-seed', type=int, default=777,
                       help='Random seed for reproducible splits (default: 777)')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for preprocessed data (overrides config file setting)')
    parser.add_argument('--mean-group-window', type=int, default=5,
                       help='The groups length across which one wants the averaging to be done, best with a 5 step up to 100, 5, then 10 etc.')
    
    args = parser.parse_args()
    
    # Validate split ratios
    if args.custom_split:
        if args.train_ratio + args.val_ratio + args.test_ratio != 100:
            parser.error("Split ratios must sum to 100")
    
    main(args)

