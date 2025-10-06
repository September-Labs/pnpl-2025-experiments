#!/usr/bin/env python3
"""
Example: Loading LibriBrain MEG Preprocessed Dataset from HuggingFace

This script demonstrates how to load and use the preprocessed MEG dataset
from HuggingFace for training the DeMEGa classifier.

Dataset: https://huggingface.co/datasets/wordcab/libribrain-meg-preprocessed
"""

import os
import sys
from pathlib import Path
from datasets import load_dataset
from pnpl.datasets import GroupedDataset
from torch.utils.data import DataLoader
import torch

# Add parent directory to path for meg_classifier imports
sys.path.append(str(Path(__file__).parent.parent))
from meg_classifier.models import SingleStageMEGClassifier


def load_from_huggingface(grouping_level=100, cache_dir="./data/hf_cache"):
    """
    Load the LibriBrain MEG dataset from HuggingFace.

    Args:
        grouping_level: Number of samples grouped (5, 10, 15, 20, 25, 30, 35, 45, 50, 55, 60, 100)
        cache_dir: Directory to cache downloaded data

    Returns:
        Dictionary with train, validation, and test datasets
    """
    print(f"Loading LibriBrain MEG dataset from HuggingFace...")
    print(f"Grouping level: {grouping_level} samples")
    print(f"Cache directory: {cache_dir}")

    # Load dataset from HuggingFace
    # This will download the data if not already cached
    dataset = load_dataset(
        "wordcab/libribrain-meg-preprocessed",
        f"grouped_{grouping_level}",
        cache_dir=cache_dir
    )

    print(f"Dataset loaded successfully!")
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Validation samples: {len(dataset['validation'])}")
    print(f"Test samples: {len(dataset['test'])}")

    return dataset


def load_preprocessed_locally(data_dir, grouping_level=100, load_to_memory=True):
    """
    Load preprocessed data from local files after downloading from HuggingFace.

    Args:
        data_dir: Directory containing the preprocessed .h5 files
        grouping_level: Number of samples grouped
        load_to_memory: Whether to load entire dataset to RAM

    Returns:
        Dictionary with train, validation, and test datasets
    """
    print(f"Loading preprocessed data from: {data_dir}")

    grouped_dir = Path(data_dir) / f"grouped_{grouping_level}"

    # Load datasets using PNPL
    train_dataset = GroupedDataset(
        preprocessed_path=grouped_dir / "train_grouped.h5",
        load_to_memory=load_to_memory
    )

    val_dataset = GroupedDataset(
        preprocessed_path=grouped_dir / "validation_grouped.h5",
        load_to_memory=load_to_memory
    )

    test_dataset = GroupedDataset(
        preprocessed_path=grouped_dir / "test_grouped.h5",
        load_to_memory=load_to_memory
    )

    print(f"Datasets loaded successfully!")
    print(f"Train groups: {len(train_dataset)}")
    print(f"Validation groups: {len(val_dataset)}")
    print(f"Test groups: {len(test_dataset)}")

    return {
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    }


def compare_grouping_levels(cache_dir="./data/hf_cache"):
    """
    Compare different grouping levels to help choose the best one.
    """
    print("Comparing different grouping levels:\n")
    print("Level | Train Size | Val Size | Test Size | Approx. Download")
    print("-" * 60)

    grouping_options = [
        (5, "47 GB"),
        (10, "24 GB"),
        (20, "12 GB"),
        (50, "4.7 GB"),
        (100, "2.4 GB")
    ]

    for level, size in grouping_options:
        # This would normally load the dataset, but we'll just show info
        print(f"{level:5} | {'~' + str(45600//level):10} | {'~' + str(425//level):8} | "
              f"{'~' + str(456//level):9} | {size:>10}")

    print("\nRecommendation:")
    print("- Use grouped_100 for quick experiments (smallest size, fastest loading)")
    print("- Use grouped_20 for better accuracy with reasonable size")
    print("- Use grouped_5 for maximum accuracy (requires significant storage)")


def create_dataloaders(dataset, batch_size=32, num_workers=4):
    """
    Create PyTorch DataLoaders from the datasets.

    Args:
        dataset: Dictionary with train, validation, test datasets
        batch_size: Batch size for training
        num_workers: Number of parallel data loading workers

    Returns:
        Dictionary with train, validation, test dataloaders
    """
    train_loader = DataLoader(
        dataset['train'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        dataset['validation'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        dataset['test'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return {
        'train': train_loader,
        'validation': val_loader,
        'test': test_loader
    }


def main():
    """
    Example usage of the HuggingFace dataset.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Load LibriBrain MEG dataset from HuggingFace")
    parser.add_argument('--grouping_level', type=int, default=100,
                        choices=[5, 10, 15, 20, 25, 30, 35, 45, 50, 55, 60, 100],
                        help='Number of samples grouped together')
    parser.add_argument('--cache_dir', type=str, default='./data/hf_cache',
                        help='Directory to cache HuggingFace datasets')
    parser.add_argument('--local_dir', type=str, default=None,
                        help='Local directory with preprocessed .h5 files')
    parser.add_argument('--compare', action='store_true',
                        help='Compare different grouping levels')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for DataLoader')

    args = parser.parse_args()

    if args.compare:
        compare_grouping_levels(args.cache_dir)
        return

    # Load dataset
    if args.local_dir:
        # Load from local preprocessed files
        datasets = load_preprocessed_locally(
            args.local_dir,
            args.grouping_level,
            load_to_memory=True
        )
    else:
        # Load from HuggingFace (will download if needed)
        datasets = load_from_huggingface(
            args.grouping_level,
            args.cache_dir
        )

    # Create DataLoaders
    dataloaders = create_dataloaders(datasets, batch_size=args.batch_size)

    # Example: Get a batch from training data
    print("\nGetting a sample batch...")
    for batch in dataloaders['train']:
        if isinstance(batch, dict):
            meg_data = batch['meg']
            labels = batch['phoneme']
        else:
            meg_data, labels = batch

        print(f"MEG data shape: {meg_data.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"MEG data type: {meg_data.dtype}")
        print(f"Labels type: {labels.dtype}")

        # Show label distribution in batch
        unique_labels = torch.unique(labels)
        print(f"Unique phonemes in batch: {len(unique_labels)}")
        print(f"Phoneme IDs: {unique_labels.tolist()[:10]}...")
        break

    print("\nDataset is ready for training!")
    print("Use these dataloaders with your training script.")


if __name__ == "__main__":
    main()