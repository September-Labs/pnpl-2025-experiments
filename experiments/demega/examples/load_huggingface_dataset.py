#!/usr/bin/env python3
"""
Example: Loading LibriBrain MEG Preprocessed Dataset from HuggingFace

This script demonstrates how to download and load the preprocessed MEG dataset
(h5 files) from HuggingFace for training the DeMEGa classifier.

IMPORTANT: This dataset contains binary HDF5 (.h5) files, NOT standard datasets.
You must use snapshot_download() to download the files, then load them with
PNPL's GroupedDataset.

Dataset: https://huggingface.co/datasets/wordcab/libribrain-meg-preprocessed
"""

import os
import sys
from pathlib import Path
from pnpl.datasets import GroupedDataset
from torch.utils.data import DataLoader
import torch

# Add parent directory to path for meg_classifier imports
sys.path.append(str(Path(__file__).parent.parent))
from meg_classifier.models import SingleStageMEGClassifier


def download_from_huggingface(grouping_level=100, local_dir="./data"):
    """
    Download the LibriBrain MEG h5 files from HuggingFace.

    Args:
        grouping_level: Number of samples grouped (5, 10, 15, 20, 25, 30, 35, 45, 50, 55, 60, 100)
        local_dir: Directory to save downloaded h5 files

    Returns:
        Path to the downloaded data directory
    """
    from huggingface_hub import snapshot_download

    print(f"Downloading LibriBrain MEG h5 files from HuggingFace...")
    print(f"Grouping level: {grouping_level} samples")
    print(f"Download directory: {local_dir}")

    # Download h5 files using snapshot_download
    # This downloads only the specified grouping level to save space
    local_path = snapshot_download(
        repo_id="wordcab/libribrain-meg-preprocessed",
        repo_type="dataset",
        allow_patterns=[f"data/grouped_{grouping_level}/**"],
        local_dir=local_dir
    )

    print(f"Download complete! Files saved to {local_path}")
    return Path(local_dir) / f"grouped_{grouping_level}"


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


def compare_grouping_levels():
    """
    Compare different grouping levels to help choose the best one.
    """
    print("Comparing different grouping levels:\n")
    print("Level | Train Groups | Val Groups | Test Groups | Approx. Download Size")
    print("-" * 75)

    grouping_options = [
        (5, "~47 GB"),
        (10, "~24 GB"),
        (20, "~12 GB"),
        (50, "~4.7 GB"),
        (100, "~2.4 GB")
    ]

    for level, size in grouping_options:
        # Approximate group counts (based on grouping raw samples)
        print(f"{level:5} | {45600//level:>12} | {425//level:>10} | "
              f"{456//level:>11} | {size:>20}")

    print("\nRecommendation:")
    print("- Use grouped_100 for quick experiments (smallest size, fastest loading)")
    print("- Use grouped_20 for better accuracy with reasonable size")
    print("- Use grouped_5 for maximum accuracy (requires significant storage)")
    print("\nNote: Lower grouping levels = more data points but larger file sizes")


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
    Example usage: Download and load h5 files from HuggingFace.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Download and load LibriBrain MEG h5 files from HuggingFace"
    )
    parser.add_argument('--grouping_level', type=int, default=100,
                        choices=[5, 10, 15, 20, 25, 30, 35, 45, 50, 55, 60, 100],
                        help='Number of samples grouped together')
    parser.add_argument('--local_dir', type=str, default='./data',
                        help='Directory to download/load h5 files')
    parser.add_argument('--download', action='store_true',
                        help='Force download even if files exist')
    parser.add_argument('--compare', action='store_true',
                        help='Compare different grouping levels')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for DataLoader')

    args = parser.parse_args()

    if args.compare:
        compare_grouping_levels()
        return

    # Check if files exist locally
    h5_dir = Path(args.local_dir) / f"grouped_{args.grouping_level}"
    files_exist = (
        (h5_dir / "train_grouped.h5").exists() and
        (h5_dir / "validation_grouped.h5").exists() and
        (h5_dir / "test_grouped.h5").exists()
    )

    # Download if needed
    if not files_exist or args.download:
        print("Downloading h5 files from HuggingFace...")
        data_dir = download_from_huggingface(args.grouping_level, args.local_dir)
    else:
        print(f"Using existing h5 files from {h5_dir}")
        data_dir = h5_dir

    # Load datasets from h5 files
    print("\nLoading datasets from h5 files...")
    datasets = load_preprocessed_locally(
        args.local_dir,
        args.grouping_level,
        load_to_memory=True
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