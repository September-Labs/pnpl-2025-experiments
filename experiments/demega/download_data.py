#!/usr/bin/env python3
"""
Download preprocessed MEG h5 files from HuggingFace.

This script downloads the LibriBrain MEG preprocessed dataset (h5 files)
from HuggingFace Hub for offline use with the DeMEGa classifier.

Dataset: https://huggingface.co/datasets/wordcab/libribrain-meg-preprocessed
"""

import argparse
from pathlib import Path
from huggingface_hub import snapshot_download


def download_h5_files(grouping_level=100, output_dir="./data"):
    """
    Download h5 files for a specific grouping level.

    Args:
        grouping_level: Number of samples grouped (5, 10, 15, 20, 25, 30, 35, 45, 50, 55, 60, 100)
        output_dir: Directory to save downloaded files

    Returns:
        Path to downloaded files
    """
    print(f"Downloading LibriBrain MEG preprocessed dataset from HuggingFace...")
    print(f"Grouping level: {grouping_level}")
    print(f"Output directory: {output_dir}")
    print()

    # Check if files already exist
    h5_dir = Path(output_dir) / f"grouped_{grouping_level}"
    train_h5 = h5_dir / "train_grouped.h5"
    val_h5 = h5_dir / "validation_grouped.h5"
    test_h5 = h5_dir / "test_grouped.h5"

    if all([train_h5.exists(), val_h5.exists(), test_h5.exists()]):
        print(f"WARNING: Files already exist at {h5_dir}")
        response = input("Do you want to re-download? (y/N): ")
        if response.lower() != 'y':
            print("Skipping download. Using existing files.")
            return h5_dir

    # Download from HuggingFace
    print(f"Downloading from wordcab/libribrain-meg-preprocessed...")
    print(f"This may take a while depending on your internet speed.")
    print()

    try:
        local_path = snapshot_download(
            repo_id="wordcab/libribrain-meg-preprocessed",
            repo_type="dataset",
            allow_patterns=[f"data/grouped_{grouping_level}/**"],
            local_dir=output_dir
        )

        print()
        print(f"✓ Download complete!")
        print(f"Files saved to: {h5_dir}")
        print()
        print("Expected files:")
        print(f"  - {train_h5}")
        print(f"  - {val_h5}")
        print(f"  - {test_h5}")
        print()
        print("You can now train using:")
        print(f"  python scripts/train.py --use_huggingface --grouping_level {grouping_level} --local_dir {output_dir}")

        return h5_dir

    except Exception as e:
        print(f"✗ Download failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Download LibriBrain MEG preprocessed h5 files from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download grouped_100 (2.4 GB - recommended for quick experiments)
  python download_data.py --grouping_level 100 --output_dir ./data

  # Download grouped_20 (12 GB - better accuracy)
  python download_data.py --grouping_level 20 --output_dir ./data

Available grouping levels and sizes:
  grouped_5:   ~47 GB (maximum data points, highest accuracy)
  grouped_10:  ~24 GB
  grouped_20:  ~12 GB
  grouped_50:  ~4.7 GB
  grouped_100: ~2.4 GB (minimum size, fastest loading)
        """
    )

    parser.add_argument(
        "--grouping_level",
        type=int,
        default=100,
        choices=[5, 10, 15, 20, 25, 30, 35, 45, 50, 55, 60, 100],
        help="Number of samples grouped together (default: 100)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Output directory for downloaded files (default: ./data)"
    )

    args = parser.parse_args()

    # Download files
    result = download_h5_files(args.grouping_level, args.output_dir)

    if result:
        print("Download successful!")
        return 0
    else:
        print("Download failed.")
        return 1


if __name__ == "__main__":
    exit(main())