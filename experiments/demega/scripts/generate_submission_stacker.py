#!/usr/bin/env python3
"""
Generate Submission with Stacking Model

This script generates competition submissions using a trained stacking model by:
1. Loading base models from Stage 1
2. Generating predictions (logits) from base models on holdout data
3. Feeding concatenated logits to the stacking model
4. Saving predictions to CSV for submission
"""

import argparse
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from meg_classifier.models import SingleStageMEGClassifier
from scripts.train_stacker import StackingClassifier, patched_predict_step
from pnpl.datasets import LibriBrainCompetitionHoldout, GroupedDataset


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    print("="*80)
    print("STACKING MODEL SUBMISSION GENERATION")
    print("="*80)
    print(f"Config: {args.config}")
    print(f"Base model checkpoints: {args.base_model_checkpoints}")
    print(f"Stacker checkpoint: {args.stacker_checkpoint}")
    print(f"Output path: {args.output_path}")
    print("="*80 + "\n")

    # Load config
    config = load_config(args.config)
    data_config = config['data']

    # Parse base model checkpoints
    base_model_ckpts = args.base_model_checkpoints.split(',')
    base_data_dirs = args.base_data_dirs.split(',')

    print(f"Number of base models: {len(base_model_ckpts)}")
    print(f"Preprocessing methods: {base_data_dirs}")

    if len(base_model_ckpts) != len(base_data_dirs):
        raise ValueError("Number of base models must match number of preprocessing directories")

    # Load holdout dataset
    print("\n" + "="*80)
    print("LOADING HOLDOUT DATA")
    print("="*80)

    holdout_dataset = LibriBrainCompetitionHoldout(
        data_path=data_config['data_path'],
        task='phoneme',
        tmin=data_config['tmin'],
        tmax=data_config['tmax'],
        standardize=False
    )

    print(f"Holdout dataset loaded: {len(holdout_dataset)} segments")
    print(f"Segment shape: {holdout_dataset[0].shape}")

    # Patch the model class with predict_step
    SingleStageMEGClassifier.predict_step = patched_predict_step

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = config['submission'].get('batch_size', 64)
    num_workers = config['submission'].get('num_workers', 8)

    # Generate predictions from each base model
    print("\n" + "="*80)
    print("GENERATING BASE MODEL PREDICTIONS")
    print("="*80)

    import lightning as L
    pred_trainer = L.Trainer(
        devices=1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        logger=False,
        precision="16-mixed" if torch.cuda.is_available() else 32
    )

    all_logits = []

    for i, (ckpt_path, data_dir) in enumerate(zip(base_model_ckpts, base_data_dirs), 1):
        print(f"\n[{i}/{len(base_model_ckpts)}] Processing with: {Path(ckpt_path).name}")
        print(f"Preprocessing: {data_dir}")

        # Apply grouped averaging based on the preprocessing method
        # Extract grouping number from directory name (e.g., "grouped50" -> 50)
        try:
            if 'grouped' in data_dir:
                grouped_samples = int(data_dir.split('grouped')[-1].split('_')[0])
            else:
                grouped_samples = 1
        except:
            print(f"Warning: Could not parse grouping from {data_dir}, using grouping=1")
            grouped_samples = 1

        print(f"Applying signal averaging: {grouped_samples} samples per group")

        # Create grouped dataset
        if grouped_samples > 1:
            processed_dataset = GroupedDataset(
                holdout_dataset,
                grouped_samples=grouped_samples,
                average_grouped_samples=True,
                shuffle_seed=data_config.get('shuffle_seed', 777)
            )
        else:
            processed_dataset = holdout_dataset

        print(f"Processed dataset size: {len(processed_dataset)} groups")

        # Create dataloader
        dataloader = DataLoader(
            processed_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        # Load model and generate predictions
        print(f"Loading model...")
        model = SingleStageMEGClassifier.load_from_checkpoint(ckpt_path)

        print(f"Generating predictions...")
        logits = pred_trainer.predict(model, dataloader)
        logits_cat = torch.cat(logits, dim=0)
        all_logits.append(logits_cat)

        print(f"  ✓ Logits shape: {logits_cat.shape}")
        del model  # Free memory

    # Concatenate all logits
    meta_X = torch.cat(all_logits, dim=1)
    print(f"\nMeta-features shape: {meta_X.shape}")

    # Load stacker model
    print("\n" + "="*80)
    print("LOADING STACKER MODEL")
    print("="*80)
    print(f"Checkpoint: {args.stacker_checkpoint}")

    stacker = StackingClassifier.load_from_checkpoint(args.stacker_checkpoint)
    stacker = stacker.to(device)
    stacker.eval()

    # Generate final predictions
    print("\n" + "="*80)
    print("GENERATING FINAL PREDICTIONS")
    print("="*80)

    meta_X = meta_X.to(device)
    # Convert to float32 to match stacker model dtype
    meta_X = meta_X.float()

    with torch.no_grad():
        final_logits = stacker(meta_X)

        if args.use_probabilities:
            # Generate probability distributions
            final_probs = torch.softmax(final_logits, dim=1)
            predictions = final_probs.cpu().numpy()
        else:
            # Generate class predictions
            final_preds = torch.argmax(final_logits, dim=1)
            predictions = final_preds.cpu().numpy()

    print(f"Predictions shape: {predictions.shape}")

    # Create submission file
    print("\n" + "="*80)
    print("CREATING SUBMISSION FILE")
    print("="*80)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.use_probabilities:
        # Create DataFrame with probability columns
        num_classes = predictions.shape[1]
        columns = [f'class_{i}' for i in range(num_classes)]
        df = pd.DataFrame(predictions, columns=columns)
        df.insert(0, 'id', range(len(df)))
    else:
        # Create DataFrame with predictions
        df = pd.DataFrame({
            'id': range(len(predictions)),
            'prediction': predictions
        })

    df.to_csv(output_path, index=False)
    print(f"✓ Submission saved to: {output_path}")
    print(f"  Total predictions: {len(df)}")

    # Display sample predictions
    print(f"\nSample predictions (first 10 rows):")
    print(df.head(10))

    # Auto-submit if requested
    if args.auto_submit:
        print("\n" + "="*80)
        print("AUTO-SUBMITTING TO COMPETITION")
        print("="*80)

        try:
            import subprocess

            submit_cmd = [
                'pnpl', 'submit',
                '--task', 'phoneme',
                '--submission-path', str(output_path)
            ]

            if args.no_wait:
                submit_cmd.append('--no-wait')

            print(f"Running: {' '.join(submit_cmd)}")
            result = subprocess.run(submit_cmd, check=True, capture_output=True, text=True)

            print("✓ Submission successful!")
            print(result.stdout)

        except subprocess.CalledProcessError as e:
            print(f"✗ Submission failed: {e}")
            print(e.stderr)
        except Exception as e:
            print(f"✗ Error during submission: {e}")

    print("\n" + "="*80)
    print("SUBMISSION GENERATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Competition Submission with Stacking Model"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )

    parser.add_argument(
        "--base_model_checkpoints",
        type=str,
        required=True,
        help="Comma-separated list of base model checkpoint paths (in same order as base_data_dirs)"
    )

    parser.add_argument(
        "--base_data_dirs",
        type=str,
        required=True,
        help="Comma-separated list of preprocessing methods (e.g., 'preprocessed_data_tmax0_5_grouped50,preprocessed_data_tmax0_5_grouped100')"
    )

    parser.add_argument(
        "--stacker_checkpoint",
        type=str,
        required=True,
        help="Path to stacker model checkpoint"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="./submissions/stacker_submission.csv",
        help="Path to save submission CSV file"
    )

    parser.add_argument(
        "--use_probabilities",
        action="store_true",
        help="Output probability distributions instead of class predictions"
    )

    parser.add_argument(
        "--auto_submit",
        action="store_true",
        help="Automatically submit to competition after generating predictions"
    )

    parser.add_argument(
        "--no_wait",
        action="store_true",
        help="Don't wait for evaluation results after submission"
    )

    args = parser.parse_args()
    main(args)
