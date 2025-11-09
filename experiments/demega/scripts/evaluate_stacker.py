#!/usr/bin/env python3
"""
Evaluate Stacking Model

This script evaluates a trained stacking model by:
1. Loading base models from Stage 1
2. Generating predictions (logits) from base models on validation/test data
3. Feeding concatenated logits to the stacking model
4. Computing performance metrics
5. Saving detailed evaluation reports
"""

import argparse
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
from torchmetrics import F1Score
import sys
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from meg_classifier.models import SingleStageMEGClassifier
from scripts.train_stacker import StackingClassifier, patched_predict_step
from scripts.preprocess_data.get_preprocess_dataloader import create_dataloaders_multi_preprocessed


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_partition(partition_name, eval_loader, base_model_ckpts, stacker, config, device):
    """Evaluate on a specific partition and return results."""
    print("\n" + "="*80)
    print(f"EVALUATING ON {partition_name.upper()} SET")
    print("="*80)
    print(f"Samples: {len(eval_loader.dataset)}")

    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']

    # Create prediction loader (unshuffled, larger batch size)
    pred_loader = DataLoader(
        eval_loader.dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Load base models and generate predictions
    print("\nGenerating base model predictions...")
    all_logits = []

    import lightning as L
    pred_trainer = L.Trainer(
        devices=1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        logger=False,
        precision="16-mixed" if torch.cuda.is_available() else 32
    )

    for i, ckpt_path in enumerate(base_model_ckpts, 1):
        print(f"  [{i}/{len(base_model_ckpts)}] {Path(ckpt_path).name}...", end=' ')
        model = SingleStageMEGClassifier.load_from_checkpoint(ckpt_path)

        logits = pred_trainer.predict(model, pred_loader)
        logits_cat = torch.cat(logits, dim=0)
        all_logits.append(logits_cat)

        print(f"✓ Shape: {logits_cat.shape}")
        del model  # Free memory

    # Concatenate all logits
    meta_X = torch.cat(all_logits, dim=1)
    print(f"\nMeta-features shape: {meta_X.shape}")

    # Get labels
    meta_y = torch.cat([y for _, y in pred_loader], dim=0)

    # Generate predictions from stacker
    print("Generating final predictions from stacker...")
    meta_X = meta_X.to(device)
    meta_y = meta_y.to(device)

    # Convert to float32 to match stacker model dtype
    meta_X = meta_X.float()

    with torch.no_grad():
        final_logits = stacker(meta_X)
        final_preds = torch.argmax(final_logits, dim=1)

    # Compute metrics
    num_classes = config['model']['params'].get('vocab_size', 39)

    # Overall F1 macro
    f1_metric = F1Score(
        task="multiclass",
        average="macro",
        num_classes=num_classes
    ).to(device)

    f1_macro = f1_metric(final_preds, meta_y).item()

    # Random baseline
    random_preds = torch.randint(0, num_classes, (len(meta_y),), device=device)
    random_f1_macro = f1_metric(random_preds, meta_y).item()

    # Per-class F1 scores
    binary_f1 = F1Score(task="binary").to(device)
    f1_by_class = []
    support_by_class = []

    for c in range(num_classes):
        class_preds = final_preds == c
        class_targets = meta_y == c
        class_f1 = binary_f1(class_preds, class_targets).item()
        support = class_targets.sum().item()
        f1_by_class.append(class_f1)
        support_by_class.append(support)

    # Prepare results dictionary
    results = {
        'partition': partition_name,
        'num_samples': len(meta_y),
        'num_classes': num_classes,
        'f1_macro': f1_macro,
        'random_f1_macro': random_f1_macro,
        'improvement': (f1_macro - random_f1_macro) / random_f1_macro * 100 if random_f1_macro > 0 else 0,
        'per_class_f1': f1_by_class,
        'per_class_support': support_by_class,
        'mean_f1': sum(f1_by_class) / len(f1_by_class),
        'std_f1': torch.tensor(f1_by_class).std().item(),
    }

    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS - {partition_name.upper()}")
    print(f"{'='*60}")
    print(f"Random F1-macro:      {random_f1_macro:.4f}")
    print(f"Stacker F1-macro:     {f1_macro:.4f}")
    print(f"Improvement:          {results['improvement']:.1f}%")
    print(f"Mean F1 (per-class):  {results['mean_f1']:.4f}")
    print(f"Std F1 (per-class):   {results['std_f1']:.4f}")

    # Show top and bottom classes
    sorted_indices = sorted(range(num_classes), key=lambda i: f1_by_class[i], reverse=True)

    print(f"\nTop 5 classes:")
    for rank, idx in enumerate(sorted_indices[:5], 1):
        print(f"  {rank}. Class {idx:2d}: F1={f1_by_class[idx]:.4f} (n={support_by_class[idx]})")

    print(f"\nBottom 5 classes:")
    for rank, idx in enumerate(sorted_indices[-5:], 1):
        print(f"  {rank}. Class {idx:2d}: F1={f1_by_class[idx]:.4f} (n={support_by_class[idx]})")

    return results


def save_report(results_val, results_test, output_dir, args):
    """Save detailed evaluation reports."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON report
    json_path = output_dir / f"evaluation_report_{timestamp}.json"
    report = {
        'timestamp': timestamp,
        'stacker_checkpoint': args.stacker_checkpoint,
        'base_model_checkpoints': args.base_model_checkpoints.split(','),
        'base_data_dirs': args.base_data_dirs.split(','),
        'validation': results_val,
        'test': results_test,
    }

    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n✓ JSON report saved: {json_path}")

    # Save text report
    txt_path = output_dir / f"evaluation_report_{timestamp}.txt"
    with open(txt_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("STACKING MODEL EVALUATION REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Stacker: {args.stacker_checkpoint}\n")
        f.write(f"Base models: {len(report['base_model_checkpoints'])}\n")
        f.write("\n")

        for partition_name, results in [('VALIDATION', results_val), ('TEST', results_test)]:
            f.write("="*80 + "\n")
            f.write(f"{partition_name} SET RESULTS\n")
            f.write("="*80 + "\n")
            f.write(f"Samples:              {results['num_samples']}\n")
            f.write(f"F1-macro:             {results['f1_macro']:.4f}\n")
            f.write(f"Random baseline:      {results['random_f1_macro']:.4f}\n")
            f.write(f"Improvement:          {results['improvement']:.1f}%\n")
            f.write(f"Mean F1 (per-class):  {results['mean_f1']:.4f}\n")
            f.write(f"Std F1 (per-class):   {results['std_f1']:.4f}\n")
            f.write("\n")

            # Per-class breakdown
            f.write("PER-CLASS F1 SCORES:\n")
            f.write("-"*60 + "\n")
            f.write(f"{'Class':<8} {'F1 Score':<12} {'Support':<10}\n")
            f.write("-"*60 + "\n")

            # Sort by F1 score
            sorted_indices = sorted(
                range(results['num_classes']),
                key=lambda i: results['per_class_f1'][i],
                reverse=True
            )

            for idx in sorted_indices:
                f1 = results['per_class_f1'][idx]
                support = results['per_class_support'][idx]
                f.write(f"{idx:<8} {f1:<12.4f} {support:<10}\n")
            f.write("\n")

    print(f"✓ Text report saved: {txt_path}")

    # Save CSV for easy analysis
    csv_path = output_dir / f"per_class_results_{timestamp}.csv"
    with open(csv_path, 'w') as f:
        f.write("class,val_f1,val_support,test_f1,test_support\n")
        for i in range(results_val['num_classes']):
            f.write(f"{i},{results_val['per_class_f1'][i]:.4f},{results_val['per_class_support'][i]},")
            f.write(f"{results_test['per_class_f1'][i]:.4f},{results_test['per_class_support'][i]}\n")

    print(f"✓ CSV results saved: {csv_path}")

    return json_path, txt_path, csv_path


def main(args):
    print("="*80)
    print("STACKING MODEL EVALUATION")
    print("="*80)
    print(f"Config: {args.config}")
    print(f"Stacker checkpoint: {args.stacker_checkpoint}")
    print(f"Base models: {len(args.base_model_checkpoints.split(','))}")
    print("="*80)

    # Load config
    config = load_config(args.config)

    # Parse base model checkpoints
    base_model_ckpts = [ckpt.strip() for ckpt in args.base_model_checkpoints.split(',')]
    base_data_dirs = [d.strip() for d in args.base_data_dirs.split(',')]

    # Load preprocessed data
    print("\nLoading data...")
    preprocessed_root = Path(args.preprocessed_root)

    train_loader, val_loader, test_loader = create_dataloaders_multi_preprocessed(
        config,
        preprocessed_root,
        base_data_dirs
    )

    print(f"✓ Validation set: {len(val_loader.dataset)} samples")
    print(f"✓ Test set: {len(test_loader.dataset)} samples")

    # Patch the model class with predict_step
    SingleStageMEGClassifier.predict_step = patched_predict_step

    # Load stacker model
    print("\nLoading stacker model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stacker = StackingClassifier.load_from_checkpoint(args.stacker_checkpoint)
    stacker = stacker.to(device)
    stacker.eval()
    print(f"✓ Stacker loaded on {device}")

    # Evaluate on both partitions
    results_val = evaluate_partition('validation', val_loader, base_model_ckpts, stacker, config, device)
    results_test = evaluate_partition('test', test_loader, base_model_ckpts, stacker, config, device)

    # Save reports
    print("\n" + "="*80)
    print("SAVING REPORTS")
    print("="*80)

    output_dir = Path(args.stacker_checkpoint).parent / "evaluation_reports"
    save_report(results_val, results_test, output_dir, args)

    # Print summary comparison
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    print(f"{'Metric':<30} {'Validation':<15} {'Test':<15}")
    print("-"*60)
    print(f"{'F1-macro':<30} {results_val['f1_macro']:<15.4f} {results_test['f1_macro']:<15.4f}")
    print(f"{'Random baseline':<30} {results_val['random_f1_macro']:<15.4f} {results_test['random_f1_macro']:<15.4f}")
    print(f"{'Improvement (%)':<30} {results_val['improvement']:<15.1f} {results_test['improvement']:<15.1f}")
    print(f"{'Mean per-class F1':<30} {results_val['mean_f1']:<15.4f} {results_test['mean_f1']:<15.4f}")

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Stacking Model")

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
        help="Comma-separated list of base model checkpoint paths"
    )

    parser.add_argument(
        "--stacker_checkpoint",
        type=str,
        required=True,
        help="Path to stacker model checkpoint"
    )

    parser.add_argument(
        "--preprocessed_root",
        type=str,
        required=True,
        help="Root directory where preprocessed data folders are located"
    )

    parser.add_argument(
        "--base_data_dirs",
        type=str,
        required=True,
        help="Comma-separated list of preprocessed directory names"
    )

    args = parser.parse_args()
    main(args)
