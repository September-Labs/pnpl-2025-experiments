import argparse
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
from torchmetrics import F1Score
from pnpl.datasets import LibriBrainPhoneme, GroupedDataset
import sys
import numpy as np
import matplotlib.pyplot as plt
import importlib

# Add parent directory to path for meg_classifier imports
sys.path.append(str(Path(__file__).parent.parent))
from meg_classifier.models import SingleStageMEGClassifier


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model_from_checkpoint(checkpoint_path, config):
    """Load model from checkpoint using config for dynamic model loading.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config: Configuration dictionary
        
    Returns:
        Loaded model instance
    """
    model_config = config['model']
    
    # Import the specified module
    module_path = model_config['module']
    class_name = model_config['class']
    
    try:
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Could not import {class_name} from {module_path}: {e}")
    
    print(f"Loading {class_name} from checkpoint: {checkpoint_path}")
    
    # Load from checkpoint with strict=False to handle dynamic layers
    model = model_class.load_from_checkpoint(checkpoint_path, strict=False)
    
    return model


def validate(val_loader, model, labels):
    """Validate model and compute metrics."""
    model.eval()
    predicted_phonemes = []
    true_phonemes = []
    
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            x = x.to(model.device)
            y = y.to(model.device)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            predicted_phonemes.extend(preds)
            true_phonemes.extend(y)
    
    true_phonemes = torch.stack(true_phonemes)
    predicted_phonemes = torch.stack(predicted_phonemes)
    
    # Calculate F1 macro score
    f1_macro = F1Score(task="multiclass", average="macro", 
                       num_classes=len(labels)).to(model.device)
    
    # Random baseline for comparison
    random_preds = torch.randint(
        0, len(labels), (len(true_phonemes),), device=model.device)
    
    random_f1_macro = f1_macro(random_preds, true_phonemes)
    model_f1_macro = f1_macro(predicted_phonemes, true_phonemes)
    
    # Calculate per-class F1 scores
    binary_f1 = F1Score(task="binary").to(model.device)
    classes = torch.arange(len(labels))
    f1_by_class = []
    random_f1_by_class = []
    
    for c in classes:
        class_preds = predicted_phonemes == c
        class_targets = true_phonemes == c
        class_f1 = binary_f1(class_preds, class_targets)
        
        class_random_preds = random_preds == c
        class_random_f1 = binary_f1(class_random_preds, class_targets)
        
        f1_by_class.append(class_f1)
        random_f1_by_class.append(class_random_f1)
    
    f1_by_class = torch.stack(f1_by_class)
    random_f1_by_class = torch.stack(random_f1_by_class)
    
    return model_f1_macro, random_f1_macro, f1_by_class, random_f1_by_class


def plot_class_specific_scores(scores, random_scores, metric_name, labels, save_path=None):
    """Plot per-class F1 scores."""
    num_classes = len(labels)
    
    # Sort by model performance
    order = torch.argsort(scores).flip(dims=[0])
    scores = scores[order]
    random_scores = random_scores[order]
    labels = [labels[i] for i in order]
    
    # Create figure
    x = np.arange(num_classes)
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Plot bars
    bars1 = ax.bar(x - width/2, random_scores.cpu(), width,
                    label='Random', color='skyblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, scores.cpu(), width,
                    label='Model', color='salmon', edgecolor='black')
    
    # Formatting
    ax.set_xlabel('Phonemes', fontsize=14)
    ax.set_ylabel(metric_name, fontsize=14)
    ax.set_title(f'{metric_name} for each Phoneme', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=12)
    ax.legend(fontsize=12)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def main(args):
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        # Try relative to meta directory
        config_path = Path(__file__).parent / "configs" / args.config
    
    if not config_path.exists():
        raise ValueError(f"Config file not found: {args.config}")
    
    config = load_config(config_path)
    
    # Get configurations
    eval_config = config['evaluation']
    data_config = config['data']
    
    # Load model from checkpoint
    model = load_model_from_checkpoint(args.checkpoint, config)
    model.eval()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Load dataset
    dataset = LibriBrainPhoneme(
        data_path=data_config['data_path'],
        partition=eval_config['partition'],
        tmin=data_config['tmin'],
        tmax=data_config['tmax'],
        standardize=True
    )
    
    # Store labels before wrapping with GroupedDataset
    labels_sorted = dataset.labels_sorted
    print(f"Labels sorted: {labels_sorted}")
    
    # Apply signal averaging if requested
    eval_dataset = dataset
    if eval_config['use_signal_averaging']:
        print(f"Applying signal averaging with {eval_config['grouped_samples']} samples per group")
        eval_dataset = GroupedDataset(
            dataset,
            grouped_samples=eval_config['grouped_samples'],
            average_grouped_samples=True
        )
    
    # Create dataloader
    dataloader = DataLoader(
        eval_dataset,
        batch_size=eval_config['batch_size'],
        shuffle=False,
        num_workers=eval_config['num_workers']
    )
    
    print(f"Evaluating on {eval_config['partition']} set with {len(eval_dataset)} samples")
    print(f"Using model: {config['model']['class']} from {config['model']['module']}")
    
    # Run evaluation
    f1_macro, random_f1_macro, f1_by_class, random_f1_by_class = validate(
        dataloader, model, labels_sorted
    )
    
    # Print overall results
    print(f"\n{'='*60}")
    print(f"OVERALL RESULTS:")
    print(f"{'='*60}")
    print(f"Random F1-macro: {random_f1_macro:.4f}")
    print(f"Model F1-macro: {f1_macro:.4f}")
    print(f"Improvement: {(f1_macro - random_f1_macro) / random_f1_macro * 100:.1f}%")
    
    # Print F1 scores for ALL phonemes
    print(f"\n{'='*60}")
    print(f"F1 SCORES FOR ALL PHONEMES (sorted by performance):")
    print(f"{'='*60}")
    print(f"{'Rank':<6} {'Phoneme':<10} {'Model F1':<12} {'Random F1':<12} {'Difference':<12}")
    print(f"{'-'*60}")
    
    # Sort phonemes by F1 score (descending)
    sorted_idx = torch.argsort(f1_by_class, descending=True)
    
    for rank, idx in enumerate(sorted_idx, 1):
        phoneme = labels_sorted[idx]
        model_f1 = f1_by_class[idx].item()
        random_f1 = random_f1_by_class[idx].item()
        diff = model_f1 - random_f1
        
        # Color coding for terminal output (optional - comment out if not needed)
        if diff > 0.1:
            # Good performance (green in terminals that support it)
            print(f"{rank:<6} {phoneme:<10} {model_f1:<12.4f} {random_f1:<12.4f} {diff:+.4f}")
        elif diff < -0.05:
            # Poor performance (red in terminals that support it)
            print(f"{rank:<6} {phoneme:<10} {model_f1:<12.4f} {random_f1:<12.4f} {diff:+.4f}")
        else:
            # Average performance
            print(f"{rank:<6} {phoneme:<10} {model_f1:<12.4f} {random_f1:<12.4f} {diff:+.4f}")
    
    print(f"{'-'*60}")
    
    # Summary statistics
    print(f"\n{'='*60}")
    print(f"SUMMARY STATISTICS:")
    print(f"{'='*60}")
    print(f"Total phonemes evaluated: {len(labels_sorted)}")
    print(f"Best performing phoneme: {labels_sorted[sorted_idx[0]]} (F1: {f1_by_class[sorted_idx[0]]:.4f})")
    print(f"Worst performing phoneme: {labels_sorted[sorted_idx[-1]]} (F1: {f1_by_class[sorted_idx[-1]]:.4f})")
    print(f"Mean F1 score: {f1_by_class.mean():.4f}")
    print(f"Std F1 score: {f1_by_class.std():.4f}")
    
    # Count phonemes beating random baseline
    better_than_random = (f1_by_class > random_f1_by_class).sum().item()
    print(f"Phonemes better than random: {better_than_random}/{len(labels_sorted)} ({better_than_random/len(labels_sorted)*100:.1f}%)")
    
    # Plot results
    if eval_config['plot_results']:
        plot_path = None
        if eval_config['save_plots']:
            plot_path = Path(args.checkpoint).parent / f"{eval_config['partition']}_f1_scores.png"
        
        plot_class_specific_scores(
            f1_by_class, random_f1_by_class, "F1 Score", 
            labels_sorted, save_path=plot_path
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Phoneme Classification Model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    
    args = parser.parse_args()
    main(args)