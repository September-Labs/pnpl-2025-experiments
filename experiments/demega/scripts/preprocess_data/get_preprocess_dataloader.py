from torch.utils.data import DataLoader, ConcatDataset, Subset, WeightedRandomSampler 
from pnpl.datasets import LibriBrainPhoneme, GroupedDataset, LibriBrainCompetitionHoldout
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path
import torch
import os

from .get_weighted_data import create_weighted_sampler 

class PseudoLabeledDataset(torch.utils.data.Dataset):
    """Wrapper dataset that pairs holdout data with pseudo-labels."""
    
    def __init__(self, base_dataset, pseudo_labels):
        """
        Args:
            base_dataset: The base holdout dataset (returns only data)
            pseudo_labels: Tensor or array of pseudo-labels
        """
        self.base_dataset = base_dataset
        self.pseudo_labels = torch.tensor(pseudo_labels, dtype=torch.long) if not isinstance(pseudo_labels, torch.Tensor) else pseudo_labels
        
        assert len(self.base_dataset) == len(self.pseudo_labels), \
            f"Dataset length ({len(self.base_dataset)}) must match pseudo_labels length ({len(self.pseudo_labels)})"
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        data = self.base_dataset[idx]
        label = self.pseudo_labels[idx]
        return data, label


def load_holdout_with_pseudolabels(data_config, holdout_config, channel_means=None, channel_stds=None):
    """
    Load holdout dataset with pseudo-labels from CSV.
    
    Args:
        data_config: Data configuration dictionary
        holdout_config: Holdout configuration dictionary
        channel_means: Optional channel means for standardization
        channel_stds: Optional channel stds for standardization
        
    Returns:
        PseudoLabeledDataset with holdout data and pseudo-labels
    """
    print("\n" + "="*60)
    print("LOADING HOLDOUT DATA WITH PSEUDO-LABELS")
    print("="*60)
    
    # Load holdout dataset
    print(f"Loading holdout dataset for task: {holdout_config['task']}")
    holdout_dataset = LibriBrainCompetitionHoldout(
        data_path=data_config['data_path'],
        task=holdout_config['task'],
        tmin=data_config['tmin'],
        tmax=data_config['tmax'],
        standardize=False  # We'll standardize after if needed
    )
    
    print(f"Holdout dataset size: {len(holdout_dataset)} samples")
    print(f"Sample shape: {holdout_dataset[0].shape}")
    
    # Read pseudo-labels CSV
    csv_path = Path(holdout_config['pseudo_labels_csv'])
    if not csv_path.exists():
        raise FileNotFoundError(f"Pseudo-labels CSV not found: {csv_path}")
    
    print(f"\nReading pseudo-labels from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"CSV shape: {df.shape}")
    print(f"CSV columns: {list(df.columns)[:5]}... ({len(df.columns)} total)")
    
    # Extract probability columns (all columns except segment_idx)
    prob_columns = [col for col in df.columns if col != 'segment_idx']
    probs = df[prob_columns].values  # Shape: (num_samples, num_classes)
    
    print(f"Probability matrix shape: {probs.shape}")
    
    # Verify alignment
    if len(probs) != len(holdout_dataset):
        raise ValueError(
            f"Mismatch: CSV has {len(probs)} rows but holdout dataset has {len(holdout_dataset)} samples"
        )
    
    if holdout_config['random_label']:
        pseudo_labels = np.random.randint(0, 39, probs.shape[0])
    else:
        # Convert probabilities to hard labels using argmax
        pseudo_labels = np.argmax(probs, axis=1)
        
    print(f"\nGenerated {len(pseudo_labels)} pseudo-labels using argmax")
    print(f"Number of unique classes: {len(np.unique(pseudo_labels))}")
    
    # Show pseudo-label distribution
    print(f"\nPseudo-label distribution:")
    label_counts = Counter(pseudo_labels)
    total = len(pseudo_labels)
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        percentage = count / total * 100
        print(f"  Class {label:2d}: {count:6d} samples ({percentage:5.1f}%)")
    
    # Show confidence statistics
    max_probs = np.max(probs, axis=1)
    print(f"\nPseudo-label confidence statistics:")
    print(f"  Mean confidence: {np.mean(max_probs):.4f}")
    print(f"  Median confidence: {np.median(max_probs):.4f}")
    print(f"  Min confidence: {np.min(max_probs):.4f}")
    print(f"  Max confidence: {np.max(max_probs):.4f}")
    
    # Apply confidence threshold filtering
    confidence_threshold = holdout_config.get('confidence_threshold', 0.0)
    if confidence_threshold > 0:
        high_confidence_mask = max_probs >= confidence_threshold
        high_confidence_indices = np.where(high_confidence_mask)[0]
        n_filtered = len(high_confidence_indices)
        n_total = len(pseudo_labels)
        
        print(f"\n{'='*60}")
        print(f"APPLYING CONFIDENCE FILTERING")
        print(f"{'='*60}")
        print(f"Confidence threshold: {confidence_threshold}")
        print(f"  Samples meeting threshold: {n_filtered}/{n_total} ({n_filtered/n_total*100:.1f}%)")
        print(f"  Samples filtered out: {n_total - n_filtered} ({(n_total - n_filtered)/n_total*100:.1f}%)")
        
        if n_filtered == 0:
            raise ValueError(
                f"Confidence threshold {confidence_threshold} filtered out all samples!\n"
                f"Maximum confidence in dataset: {np.max(max_probs):.4f}\n"
                f"Consider lowering the threshold."
            )
        
        # Filter the pseudo_labels
        pseudo_labels = pseudo_labels[high_confidence_indices]
        
        # Filter the holdout dataset using Subset
        holdout_dataset = Subset(holdout_dataset, high_confidence_indices.tolist())
        
        # Show filtered statistics
        filtered_max_probs = max_probs[high_confidence_indices]
        print(f"\nFiltered confidence statistics:")
        print(f"  Mean confidence: {np.mean(filtered_max_probs):.4f}")
        print(f"  Median confidence: {np.median(filtered_max_probs):.4f}")
        print(f"  Min confidence: {np.min(filtered_max_probs):.4f}")
        print(f"  Max confidence: {np.max(filtered_max_probs):.4f}")
        
        # Show filtered label distribution
        print(f"\nFiltered pseudo-label distribution:")
        label_counts = Counter(pseudo_labels)
        for label in sorted(label_counts.keys()):
            count = label_counts[label]
            percentage = count / n_filtered * 100
            print(f"  Class {label:2d}: {count:6d} samples ({percentage:5.1f}%)")
        
        print(f"{'='*60}")
    
    print("="*60 + "\n")
    
    # Wrap in PseudoLabeledDataset
    return PseudoLabeledDataset(holdout_dataset, pseudo_labels)


def create_dataloaders_preprocessed(config, preprocessed_dir, config_path):
    """Create dataloaders from preprocessed data."""
    print("Loading preprocessed datasets...")
    
    training_config = config['training']
    data_config = config['data']
    
    # Check if preprocessed files exist
    train_path = preprocessed_dir / 'train_grouped.h5'
    val_path = preprocessed_dir / 'validation_grouped.h5' 
    test_path = preprocessed_dir / 'test_grouped.h5'
    
    if not all(p.exists() for p in [train_path, val_path, test_path]):
        raise FileNotFoundError(
            f"Preprocessed datasets not found. Please run:\n"
            f"python preprocess.py {config_path}"
        )
    
    # Load datasets
    load_to_memory = data_config.get('load_to_memory', False)
    
    train_dataset = GroupedDataset(
        preprocessed_path=train_path,
        load_to_memory=load_to_memory
    )
    
    val_dataset = GroupedDataset(
        preprocessed_path=val_path,
        load_to_memory=load_to_memory
    )
    
    test_dataset = GroupedDataset(
        preprocessed_path=test_path,
        load_to_memory=load_to_memory
    )
    
    print(f"Loaded preprocessed datasets:")
    print(f"  Train: {len(train_dataset)} groups")
    print(f"  Val: {len(val_dataset)} groups")
    print(f"  Test: {len(test_dataset)} groups")
    
    # Check if we should add holdout data with pseudo-labels
    holdout_config = config.get('holdout', {})
    if holdout_config.get('use_holdout', False):
        print("\n⚠️  PSEUDO-LABELING ENABLED: Adding holdout data to preprocessed training set")
        
        # Load holdout with pseudo-labels (not preprocessed)
        # Note: We need channel_means and channel_stds, but for preprocessed data
        # the standardization is already applied, so we pass None
        holdout_dataset = load_holdout_with_pseudolabels(
            data_config, 
            holdout_config,
            channel_means=None,
            channel_stds=None
        )
        
        # Apply signal averaging to holdout data to match preprocessed format
        if data_config.get('use_signal_averaging', True):
            print("Applying signal averaging to holdout dataset...")
            holdout_dataset = GroupedDataset(
                holdout_dataset,
                grouped_samples=1,
                shuffle_seed=data_config.get('shuffle_seed', 777)
            )
        
        # Concatenate training and holdout datasets
        print(f"\nCombining preprocessed datasets:")
        print(f"  Original training groups: {len(train_dataset)}")
        print(f"  Holdout groups (pseudo-labeled): {len(holdout_dataset)}")
        
        combined_dataset = ConcatDataset([train_dataset, holdout_dataset])
        print(f"  Combined total: {len(combined_dataset)} groups")
        
        train_dataset = combined_dataset
    
    val_dataset = ConcatDataset([val_dataset, test_dataset])

    # Create weighted sampler for training data
    use_weighted_sampling = training_config.get('use_weighted_sampling', True)
    train_sampler = create_weighted_sampler(train_dataset, use_weighted_sampling)
    
    print("Train Sampler: ", train_sampler)
    # Create dataloaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=(train_sampler is None),  # Don't shuffle if using sampler
        sampler=train_sampler,
        num_workers=min(training_config['num_workers'], 8),
        pin_memory=True,
        persistent_workers=True if training_config['num_workers'] > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=min(training_config['num_workers'], 8),
        pin_memory=True,
        persistent_workers=True if training_config['num_workers'] > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=min(training_config['num_workers'], 8),
        pin_memory=True,
        persistent_workers=True if training_config['num_workers'] > 0 else False
    )
    
    return train_loader, val_loader, test_loader


### TEMPORARILY COMMENTED OUT, TESTING ANOTHER VERSION WITH A BETTER DATA HANDLING BELOW ###

# def create_dataloaders_multi_preprocessed(config, preprocessed_root, preprocessed_dirs):
#     """Create dataloaders from multiple preprocessed datasets."""
#     print("="*80)
#     print("LOADING MULTIPLE PREPROCESSED DATASETS")
#     print("="*80)
#     print(f"Preprocessed root: {preprocessed_root}")
#     print(f"Number of datasets: {len(preprocessed_dirs)}")
#     print(f"Datasets: {preprocessed_dirs}")
#     print("="*80 + "\n")
#     
#     training_config = config['training']
#     data_config = config['data']
#     load_to_memory = data_config.get('load_to_memory', False)
#     
#     # Lists to collect datasets from all directories
#     train_datasets = []
#     val_datasets = []
#     test_datasets = []
#     
#     # Load datasets from each directory
#     train_path, val_path, test_path =  None, None, None
#     for idx, dir_name in enumerate(preprocessed_dirs, 1):
#     
#         if os.path.exists(dir_name):
#             preprocessed_dir = Path(dir_name)
#         else:
#             preprocessed_dir = preprocessed_root / dir_name
#         
#         print(f"\n[{idx}/{len(preprocessed_dirs)}] Loading from: {dir_name}")
#         print("-" * 60)
#         
#         # Check if preprocessed files exist
#         train_path = preprocessed_dir / 'train_grouped.h5'
#         val_path = preprocessed_dir / 'validation_grouped.h5' 
#         test_path = preprocessed_dir / 'test_grouped.h5'
#         
#         if not train_path.exists():
#             print(f"⚠️  WARNING: Train file not found: {train_path}")
#             print(f"   Skipping this dataset...")
#             continue
#         
#         if not val_path.exists():
#             print(f"⚠️  WARNING: Validation file not found: {val_path}")
#             print(f"   Skipping this dataset...")
#             continue
#             
#         if not test_path.exists():
#             print(f"⚠️  WARNING: Test file not found: {test_path}")
#             print(f"   Skipping this dataset...")
#             continue
#         
#         # Load datasets
#         try:
#             train_dataset = GroupedDataset(
#                 preprocessed_path=train_path,
#                 load_to_memory=load_to_memory
#             )
#             train_datasets.append(train_dataset)
#             print(f"  ✓ Train: {len(train_dataset)} groups")
#             
#         except Exception as e:
#             print(f"  ✗ Error loading dataset: {e}")
#             print(f"  Skipping this dataset...")
#             continue
#     
#     # Check if we loaded any datasets
#     if not train_datasets:
#         raise ValueError("No valid datasets were loaded! Check your preprocessed_dirs configuration.")
#     
#     print("\n" + "="*80)
#     print("COMBINING DATASETS")
#     print("="*80)
#     
#     val_combined = GroupedDataset(
#         preprocessed_path=val_path,
#         load_to_memory=load_to_memory
#     )
#     
#     test_combined = GroupedDataset(
#         preprocessed_path=test_path,
#         load_to_memory=load_to_memory
#     )
#
#     
#     # Combine all datasets
#     if len(train_datasets) == 1:
#         print("Only one dataset loaded, using it directly")
#         train_combined = train_datasets[0]
#     else:
#         print(f"Combining {len(train_datasets)} datasets using ConcatDataset")
#         train_combined = ConcatDataset(train_datasets)
#     
#     print(f"\nCombined dataset sizes:")
#     print(f"  Train: {len(train_combined)} groups")
#     print(f"  Val: {len(val_combined)} groups")
#     print(f"  Test: {len(test_combined)} groups")
#     
#     # Check if we should add holdout data with pseudo-labels
#     holdout_config = config.get('holdout', {})
#     if holdout_config.get('use_holdout', False):
#         print("\n" + "="*80)
#         print("⚠️  PSEUDO-LABELING ENABLED: Adding holdout data to training set")
#         print("="*80)
#         
#         # Load holdout with pseudo-labels (not preprocessed)
#         holdout_dataset = load_holdout_with_pseudolabels(
#             data_config, 
#             holdout_config,
#             channel_means=None,
#             channel_stds=None
#         )
#         
#         # Apply signal averaging to holdout data to match preprocessed format
#         if data_config.get('use_signal_averaging', True):
#             print("\nApplying signal averaging to holdout dataset...")
#             holdout_dataset = GroupedDataset(
#                 holdout_dataset,
#                 grouped_samples=1,
#                 shuffle_seed=data_config.get('shuffle_seed', 777)
#             )
#         
#         # Concatenate training and holdout datasets
#         print(f"\nCombining with holdout data:")
#         print(f"  Combined training groups: {len(train_combined)}")
#         print(f"  Holdout groups (pseudo-labeled): {len(holdout_dataset)}")
#         
#         train_combined = ConcatDataset([train_combined, holdout_dataset])
#         print(f"  Final total: {len(train_combined)} groups")
#     
#     print("="*80 + "\n")
#     
#     # Combine val and test
#     val_combined = ConcatDataset([val_combined, test_combined])
#     
#     # Create weighted sampler for training data
#     use_weighted_sampling = training_config.get('use_weighted_sampling', True)
#     train_sampler = create_weighted_sampler(train_combined, use_weighted_sampling)
#     
#     print(f"Train Sampler: {train_sampler}")
#     
#     # Create dataloaders with optimized settings
#     train_loader = DataLoader(
#         train_combined,
#         batch_size=training_config['batch_size'],
#         shuffle=(train_sampler is None),
#         sampler=train_sampler,
#         num_workers=min(training_config['num_workers'], 8),
#         pin_memory=True,
#         persistent_workers=True if training_config['num_workers'] > 0 else False
#     )
#     
#     val_loader = DataLoader(
#         val_combined,
#         batch_size=training_config['batch_size'],
#         shuffle=False,
#         num_workers=min(training_config['num_workers'], 8),
#         pin_memory=True,
#         persistent_workers=True if training_config['num_workers'] > 0 else False
#     )
#     
#     test_loader = DataLoader(
#         test_combined,
#         batch_size=training_config['batch_size'],
#         shuffle=False,
#         num_workers=min(training_config['num_workers'], 8),
#         pin_memory=True,
#         persistent_workers=True if training_config['num_workers'] > 0 else False
#     )
#     
#     return train_loader, val_loader, test_loader


### UPDATED VERSION, INSTEAD OF LATEST DATA LOAD USES ALL DATA FROM ALL BASE MODELS, ###
### SHOULD WORK BETTER, DISABLE IF THINGS GO WRONG ###
def create_dataloaders_multi_preprocessed(config, preprocessed_root, preprocessed_dirs):
    """Create dataloaders from multiple preprocessed datasets."""
    print("="*80)
    print("LOADING MULTIPLE PREPROCESSED DATASETS (FIXED)")
    print("="*80)
    print(f"Preprocessed root: {preprocessed_root}")
    print(f"Number of datasets: {len(preprocessed_dirs)}")
    print(f"Datasets: {preprocessed_dirs}")
    print("="*80 + "\n")
    
    training_config = config['training']
    data_config = config['data']
    load_to_memory = data_config.get('load_to_memory', False)
    
    # Lists to collect datasets from all directories
    train_datasets = []
    val_datasets = []
    test_datasets = []
    
    # Load datasets from each directory
    # train_path, val_path, test_path =  None, None, None # This line is not needed
    
    for idx, dir_name in enumerate(preprocessed_dirs, 1):
    
        if os.path.exists(dir_name):
            preprocessed_dir = Path(dir_name)
        else:
            preprocessed_dir = preprocessed_root / dir_name
        
        print(f"\n[{idx}/{len(preprocessed_dirs)}] Loading from: {dir_name}")
        print("-" * 60)
        
        # Check if preprocessed files exist
        train_path = preprocessed_dir / 'train_grouped.h5'
        val_path = preprocessed_dir / 'validation_grouped.h5' 
        test_path = preprocessed_dir / 'test_grouped.h5'
        
        if not train_path.exists():
            print(f"⚠️  WARNING: Train file not found: {train_path}")
            print(f"   Skipping this dataset...")
            continue
        
        if not val_path.exists():
            print(f"⚠️  WARNING: Validation file not found: {val_path}")
            print(f"   Skipping this dataset...")
            continue
            
        if not test_path.exists():
            print(f"⚠️  WARNING: Test file not found: {test_path}")
            print(f"   Skipping this dataset...")
            continue
        
        # Load datasets
        try:
            train_dataset = GroupedDataset(
                preprocessed_path=train_path,
                load_to_memory=load_to_memory
            )
            train_datasets.append(train_dataset)
            print(f"  ✓ Train: {len(train_dataset)} groups")

            # <-- EDIT: Load and append val dataset -->
            val_dataset = GroupedDataset(
                preprocessed_path=val_path,
                load_to_memory=load_to_memory
            )
            val_datasets.append(val_dataset)
            print(f"  ✓ Val: {len(val_dataset)} groups")

            # <-- EDIT: Load and append test dataset -->
            test_dataset = GroupedDataset(
                preprocessed_path=test_path,
                load_to_memory=load_to_memory
            )
            test_datasets.append(test_dataset)
            print(f"  ✓ Test: {len(test_dataset)} groups")
            
        except Exception as e:
            print(f"  ✗ Error loading dataset: {e}")
            print(f"  Skipping this dataset...")
            continue
    
    # Check if we loaded any datasets
    if not train_datasets:
        raise ValueError("No valid datasets were loaded! Check your preprocessed_dirs configuration.")
    
    print("\n" + "="*80)
    print("COMBINING DATASETS")
    print("="*80)
    
    ### Remove buggy single-dataset loads##
    # val_combined = GroupedDataset(...)
    # test_combined = GroupedDataset(...)

    
    # Combine all datasets
    if len(train_datasets) == 1:
        print("Only one dataset loaded, using it directly")
        train_combined = train_datasets[0]
        val_combined = val_datasets[0]
        test_combined = test_datasets[0]
    else:
        print(f"Combining {len(train_datasets)} datasets using ConcatDataset")
        train_combined = ConcatDataset(train_datasets)
        val_combined = ConcatDataset(val_datasets)
        test_combined = ConcatDataset(test_datasets)
    
    print(f"\nCombined dataset sizes:")
    print(f"  Train: {len(train_combined)} groups")
    print(f"  Val: {len(val_combined)} groups")
    print(f"  Test: {len(test_combined)} groups")
    
    # Check if we should add holdout data with pseudo-labels
    holdout_config = config.get('holdout', {})
    if holdout_config.get('use_holdout', False):
        print("\n" + "="*80)
        print("⚠️  PSEUDO-LABELING ENABLED: Adding holdout data to training set")
        print("="*80)
        
        # Load holdout with pseudo-labels (not preprocessed)
        holdout_dataset = load_holdout_with_pseudolabels(
            data_config, 
            holdout_config,
            channel_means=None,
            channel_stds=None
        )
        
        # Apply signal averaging to holdout data to match preprocessed format
        if data_config.get('use_signal_averaging', True):
            print("\nApplying signal averaging to holdout dataset...")
            holdout_dataset = GroupedDataset(
                holdout_dataset,
                grouped_samples=1,
                shuffle_seed=data_config.get('shuffle_seed', 777)
            )
        
        # Concatenate training and holdout datasets
        print(f"\nCombining with holdout data:")
        print(f"  Combined training groups: {len(train_combined)}")
        print(f"  Holdout groups (pseudo-labeled): {len(holdout_dataset)}")
        
        train_combined = ConcatDataset([train_combined, holdout_dataset])
        print(f"  Final total: {len(train_combined)} groups")
    
    print("="*80 + "\n")
    
    # Combine val and test (This is your intended leakage)
    val_combined = ConcatDataset([val_combined, test_combined])
    
    # Create weighted sampler for training data
    use_weighted_sampling = training_config.get('use_weighted_sampling', True)
    train_sampler = create_weighted_sampler(train_combined, use_weighted_sampling)
    
    print(f"Train Sampler: {train_sampler}")
    
    # Create dataloaders with optimized settings
    train_loader = DataLoader(
        train_combined,
        batch_size=training_config['batch_size'],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=min(training_config['num_workers'], 8),
        pin_memory=True,
        persistent_workers=True if training_config['num_workers'] > 0 else False
    )
    
    val_loader = DataLoader(
        val_combined,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=min(training_config['num_workers'], 8),
        pin_memory=True,
        persistent_workers=True if training_config['num_workers'] > 0 else False
    )
    
    test_loader = DataLoader(
        test_combined,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=min(training_config['num_workers'], 8),
        pin_memory=True,
        persistent_workers=True if training_config['num_workers'] > 0 else False
    )
    
    return train_loader, val_loader, test_loader
