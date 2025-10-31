import tqdm
from torch.utils.data import WeightedRandomSampler 
from collections import Counter


def get_dataset_labels(dataset):
    """
    Extract all labels from a dataset, handling both regular and grouped datasets.
    
    Args:
        dataset: Dataset object (LibriBrainPhoneme or GroupedDataset)
        
    Returns:
        List of labels
    """
    print("Extracting labels for class distribution analysis...")
    
    if hasattr(dataset, 'dataset'):
        # This is likely a GroupedDataset, get labels from underlying dataset
        base_dataset = dataset.dataset
        if hasattr(base_dataset, 'labels'):
            return base_dataset.labels.tolist() if hasattr(base_dataset.labels, 'tolist') else base_dataset.labels
        else:
            # Fallback: iterate through base dataset
            print("  Iterating through base dataset to extract labels...")
            labels = []
            for i in tqdm(range(len(base_dataset))):
                try:
                    _, label = base_dataset[i]
                    labels.append(label.item() if hasattr(label, 'item') else label)
                except Exception as e:
                    print(f"  Warning: Could not extract label for sample {i}: {e}")
                    continue
            return labels
    else:
        # Regular dataset
        if hasattr(dataset, 'labels'):
            return dataset.labels.tolist() if hasattr(dataset.labels, 'tolist') else dataset.labels
        else:
            # Fallback: iterate through dataset
            print("  Iterating through dataset to extract labels...")
            labels = []
            for i in tqdm(range(len(dataset))):
                try:
                    _, label = dataset[i]
                    labels.append(label.item() if hasattr(label, 'item') else label)
                except Exception as e:
                    print(f"  Warning: Could not extract label for sample {i}: {e}")
                    continue
            return labels

def calculate_class_weights(labels):
    """
    Calculate class weights for balanced sampling.
    
    Args:
        labels: List of class labels
        
    Returns:
        Dictionary mapping class indices to weights
    """
    # Calculate class frequencies
    class_counts = Counter(labels)

    num_samples = len(labels)
    num_classes = len(class_counts)
    
    print(f"\nClass distribution in training set:")
    print(f"Total samples: {num_samples}")
    print(f"Number of classes: {num_classes}")
    
    for class_idx in sorted(class_counts.keys()):
        count = class_counts[class_idx]
        percentage = count / num_samples * 100
        print(f"  Class {class_idx}: {count:6d} samples ({percentage:5.1f}%)")
    
    # Calculate weights (inverse frequency normalized)
    class_weights = {}
    for class_idx, count in class_counts.items():
        class_weights[class_idx] = num_samples / (num_classes * count)
    
    print(f"\nCalculated class weights:")
    for class_idx in sorted(class_weights.keys()):
        print(f"  Class {class_idx}: {class_weights[class_idx]:.4f}")
    
    return class_weights



def create_weighted_sampler(dataset, use_weighted_sampling=True):
    """
    Create a weighted sampler for balanced class representation.
    
    Args:
        dataset: Training dataset
        use_weighted_sampling: Whether to use weighted sampling
        
    Returns:
        WeightedRandomSampler or None
    """
    print("Created weighted random sampler")
    if not use_weighted_sampling:
        print("Weighted sampling disabled - using standard random sampling")
        return None
    
    try:
        # Extract labels from dataset
        labels = get_dataset_labels(dataset)        

        # Calculate class weights
        class_weights = calculate_class_weights(labels)
        
        # For GroupedDataset, we need to handle sampling differently
        print("\nNote: Using GroupedDataset - weighted sampling will be applied at group level")
        # Get labels for the groups (not individual samples)
        group_labels = []
        for i in tqdm(range(len(dataset))):
            try:
                _, label = dataset[i]
                group_labels.append(label.item() if hasattr(label, 'item') else label)
            except Exception as e:
                print(f"Warning: Could not extract label for group {i}: {e}")
                continue
        
        if not group_labels:
            print("Warning: Could not extract group labels. Using standard sampling.")
            return None
        
        # Create sample weights for groups
        sample_weights = [class_weights[label] for label in group_labels]
        num_samples = len(group_labels)
        
        print(f"\nCreating WeightedRandomSampler with {num_samples} samples")
        
        # Create weighted sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=num_samples,
            replacement=True
        )
        
        return sampler
        
    except Exception as e:
        print(f"Error creating weighted sampler: {e}")
        print("Falling back to standard random sampling")
        return None

