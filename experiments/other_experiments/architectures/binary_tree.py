"""
Hierarchical Binary Tree MEG Classifier for Phoneme Classification
Uses divide-and-conquer with balanced binary splits based on sample counts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import F1Score
from collections import defaultdict
import numpy as np
from typing import Dict, List, Tuple, Optional
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import seaborn as sns

# ============================================
# Tree Construction Module
# ============================================

class PhonemeTreeBuilder:
    """
    Builds a balanced binary tree based on phoneme sample counts.
    Each split aims to balance the total number of samples on each side.
    """
    
    def __init__(self, phoneme_counts: Dict[int, int]):
        self.phoneme_counts = phoneme_counts
        self.tree = self._build_tree(list(phoneme_counts.keys()))
        
    def _build_tree(self, phoneme_ids: List[int]) -> Dict:
        """
        Recursively build a binary tree that balances sample counts.
        """
        if len(phoneme_ids) == 1:
            return {'leaf': True, 'phoneme': phoneme_ids[0]}
        
        if len(phoneme_ids) == 0:
            raise ValueError("Empty phoneme list")
        
        # Sort phonemes by their counts
        sorted_phonemes = sorted(phoneme_ids, key=lambda x: self.phoneme_counts[x], reverse=True)
        
        # Use a greedy approach to balance the two groups
        left_group = []
        right_group = []
        left_count = 0
        right_count = 0
        
        for phoneme in sorted_phonemes:
            count = self.phoneme_counts[phoneme]
            if left_count <= right_count:
                left_group.append(phoneme)
                left_count += count
            else:
                right_group.append(phoneme)
                right_count += count
        
        # Ensure both groups are non-empty
        if not left_group or not right_group:
            mid = len(sorted_phonemes) // 2
            left_group = sorted_phonemes[:mid]
            right_group = sorted_phonemes[mid:]
            left_count = sum(self.phoneme_counts[p] for p in left_group)
            right_count = sum(self.phoneme_counts[p] for p in right_group)
        
        return {
            'leaf': False,
            'left_phonemes': left_group,
            'right_phonemes': right_group,
            'left_count': left_count,
            'right_count': right_count,
            'left': self._build_tree(left_group),
            'right': self._build_tree(right_group)
        }
    
    def get_node_paths(self) -> Dict[str, Dict]:
        """
        Get all paths from root to each node for easier traversal during training/inference.
        Returns dict with node_id -> {path, phonemes, is_leaf}
        """
        paths = {}
        
        def traverse(node, path="root"):
            if node['leaf']:
                paths[path] = {
                    'phonemes': [node['phoneme']], 
                    'is_leaf': True,
                    'node': node
                }
            else:
                paths[path] = {
                    'phonemes': node['left_phonemes'] + node['right_phonemes'],
                    'left_phonemes': node['left_phonemes'],
                    'right_phonemes': node['right_phonemes'],
                    'is_leaf': False,
                    'node': node
                }
                traverse(node['left'], path + "_L")
                traverse(node['right'], path + "_R")
        
        traverse(self.tree)
        return paths

# ============================================
# MEG Conformer Layer (from original)
# ============================================

class MEGConformerLayer(nn.Module):
    """Conformer layer adapted for MEG data."""
    
    def __init__(self, dim: int, num_heads: int = 4, ff_dim: int = None, 
                 kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        ff_dim = ff_dim or 2 * dim
        
        # Depthwise separable convolution
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim),
            nn.BatchNorm1d(dim),
            nn.Conv1d(dim, dim, 1),
            nn.ReLU()
        )
        
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ln3 = nn.LayerNorm(dim)
        
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (B, T, D)
        # Convolution module
        res = x
        x_conv = x.transpose(1, 2)  # (B, D, T)
        x_conv = self.conv(x_conv).transpose(1, 2)  # (B, T, D)
        x = self.ln1(x_conv + res)
        
        # Self-attention module
        res = x
        attn_out, _ = self.attention(x, x, x)
        x = self.ln2(self.dropout(attn_out) + res)
        
        # Feed-forward module
        res = x
        x = self.ffn(x)
        x = self.ln3(x + res)
        
        return x

# ============================================
# Binary Classifier Head
# ============================================

class BinaryClassifierHead(nn.Module):
    """
    Binary classification head for a tree node.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(128),
            nn.Linear(128, 1)  # Binary classification
        )
    
    def forward(self, x):
        return self.classifier(x)

# ============================================
# Main Hierarchical MEG Classifier
# ============================================

class HierarchicalMEGClassifier(L.LightningModule):
    """
    Hierarchical binary tree classifier for MEG phoneme classification.
    Uses soft routing through the tree during inference.
    """
    
    def __init__(self,
                 meg_channels: int = 306,
                 time_points: int = 125,
                 vocab_size: int = 39,
                 hidden_dim: int = 256,
                 num_conformers: int = 4,
                 learning_rate: float = 1e-4,
                 loss_type: str = 'cross_entropy',
                 use_class_weights: bool = True,
                 dropout_rate: float = 0.2,
                 label_smoothing: float = 0.0,
                 weight_decay: float = 0.01,
                 temperature: float = 1.0,  # For soft routing
                 warmup_epochs: int = 5,
                 total_epochs: int = 50):
        super().__init__()
        self.save_hyperparameters()
        
        # Phoneme counts from the data
        phoneme_counts = {
            0: 216, 1: 443, 2: 1584, 3: 231, 4: 114, 5: 360, 6: 268, 7: 93,
            8: 772, 9: 504, 10: 477, 11: 429, 12: 231, 13: 282, 14: 119,
            15: 428, 16: 1052, 17: 570, 18: 63, 19: 430, 20: 566, 21: 518,
            22: 1128, 23: 154, 24: 226, 25: 14, 26: 276, 27: 634, 28: 743,
            29: 113, 30: 1143, 31: 110, 32: 96, 33: 236, 34: 326, 35: 428,
            36: 151, 37: 456, 38: 7
        }
        
        # Build the tree
        self.tree_builder = PhonemeTreeBuilder(phoneme_counts)
        self.node_paths = self.tree_builder.get_node_paths()
        
        # Shared MEG encoder (from original architecture)
        self.input_projection = nn.Sequential(
            nn.Conv1d(meg_channels, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Conformer layers
        self.meg_encoder = nn.ModuleList([
            MEGConformerLayer(hidden_dim, 4, hidden_dim*2, dropout=dropout_rate) 
            for _ in range(num_conformers)
        ])
        
        # Binary classifiers for each non-leaf node
        self.binary_classifiers = nn.ModuleDict()
        flattened_dim = hidden_dim * time_points
        
        for node_id, node_info in self.node_paths.items():
            if not node_info['is_leaf']:
                self.binary_classifiers[node_id] = BinaryClassifierHead(
                    flattened_dim, hidden_dim // 2, dropout_rate
                )
        
        # Create phoneme to leaf path mapping
        self.phoneme_to_path = {}
        for node_id, node_info in self.node_paths.items():
            if node_info['is_leaf']:
                self.phoneme_to_path[node_info['phonemes'][0]] = node_id
        
        # Temperature for soft routing
        self.temperature = temperature
        
        # Metrics
        self.train_f1 = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
        self.val_f1 = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
        self.test_f1 = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
        
        # Class weights for balanced training
        if use_class_weights:
            self.class_weights = torch.zeros(vocab_size)
            for i, count in phoneme_counts.items():
                self.class_weights[i] = 1.0 / (count/15991 + 0.001)
            self.class_weights = self.class_weights / self.class_weights.mean()
        else:
            self.class_weights = None
        
        # Print initialization info
        self._print_model_info()
    
    def _print_model_info(self):
        """
        Print model initialization info with overfitting warning.
        """
        total_params = sum(p.numel() for p in self.parameters())
        encoder_params = sum(p.numel() for p in self.meg_encoder.parameters())
        classifier_params = sum(p.numel() for p in self.binary_classifiers.parameters())
        
        print("\n" + "="*80)
        print("HIERARCHICAL BINARY TREE MEG CLASSIFIER INITIALIZED")
        print("="*80)
        print(f"Model Parameters:")
        print(f"  Total: {total_params/1e6:.1f}M parameters")
        print(f"  Encoder: {encoder_params/1e6:.1f}M")
        print(f"  Binary Classifiers: {classifier_params/1e6:.1f}M")
        print(f"  Binary Nodes: {len(self.binary_classifiers)}")
        print(f"\n⚠️  WARNING: SEVERE OVERFITTING RISK!")
        print(f"  {total_params/1e6:.1f}M parameters for only 15,991 training samples")
        print(f"  = {total_params/15991:.0f} parameters per sample")
        print(f"  Recommended: <1M parameters for this dataset size")
        print(f"\nTraining Config:")
        print(f"  Hidden Dim: {self.hparams.hidden_dim}")
        print(f"  Conformer Layers: {self.hparams.num_conformers}")
        print(f"  Learning Rate: {self.hparams.learning_rate}")
        print(f"  Dropout: {self.hparams.dropout_rate}")
        print(f"  Weight Decay: {self.hparams.weight_decay}")
        print(f"  Temperature: {self.hparams.temperature}")
        print(f"\nVisualization Schedule:")
        print(f"  Tree traversal analysis: Every 5 epochs")
        print(f"  Confusion matrix analysis: Every 10 epochs")
        print("="*80 + "\n")
    
    def visualize_sample(self, sample_input, true_label=None, save_path=None):
        """
        Visualize a single sample's tree traversal.
        """
        # Print CLI visualization
        self.print_traversal_path(sample_input, true_label)
        
        # Plot visualization if matplotlib available
        try:
            fig = self.plot_tree_traversal(sample_input, true_label, save_path)
            return fig
        except:
            print("Matplotlib not available for plotting")
            return None
    
    def print_traversal_path(self, sample_input, true_label=None):
        """
        Print CLI visualization of tree traversal.
        """
        with torch.no_grad():
            if sample_input.dim() == 2:
                sample_input = sample_input.unsqueeze(0)
            
            phoneme_probs, path_probs = self(sample_input, return_paths=True)
            predicted = phoneme_probs.argmax().item()
        
        print("\n" + "="*70)
        print("TREE TRAVERSAL VISUALIZATION")
        print("="*70)
        
        if true_label is not None:
            print(f"True Phoneme: {true_label}")
        print(f"Predicted Phoneme: {predicted}")
        print(f"Confidence: {phoneme_probs.max().item():.3f}")
        
        print("\nDecision Path:")
        print("-"*70)
        
        # Trace the most likely path
        current = "root"
        depth = 0
        
        while current in self.node_paths and not self.node_paths[current]['is_leaf']:
            indent = "  " * depth
            node_info = self.node_paths[current]
            
            if current in path_probs:
                prob = path_probs[current].item()
                decision = "RIGHT" if prob > 0.5 else "LEFT"
                confidence = prob if prob > 0.5 else (1 - prob)
                
                left_phonemes = node_info['left_phonemes'][:3]
                right_phonemes = node_info['right_phonemes'][:3]
                
                print(f"{indent}Node {current}:")
                print(f"{indent}  Left  [{','.join(map(str, left_phonemes))}...] prob={1-prob:.3f}")
                print(f"{indent}  Right [{','.join(map(str, right_phonemes))}...] prob={prob:.3f}")
                print(f"{indent}  → Decision: {decision} (confidence: {confidence:.3f})")
                
                # Move to next node
                if prob > 0.5:
                    current = current + "_R"
                else:
                    current = current + "_L"
                depth += 1
            else:
                break
        
        # Print final phoneme
        if current in self.node_paths and self.node_paths[current]['is_leaf']:
            indent = "  " * depth
            phoneme = self.node_paths[current]['phonemes'][0]
            print(f"{indent}LEAF: Phoneme {phoneme}")
        
        print("="*70 + "\n")
    
    def plot_tree_traversal(self, sample_input, true_label=None, save_path=None):
        """
        Visualize how a sample traverses the tree with matplotlib.
        """
        with torch.no_grad():
            # Get predictions and path probabilities
            if sample_input.dim() == 2:
                sample_input = sample_input.unsqueeze(0)
            
            phoneme_probs, path_probs = self(sample_input, return_paths=True)
            predicted_phoneme = phoneme_probs.argmax().item()
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Tree structure with path probabilities
        ax1.set_title("Tree Traversal Path Probabilities")
        ax1.set_xlim(-1, 10)
        ax1.set_ylim(-1, 8)
        ax1.axis('off')
        
        # Draw tree nodes recursively
        def draw_node(node_id, x=5, y=7, width=9):
            if node_id not in self.node_paths:
                return
                
            node_info = self.node_paths[node_id]
            
            # Get probability for this node
            prob = path_probs.get(node_id, None)
            if prob is not None:
                prob = prob.item()
                color = plt.cm.RdYlGn(prob)  # Red to Green colormap
                alpha = 0.7
            else:
                color = 'lightgray'
                alpha = 0.3
                prob = 0.0
            
            # Draw rectangle for node
            rect = Rectangle((x - 0.4, y - 0.2), 0.8, 0.4, 
                           facecolor=color, alpha=alpha, edgecolor='black')
            ax1.add_patch(rect)
            
            # Add text
            if node_info['is_leaf']:
                phoneme_id = node_info['phonemes'][0]
                text = f"P{phoneme_id}"
                if true_label is not None and phoneme_id == true_label:
                    ax1.text(x, y + 0.3, "TRUE", fontsize=8, ha='center', color='blue')
            else:
                text = f"{prob:.2f}" if prob > 0 else "Node"
            
            ax1.text(x, y, text, fontsize=10, ha='center', va='center')
            
            # Draw children
            if not node_info['is_leaf']:
                # Left child
                new_width = width / 2
                left_x = x - new_width / 2
                draw_node(node_id + "_L", left_x, y - 1, new_width)
                ax1.plot([x, left_x], [y - 0.2, y - 0.8], 'k-', alpha=0.3)
                
                # Right child
                right_x = x + new_width / 2
                draw_node(node_id + "_R", right_x, y - 1, new_width)
                ax1.plot([x, right_x], [y - 0.2, y - 0.8], 'k-', alpha=0.3)
        
        draw_node("root")
        
        # Plot 2: Final phoneme probabilities
        ax2.set_title("Final Phoneme Probabilities")
        phoneme_probs_np = phoneme_probs.squeeze().cpu().numpy()
        colors = ['green' if i == predicted_phoneme else 
                 ('blue' if i == true_label else 'gray') 
                 for i in range(39)]
        bars = ax2.bar(range(39), phoneme_probs_np, color=colors, alpha=0.7)
        ax2.set_xlabel("Phoneme ID")
        ax2.set_ylabel("Probability")
        ax2.set_ylim(0, max(phoneme_probs_np) * 1.1)
        
        # Add legend
        green_patch = mpatches.Patch(color='green', label='Predicted')
        blue_patch = mpatches.Patch(color='blue', label='True')
        ax2.legend(handles=[green_patch, blue_patch])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        return fig
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features using the shared encoder.
        """
        B, C, T = x.shape
        
        # Apply initial convolution
        features = self.input_projection(x)  # (B, hidden_dim, T)
        features = features.transpose(1, 2)  # (B, T, hidden_dim)
        
        # Apply conformer layers
        for conformer in self.meg_encoder:
            features = conformer(features)
        
        return features
    
    def get_path_to_phoneme(self, phoneme_id: int) -> List[str]:
        """
        Get the path from root to a specific phoneme.
        """
        leaf_path = self.phoneme_to_path[phoneme_id]
        path = []
        current = "root"
        
        for direction in leaf_path.split("_")[1:]:  # Skip 'root'
            path.append((current, direction))
            current = current + "_" + direction
        
        return path
    
    def forward(self, x: torch.Tensor, return_paths: bool = False):
        """
        Forward pass with soft routing through the tree.
        """
        B, C, T = x.shape
        features = self.extract_features(x)  # (B, T, D)
        features_flat = features.reshape(B, -1)  # (B, T*D)
        
        # Soft routing through the tree
        device = x.device
        phoneme_probs = torch.zeros(B, 39).to(device)
        
        # Start at root with probability 1.0
        node_queue = [("root", torch.ones(B, 1).to(device))]
        path_probabilities = {} if return_paths else None
        
        while node_queue:
            node_id, node_prob = node_queue.pop(0)
            node_info = self.node_paths[node_id]
            
            if node_info['is_leaf']:
                # Accumulate probability for this phoneme
                phoneme_id = node_info['phonemes'][0]
                phoneme_probs[:, phoneme_id] += node_prob.squeeze()
            else:
                # Binary classification at this node
                logit = self.binary_classifiers[node_id](features_flat)  # (B, 1)
                prob = torch.sigmoid(logit / self.temperature)  # (B, 1)
                
                # Soft routing: distribute probability to children
                left_prob = node_prob * (1 - prob)  # Go left (class 0)
                right_prob = node_prob * prob  # Go right (class 1)
                
                if return_paths:
                    path_probabilities[node_id] = prob.squeeze()
                
                # Add children to queue
                node_queue.append((node_id + "_L", left_prob))
                node_queue.append((node_id + "_R", right_prob))
        
        if return_paths:
            return phoneme_probs, path_probabilities
        return phoneme_probs
    
    def compute_tree_loss(self, features_flat: torch.Tensor, targets: torch.Tensor):
        """
        Compute loss for all binary decisions along the paths to target phonemes.
        """
        B = features_flat.shape[0]
        device = features_flat.device
        total_loss = 0
        num_decisions = 0
        
        for b in range(B):
            target_phoneme = targets[b].item()
            path = self.get_path_to_phoneme(target_phoneme)
            
            for node_id, direction in path:
                if node_id in self.binary_classifiers:
                    logit = self.binary_classifiers[node_id](features_flat[b:b+1])  # Shape: [1, 1]
                    
                    # Target is 1 for right, 0 for left
                    target_direction = torch.tensor([1.0 if direction == "R" else 0.0]).to(device)
                    
                    # Binary cross-entropy loss - reshape both to ensure compatibility
                    loss = F.binary_cross_entropy_with_logits(
                        logit.view(-1),  # Flatten to 1D tensor [1]
                        target_direction.view(-1)  # Flatten to 1D tensor [1]
                    )
                    
                    # Weight by inverse frequency of this phoneme if using class weights
                    if self.class_weights is not None:
                        loss = loss * self.class_weights[target_phoneme]
                    
                    total_loss += loss
                    num_decisions += 1
        
        return total_loss / max(num_decisions, 1)
    
    def training_step(self, batch, batch_idx):
        x, y = batch  # x: (B, C, T), y: (B,)
        
        # Extract features once
        features = self.extract_features(x)
        features_flat = features.reshape(x.shape[0], -1)
        
        # Compute tree loss (sum of binary decisions)
        tree_loss = self.compute_tree_loss(features_flat, y)
        
        # Also compute standard cross-entropy on final predictions for monitoring
        phoneme_probs = self(x)
        ce_loss = F.cross_entropy(phoneme_probs, y)
        
        # Combined loss (tree loss is primary, CE for regularization)
        loss = 0.8 * tree_loss + 0.2 * ce_loss
        
        # Metrics
        with torch.no_grad():
            preds = phoneme_probs.argmax(dim=-1)
            acc = (preds == y).float().mean()
            f1 = self.train_f1(phoneme_probs, y)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_tree_loss', tree_loss)
        self.log('train_ce_loss', ce_loss)
        self.log('train_f1', f1, prog_bar=True)
        self.log('train_acc', acc)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        # Use soft routing for validation
        phoneme_probs = self(x)
        loss = F.cross_entropy(phoneme_probs, y)
        
        # Metrics
        preds = phoneme_probs.argmax(dim=-1)
        acc = (preds == y).float().mean()
        f1 = self.val_f1(phoneme_probs, y)
        
        # Track predictions for confusion analysis
        if not hasattr(self, '_val_predictions'):
            self._val_predictions = []
        
        for true, pred in zip(y.cpu(), preds.cpu()):
            self._val_predictions.append((true.item(), pred.item()))
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_f1_macro', f1, prog_bar=True)
        self.log('val_acc', acc)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        
        phoneme_probs = self(x)
        loss = F.cross_entropy(phoneme_probs, y)
        
        # Metrics
        preds = phoneme_probs.argmax(dim=-1)
        acc = (preds == y).float().mean()
        f1 = self.test_f1(phoneme_probs, y)
        
        self.log('test_loss', loss)
        self.log('test_f1_macro', f1)
        self.log('test_acc', acc)
        
        return loss
    
    def configure_optimizers(self):
        """
        Configure optimizer with warmup schedule.
        """
        # Different learning rates for encoder and classifiers
        params = [
            {'params': self.input_projection.parameters(), 'lr': self.hparams.learning_rate},
            {'params': self.meg_encoder.parameters(), 'lr': self.hparams.learning_rate},
            {'params': self.binary_classifiers.parameters(), 'lr': self.hparams.learning_rate * 2}
        ]
        
        optimizer = torch.optim.AdamW(params, weight_decay=self.hparams.weight_decay)
        
        # Learning rate scheduling with warmup
        def lr_lambda(epoch):
            if epoch < self.hparams.warmup_epochs:
                return (epoch + 1) / self.hparams.warmup_epochs
            else:
                progress = (epoch - self.hparams.warmup_epochs) / (self.hparams.total_epochs - self.hparams.warmup_epochs)
                return 0.5 * (1 + math.cos(math.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }
    
    def on_train_epoch_end(self):
        """
        Visualize tree traversals in terminal at end of each epoch.
        """
        # Only visualize every 5 epochs to avoid clutter
        if self.current_epoch % 5 != 0:
            return
        
        print("\n" + "="*80)
        print(f"EPOCH {self.current_epoch} - TREE TRAVERSAL ANALYSIS")
        print("="*80)
        
        # Get a sample from validation to visualize
        if hasattr(self.trainer, 'val_dataloaders') and self.trainer.val_dataloaders:
            val_loader = self.trainer.val_dataloaders[0] if isinstance(self.trainer.val_dataloaders, list) else self.trainer.val_dataloaders
            
            # Get first batch
            for batch in val_loader:
                x, y = batch
                if x.shape[0] > 0:
                    # Move to device
                    x = x.to(self.device)
                    y = y.to(self.device)
                    
                    # Analyze first few samples
                    num_samples = min(3, x.shape[0])
                    
                    with torch.no_grad():
                        # Get predictions for batch
                        phoneme_probs = self(x[:num_samples])
                        predictions = phoneme_probs.argmax(dim=1)
                        
                        # Print summary statistics
                        correct = (predictions == y[:num_samples]).sum().item()
                        print(f"\nQuick Validation Check: {correct}/{num_samples} correct")
                        
                        # Show detailed path for one sample
                        sample_idx = 0
                        _, path_probs = self(x[sample_idx:sample_idx+1], return_paths=True)
                        
                        true_label = y[sample_idx].item()
                        predicted = predictions[sample_idx].item()
                        confidence = phoneme_probs[sample_idx, predicted].item()
                        
                        print(f"\nDETAILED PATH ANALYSIS (Sample 1):")
                        print("-"*80)
                        print(f"True: Phoneme {true_label} | Predicted: Phoneme {predicted} | Conf: {confidence:.3f}")
                        
                        # Trace the path
                        current = "root"
                        depth = 0
                        path_taken = []
                        
                        while current in self.node_paths and not self.node_paths[current]['is_leaf']:
                            if current in path_probs:
                                prob = path_probs[current].item()
                                node_info = self.node_paths[current]
                                
                                # Determine decision
                                goes_right = prob > 0.5
                                decision = "R" if goes_right else "L"
                                path_taken.append(decision)
                                
                                # Count samples in each branch
                                left_samples = sum(self.tree_builder.phoneme_counts[p] for p in node_info['left_phonemes'])
                                right_samples = sum(self.tree_builder.phoneme_counts[p] for p in node_info['right_phonemes'])
                                
                                # Print node info
                                indent = "  " * depth
                                print(f"{indent}Node {depth}: [{left_samples:4d} | {right_samples:4d}] samples")
                                print(f"{indent}         Prob: L={1-prob:.3f} | R={prob:.3f} → {decision}")
                                
                                # Move to next node
                                current = current + "_" + decision
                                depth += 1
                            else:
                                break
                        
                        # Print final path
                        print(f"\nPath taken: root → {' → '.join(path_taken)}")
                        
                        # Show distribution of probabilities at each level
                        print("\nNode Activation Statistics:")
                        print("-"*40)
                        total_nodes = len(path_probs)
                        high_conf = sum(1 for p in path_probs.values() if p > 0.8 or p < 0.2)
                        medium_conf = sum(1 for p in path_probs.values() if 0.3 <= p <= 0.7)
                        
                        print(f"High confidence decisions (>0.8 or <0.2): {high_conf}/{total_nodes}")
                        print(f"Uncertain decisions (0.3-0.7): {medium_conf}/{total_nodes}")
                    
                    break  # Only process first batch
        
        # Print parameter usage warning
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\nWARNING: Model has {total_params/1e6:.1f}M parameters for {15991} training samples")
        print(f"         That's {total_params/15991:.0f} parameters per sample - severe overfitting risk!")
        
        print("="*80 + "\n")
    
    def on_validation_epoch_end(self):
        """
        Analyze per-phoneme performance through the tree.
        """
        if self.current_epoch % 10 != 0:
            return
            
        print("\n" + "="*80)
        print(f"EPOCH {self.current_epoch} - PER-PHONEME TREE PERFORMANCE")
        print("="*80)
        
        # Analyze which phonemes are being confused
        if hasattr(self, '_val_predictions'):
            confusion_matrix = torch.zeros(39, 39)
            
            for true, pred in self._val_predictions:
                confusion_matrix[true, pred] += 1
            
            # Find most confused pairs
            confusion_matrix_no_diag = confusion_matrix.clone()
            confusion_matrix_no_diag.fill_diagonal_(0)
            
            top_confusions = []
            for _ in range(5):
                max_val = confusion_matrix_no_diag.max()
                if max_val == 0:
                    break
                indices = (confusion_matrix_no_diag == max_val).nonzero()[0]
                i, j = indices[0].item(), indices[1].item()
                top_confusions.append((i, j, max_val.item()))
                confusion_matrix_no_diag[i, j] = 0
            
            print("\nTop Confused Phoneme Pairs:")
            for true_p, pred_p, count in top_confusions:
                # Check if they're in same subtree
                true_path = self.phoneme_to_path.get(true_p, "")
                pred_path = self.phoneme_to_path.get(pred_p, "")
                
                # Find common prefix
                common_depth = 0
                for a, b in zip(true_path.split("_"), pred_path.split("_")):
                    if a == b:
                        common_depth += 1
                    else:
                        break
                
                print(f"  Phoneme {true_p} → {pred_p}: {int(count)} times (split at depth {common_depth-1})")
            
            # Clear predictions for next epoch
            self._val_predictions = []
        
        print("="*80 + "\n")
    
    def print_tree_structure(self):
        """
        Print the tree structure for visualization.
        """
        print("\n" + "="*60)
        print("HIERARCHICAL TREE STRUCTURE")
        print("="*60)
        
        def print_node(node_id, depth=0):
            node_info = self.node_paths[node_id]
            indent = "  " * depth
            
            if node_info['is_leaf']:
                phoneme_id = node_info['phonemes'][0]
                count = self.tree_builder.phoneme_counts[phoneme_id]
                print(f"{indent}└─ Phoneme {phoneme_id}: {count} samples")
            else:
                left_count = sum(self.tree_builder.phoneme_counts[p] for p in node_info['left_phonemes'])
                right_count = sum(self.tree_builder.phoneme_counts[p] for p in node_info['right_phonemes'])
                print(f"{indent}├─ Node {node_id}: L={left_count} vs R={right_count} samples")
                print_node(node_id + "_L", depth + 1)
                print_node(node_id + "_R", depth + 1)
        
        print_node("root")
        print("="*60 + "\n")


# ============================================
# Example Usage
# ============================================

if __name__ == "__main__":
    # Initialize model
    model = HierarchicalMEGClassifier(
        meg_channels=306,
        time_points=125,
        vocab_size=39,
        hidden_dim=256,
        num_conformers=4,
        loss_type='cross_entropy',
        use_class_weights=True,
        temperature=1.0,  # Soft routing temperature
        dropout_rate=0.1,
        label_smoothing=0.1
    )
    
    # Print tree structure
    model.print_tree_structure()
    
    # Test with dummy data
    dummy_input = torch.randn(4, 306, 125)
    dummy_target = torch.randint(0, 39, (4,))
    
    # Forward pass
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Output sum per sample: {output.sum(dim=1)}")
    
    # Training step
    loss = model.training_step((dummy_input, dummy_target), 0)
    print(f"Training loss: {loss.item():.4f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.meg_encoder.parameters())
    classifier_params = sum(p.numel() for p in model.binary_classifiers.parameters())
    
    print(f"\nModel Statistics:")
    print(f"Total parameters: {total_params/1e6:.2f}M")
    print(f"Encoder parameters: {encoder_params/1e6:.2f}M")
    print(f"Binary classifiers: {classifier_params/1e6:.2f}M")
    print(f"Number of binary nodes: {len(model.binary_classifiers)}")