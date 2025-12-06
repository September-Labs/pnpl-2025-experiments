"""
Hierarchical Binary Tree MEG Model for Phoneme Classification
Handles class imbalance through frequency-based binary splits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import F1Score
from collections import defaultdict
from typing import List, Tuple, Optional, Dict
import numpy as np

# ============================================
# Tree Node Definition
# ============================================

class PhonemeNode:
    """Node in the phoneme binary tree."""
    
    def __init__(self, phoneme_ids: List[int], phoneme_counts: Dict[int, int], 
                 node_id: str = "root", depth: int = 0):
        self.phoneme_ids = phoneme_ids
        self.phoneme_counts = {pid: phoneme_counts[pid] for pid in phoneme_ids}
        self.total_count = sum(self.phoneme_counts.values())
        self.node_id = node_id
        self.depth = depth
        self.left_child = None
        self.right_child = None
        self.is_leaf = len(phoneme_ids) == 1
        
        # For binary classification at this node
        self.left_phonemes = []
        self.right_phonemes = []
    
    def create_balanced_split(self):
        """Split phonemes into two balanced groups by cumulative frequency."""
        if self.is_leaf:
            return
        
        # Sort phonemes by count
        sorted_phonemes = sorted(self.phoneme_counts.items(), key=lambda x: x[1], reverse=True)
        
        left_count = 0
        right_count = 0
        left_group = []
        right_group = []
        
        # Greedy assignment to balance counts
        for pid, count in sorted_phonemes:
            if left_count <= right_count:
                left_group.append(pid)
                left_count += count
            else:
                right_group.append(pid)
                right_count += count
        
        # Ensure both groups are non-empty
        if not left_group or not right_group:
            mid = len(sorted_phonemes) // 2
            left_group = [p[0] for p in sorted_phonemes[:mid]]
            right_group = [p[0] for p in sorted_phonemes[mid:]]
        
        self.left_phonemes = left_group
        self.right_phonemes = right_group
        
        return left_group, right_group

# ============================================
# Mini Conformer Components
# ============================================

class LightweightConformerBlock(nn.Module):
    """Simplified conformer block for tree nodes."""
    
    def __init__(self, dim: int = 64, num_heads: int = 2, ff_mult: float = 1.5, 
                 kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        ff_dim = int(dim * ff_mult)
        
        # Depthwise separable convolution (simplified)
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, 1)
        )
        
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim)
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
        x = res + self.dropout(attn_out)
        
        # Feed-forward module
        res = x
        x = self.ln2(res + self.ffn(x))
        
        return x

# ============================================
# Tree Node Classifier Module
# ============================================

class TreeNodeClassifier(nn.Module):
    """Binary classifier for a single tree node."""
    
    def __init__(self, 
                 input_channels: int = 306,
                 hidden_dim: int = 64,
                 num_conformers: int = 1,
                 dropout: float = 0.1,
                 depth: int = 0):
        super().__init__()
        self.depth = depth
        
        # Adaptive dimensions based on depth (deeper = more specialized)
        self.hidden_dim = hidden_dim + (depth * 16)  # Increase capacity with depth
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_channels, self.hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(self.hidden_dim),
            nn.GELU(),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1, 
                     groups=self.hidden_dim),  # Depthwise
            nn.BatchNorm1d(self.hidden_dim),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, 1),  # Pointwise
            nn.GELU()
        )
        
        # Mini conformer blocks
        self.conformers = nn.ModuleList([
            LightweightConformerBlock(self.hidden_dim, num_heads=2, dropout=dropout)
            for _ in range(num_conformers)
        ])
        
        # Pooling and classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 128),  # *2 for mean+max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1)  # Binary output
        )
        
        # Confidence estimation head
        self.confidence_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, C, T) MEG signal
        Returns:
            logit: (B, 1) binary classification logit
            confidence: (B, 1) confidence score
            features: (B, D) pooled features
        """
        B, C, T = x.shape
        
        # Extract features
        features = self.feature_extractor(x)  # (B, hidden_dim, T)
        features = features.transpose(1, 2)  # (B, T, hidden_dim)
        
        # Apply conformers
        for conformer in self.conformers:
            features = conformer(features)
        
        # Pool over time dimension
        mean_pool = features.mean(dim=1)  # (B, hidden_dim)
        max_pool = features.max(dim=1)[0]  # (B, hidden_dim)
        pooled = torch.cat([mean_pool, max_pool], dim=1)  # (B, hidden_dim*2)
        
        # Binary classification
        logit = self.classifier(pooled)  # (B, 1)
        
        # Confidence estimation
        confidence = self.confidence_head(pooled)  # (B, 1)
        
        return logit, confidence, pooled

# ============================================
# Main Hierarchical Model
# ============================================

class HierarchicalMEGClassifier(L.LightningModule):
    """
    Hierarchical binary tree classifier for MEG phoneme classification.
    Handles class imbalance through frequency-based tree structure.
    """
    
    def __init__(self,
                 meg_channels: int = 306,
                 time_points: int = 125,
                 vocab_size: int = 39,
                 base_hidden_dim: int = 64,
                 num_conformers_per_node: int = 1,
                 learning_rate: float = 1e-3,
                 dropout_rate: float = 0.1,
                 weight_decay: float = 0.01,
                 confidence_threshold: float = 0.9,
                 depth_weight_factor: float = 1.2,
                 warmup_epochs: int = 5,
                 total_epochs: int = 50):
        super().__init__()
        self.save_hyperparameters()
        
        # Phoneme counts from the original model
        self.phoneme_counts = {
            0: 216, 1: 443, 2: 1584, 3: 231, 4: 114, 5: 360, 6: 268, 7: 93,
            8: 772, 9: 504, 10: 477, 11: 429, 12: 231, 13: 282, 14: 119,
            15: 428, 16: 1052, 17: 570, 18: 63, 19: 430, 20: 566, 21: 518,
            22: 1128, 23: 154, 24: 226, 25: 14, 26: 276, 27: 634, 28: 743,
            29: 113, 30: 1143, 31: 110, 32: 96, 33: 236, 34: 326, 35: 428,
            36: 151, 37: 456, 38: 7
        }
        
        # Build the tree structure
        self.tree_structure = self._build_tree()
        
        # Create node classifiers
        self.node_classifiers = nn.ModuleDict()
        self._create_node_classifiers(self.tree_structure)
        
        # Create phoneme to path mapping
        self.phoneme_paths = self._create_phoneme_paths()
        
        # Metrics
        self.train_f1 = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
        self.val_f1 = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
        self.test_f1 = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
        
        # Track per-node and per-phoneme performance
        self.node_accuracies = defaultdict(lambda: {'correct': 0, 'total': 0})
        self.phoneme_accuracies = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    def _build_tree(self) -> PhonemeNode:
        """Build the balanced binary tree structure."""
        root = PhonemeNode(list(self.phoneme_counts.keys()), self.phoneme_counts, "root", 0)
        
        queue = [root]
        while queue:
            node = queue.pop(0)
            if len(node.phoneme_ids) > 1:
                left_ids, right_ids = node.create_balanced_split()
                
                # Create child nodes
                node.left_child = PhonemeNode(
                    left_ids, self.phoneme_counts, 
                    f"{node.node_id}_L", node.depth + 1
                )
                node.right_child = PhonemeNode(
                    right_ids, self.phoneme_counts,
                    f"{node.node_id}_R", node.depth + 1
                )
                
                queue.append(node.left_child)
                queue.append(node.right_child)
        
        return root
    
    def _create_node_classifiers(self, node: PhonemeNode):
        """Recursively create classifiers for each non-leaf node."""
        if not node.is_leaf:
            # Adjust conformer count based on depth (more for deeper nodes)
            num_conformers = self.hparams.num_conformers_per_node + (node.depth // 2)
            
            self.node_classifiers[node.node_id] = TreeNodeClassifier(
                input_channels=self.hparams.meg_channels,
                hidden_dim=self.hparams.base_hidden_dim,
                num_conformers=min(num_conformers, 3),  # Cap at 3
                dropout=self.hparams.dropout_rate,
                depth=node.depth
            )
            
            if node.left_child:
                self._create_node_classifiers(node.left_child)
            if node.right_child:
                self._create_node_classifiers(node.right_child)
    
    def _create_phoneme_paths(self) -> Dict[int, List[Tuple[str, bool]]]:
        """Create mapping from phoneme to path through tree."""
        paths = {}
        
        def traverse(node: PhonemeNode, path: List[Tuple[str, bool]]):
            if node.is_leaf:
                # Reached a leaf - store the path for this phoneme
                paths[node.phoneme_ids[0]] = path.copy()
                return
            
            # Process left subtree
            if node.left_child:
                new_path = path + [(node.node_id, False)]  # False = go left
                traverse(node.left_child, new_path)
            
            # Process right subtree
            if node.right_child:
                new_path = path + [(node.node_id, True)]  # True = go right
                traverse(node.right_child, new_path)
        
        traverse(self.tree_structure, [])
        
        # Verify all phonemes have paths
        for pid in range(self.hparams.vocab_size):
            if pid not in paths:
                print(f"Warning: No path found for phoneme {pid}")
        
        return paths
    
    def forward(self, x: torch.Tensor, return_paths: bool = False) -> torch.Tensor:
        """
        Forward pass through the tree.
        
        Args:
            x: (B, C, T) MEG signals
            return_paths: If True, return the paths taken through tree
        
        Returns:
            logits: (B, vocab_size) phoneme logits
        """
        B = x.shape[0]
        device = x.device
        
        # Initialize logits
        logits = torch.zeros(B, self.hparams.vocab_size, device=device)
        
        # Process each sample (could be optimized with batching)
        paths_taken = []
        for b in range(B):
            sample = x[b:b+1]  # Keep batch dimension
            path = []
            node = self.tree_structure
            
            # Traverse tree
            while not node.is_leaf:
                # Check if we have a classifier for this node
                if node.node_id not in self.node_classifiers:
                    print(f"Warning: No classifier for node {node.node_id}")
                    break
                    
                classifier = self.node_classifiers[node.node_id]
                node_logit, confidence, features = classifier(sample)
                
                # Binary decision
                prob = torch.sigmoid(node_logit).item()
                go_right = prob > 0.5
                
                path.append((node.node_id, go_right, confidence.item()))
                
                # Early stopping based on confidence
                if confidence.item() > self.hparams.confidence_threshold and node.depth < 3:
                    # High confidence - can stop early for common phonemes
                    # Set high logit for predicted phonemes
                    if go_right:
                        for pid in node.right_phonemes:
                            logits[b, pid] = 5.0 / len(node.right_phonemes)
                    else:
                        for pid in node.left_phonemes:
                            logits[b, pid] = 5.0 / len(node.left_phonemes)
                    break
                
                # Move to next node
                next_node = node.right_child if go_right else node.left_child
                if next_node is None:
                    print(f"Warning: Missing {'right' if go_right else 'left'} child for node {node.node_id}")
                    break
                node = next_node
            
            # If we reached a leaf normally
            if node.is_leaf:
                logits[b, node.phoneme_ids[0]] = 10.0  # High logit for selected phoneme
            
            paths_taken.append(path)
        
        if return_paths:
            return logits, paths_taken
        return logits
    
    def compute_hierarchical_loss(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for all nodes in the paths to target phonemes.
        """
        B = x.shape[0]
        device = x.device
        total_loss = 0
        loss_weights = []
        losses = []
        
        for b in range(B):
            sample = x[b:b+1]
            target_phoneme = targets[b].item()
            
            # Handle case where phoneme_paths might not have this phoneme yet
            if target_phoneme not in self.phoneme_paths:
                continue
                
            path = self.phoneme_paths[target_phoneme]
            
            # Compute loss at each node in the path
            for depth, (node_id, should_go_right) in enumerate(path):
                if node_id not in self.node_classifiers:
                    continue
                    
                classifier = self.node_classifiers[node_id]
                node_logit, confidence, features = classifier(sample)
                
                # Binary cross entropy loss - ensure shapes match
                # node_logit is (1, 1), so we keep it that way
                target_binary = torch.full_like(node_logit, 1.0 if should_go_right else 0.0)
                node_loss = F.binary_cross_entropy_with_logits(node_logit, target_binary)
                
                # Weight by depth (deeper nodes more important for rare phonemes)
                depth_weight = self.hparams.depth_weight_factor ** depth
                losses.append(node_loss)
                loss_weights.append(depth_weight)
                
                # Track node accuracy
                pred_right = torch.sigmoid(node_logit).item() > 0.5
                correct = pred_right == should_go_right
                self.node_accuracies[node_id]['correct'] += int(correct)
                self.node_accuracies[node_id]['total'] += 1
        
        # Combine losses
        if losses:
            losses = torch.stack(losses)
            weights = torch.tensor(loss_weights, device=device)
            weights = weights / weights.sum()
            total_loss = (losses * weights).sum()
        else:
            # Fallback to prevent NaN
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        return total_loss
    
    def training_step(self, batch, batch_idx):
        x, y = batch  # x: (B, C, T), y: (B,)
        
        # Hierarchical loss
        loss = self.compute_hierarchical_loss(x, y)
        
        # Get predictions for metrics
        with torch.no_grad():
            logits = self(x)
            preds = logits.argmax(dim=-1)
            acc = (preds == y).float().mean()
            f1 = self.train_f1(logits, y)
            
            # Track per-phoneme accuracy
            for i in range(len(y)):
                phoneme_id = y[i].item()
                correct = preds[i].item() == phoneme_id
                self.phoneme_accuracies[phoneme_id]['correct'] += int(correct)
                self.phoneme_accuracies[phoneme_id]['total'] += 1
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_f1', f1, prog_bar=True)
        self.log('train_f1_macro', f1)
        self.log('train_acc', acc)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        logits = self(x)
        
        # Standard cross entropy for validation
        loss = F.cross_entropy(logits, y)
        
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()
        f1 = self.val_f1(logits, y)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_f1', f1, prog_bar=True)
        self.log('val_f1_macro', f1, prog_bar=True)
        self.log('val_acc', acc)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        
        logits, paths = self(x, return_paths=True)
        
        loss = F.cross_entropy(logits, y)
        
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()
        f1 = self.test_f1(logits, y)
        
        self.log('test_loss', loss)
        self.log('test_f1', f1)
        self.log('test_f1_macro', f1)
        self.log('test_acc', acc)
        
        # Log average path length (efficiency metric)
        avg_path_length = np.mean([len(p) for p in paths])
        self.log('test_avg_path_length', avg_path_length)
        
        return loss
    
    def on_train_epoch_end(self):
        """Log hierarchical performance statistics."""
        if self.current_epoch % 5 == 0:
            print(f"\n{'='*60}")
            print(f"Epoch {self.current_epoch} - Hierarchical Performance:")
            
            # Node-level accuracy
            print("\nNode Accuracies by Depth:")
            nodes_by_depth = defaultdict(list)
            
            def collect_nodes(node, depth=0):
                if not node.is_leaf:
                    nodes_by_depth[depth].append(node.node_id)
                    if node.left_child:
                        collect_nodes(node.left_child, depth + 1)
                    if node.right_child:
                        collect_nodes(node.right_child, depth + 1)
            
            collect_nodes(self.tree_structure)
            
            for depth in sorted(nodes_by_depth.keys()):
                accs = []
                for node_id in nodes_by_depth[depth]:
                    if self.node_accuracies[node_id]['total'] > 0:
                        acc = self.node_accuracies[node_id]['correct'] / self.node_accuracies[node_id]['total']
                        accs.append(acc)
                if accs:
                    print(f"  Depth {depth}: {np.mean(accs):.3f} (±{np.std(accs):.3f})")
            
            # Phoneme-level accuracy
            print("\nPhoneme Performance (sorted by frequency):")
            phoneme_accs = []
            for pid in sorted(self.phoneme_counts.keys(), 
                            key=lambda x: self.phoneme_counts[x], reverse=True)[:10]:
                if self.phoneme_accuracies[pid]['total'] > 0:
                    acc = self.phoneme_accuracies[pid]['correct'] / self.phoneme_accuracies[pid]['total']
                    count = self.phoneme_counts[pid]
                    phoneme_accs.append((pid, acc, count))
            
            for pid, acc, count in phoneme_accs:
                print(f"  Phoneme {pid:2d} (n={count:4d}): {acc:.3f}")
            
            # Rare phoneme performance
            print("\nRare Phoneme Performance (n<100):")
            rare_accs = []
            for pid, count in self.phoneme_counts.items():
                if count < 100 and self.phoneme_accuracies[pid]['total'] > 0:
                    acc = self.phoneme_accuracies[pid]['correct'] / self.phoneme_accuracies[pid]['total']
                    rare_accs.append((pid, acc, count))
            
            rare_accs.sort(key=lambda x: x[2])  # Sort by count
            for pid, acc, count in rare_accs[:5]:
                print(f"  Phoneme {pid:2d} (n={count:3d}): {acc:.3f}")
            
            print(f"{'='*60}\n")
            
            # Reset counters
            self.node_accuracies = defaultdict(lambda: {'correct': 0, 'total': 0})
            self.phoneme_accuracies = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    def configure_optimizers(self):
        """Configure optimizer with depth-based learning rates."""
        param_groups = []
        
        # Group parameters by node depth
        for node_id, classifier in self.node_classifiers.items():
            depth = node_id.count('_')  # Simple depth estimation
            lr_scale = 1.0 + (depth * 0.2)  # Higher LR for deeper nodes
            
            param_groups.append({
                'params': classifier.parameters(),
                'lr': self.hparams.learning_rate * lr_scale,
                'name': f'node_{node_id}'
            })
        
        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.hparams.weight_decay)
        
        # Cosine annealing with warmup
        def lr_lambda(epoch):
            if epoch < self.hparams.warmup_epochs:
                return (epoch + 1) / self.hparams.warmup_epochs
            else:
                progress = (epoch - self.hparams.warmup_epochs) / (self.hparams.total_epochs - self.hparams.warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}
        }


# ============================================
# Training Script
# ============================================

if __name__ == "__main__":
    # Example usage
    model = HierarchicalMEGClassifier(
        meg_channels=306,
        time_points=125,
        vocab_size=39,
        base_hidden_dim=64,
        num_conformers_per_node=1,
        learning_rate=1e-3,
        dropout_rate=0.15,
        confidence_threshold=0.9,
        depth_weight_factor=1.2
    )
    
    # Example forward pass
    dummy_input = torch.randn(4, 306, 125)
    dummy_target = torch.randint(0, 39, (4,))
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Number of tree nodes: {len(model.node_classifiers)}")
    
    # Verify tree structure
    print(f"\nTree Structure:")
    print(f"  Root node has {len(model.tree_structure.left_phonemes)} left phonemes, "
          f"{len(model.tree_structure.right_phonemes)} right phonemes")
    print(f"  Left total count: {sum(model.phoneme_counts[p] for p in model.tree_structure.left_phonemes)}")
    print(f"  Right total count: {sum(model.phoneme_counts[p] for p in model.tree_structure.right_phonemes)}")
    
    # Test forward pass
    logits = model(dummy_input)
    print(f"\nForward pass output shape: {logits.shape}")
    
    # Test loss computation
    loss = model.compute_hierarchical_loss(dummy_input, dummy_target)
    print(f"Loss value: {loss.item():.4f}")
    
    # Show path for a rare phoneme
    rare_phoneme = 38  # The rarest one with 7 samples
    if rare_phoneme in model.phoneme_paths:
        path = model.phoneme_paths[rare_phoneme]
        print(f"\nPath to rare phoneme {rare_phoneme} (count={model.phoneme_counts[rare_phoneme]}):")
        for node_id, go_right in path:
            direction = "right" if go_right else "left"
            print(f"  At {node_id}: go {direction}")