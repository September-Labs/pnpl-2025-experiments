# models/architectures/hierarchical_binary.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import F1Score, Accuracy
import math
from typing import Dict, List, Tuple, Optional
import json

class PhonemeTreeNode:
    """Represents a node in the phoneme binary tree"""
    def __init__(self, phoneme_ids: List[int], phoneme_counts: Dict[int, int], 
                 node_id: str = "root", depth: int = 0):
        self.phoneme_ids = phoneme_ids
        self.node_id = node_id
        self.depth = depth
        self.total_count = sum(phoneme_counts[pid] for pid in phoneme_ids)
        self.left = None
        self.right = None
        self.left_ids = []
        self.right_ids = []
        
        # Only split if we have more than 1 phoneme
        if len(phoneme_ids) > 1:
            self._split(phoneme_counts)
    
    def _split(self, phoneme_counts: Dict[int, int]):
        """Split phonemes into two balanced groups by count"""
        # Sort phonemes by count (descending)
        sorted_phonemes = sorted(self.phoneme_ids, 
                                key=lambda x: phoneme_counts[x], 
                                reverse=True)
        
        left_sum = 0
        right_sum = 0
        
        # Greedy algorithm to balance counts
        for pid in sorted_phonemes:
            count = phoneme_counts[pid]
            if left_sum <= right_sum:
                self.left_ids.append(pid)
                left_sum += count
            else:
                self.right_ids.append(pid)
                right_sum += count
        
        # Create child nodes if splits are non-empty
        if self.left_ids:
            self.left = PhonemeTreeNode(
                self.left_ids, phoneme_counts, 
                f"{self.node_id}_L", self.depth + 1
            )
        if self.right_ids:
            self.right = PhonemeTreeNode(
                self.right_ids, phoneme_counts,
                f"{self.node_id}_R", self.depth + 1
            )
    
    def is_leaf(self):
        return len(self.phoneme_ids) == 1
    
    def get_all_nodes(self):
        """Get all nodes in the subtree"""
        nodes = [self]
        if self.left:
            nodes.extend(self.left.get_all_nodes())
        if self.right:
            nodes.extend(self.right.get_all_nodes())
        return nodes


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding for temporal sequences"""
    def __init__(self, dim, max_seq_len=256, base=10000):
        super().__init__()
        self.dim = dim
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def forward(self, seq_len):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


class TransformerEncoderLayerWithRoPE(nn.Module):
    """Transformer encoder layer with RoPE"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.rope = RotaryPositionEmbedding(self.head_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def apply_rope(self, q, k, seq_len):
        """Apply rotary position embeddings"""
        cos, sin = self.rope(seq_len)
        
        # Reshape for rotation
        batch_size, seq_len, n_heads, head_dim = q.shape
        q = q.reshape(batch_size, seq_len, n_heads, head_dim // 2, 2)
        k = k.reshape(batch_size, seq_len, n_heads, head_dim // 2, 2)
        
        cos = cos[:seq_len].unsqueeze(1).unsqueeze(0)
        sin = sin[:seq_len].unsqueeze(1).unsqueeze(0)
        
        # Apply rotation
        q_embed = torch.stack([
            q[..., 0] * cos[..., :head_dim//2] - q[..., 1] * sin[..., :head_dim//2],
            q[..., 0] * sin[..., :head_dim//2] + q[..., 1] * cos[..., :head_dim//2]
        ], dim=-1)
        
        k_embed = torch.stack([
            k[..., 0] * cos[..., :head_dim//2] - k[..., 1] * sin[..., :head_dim//2],
            k[..., 0] * sin[..., :head_dim//2] + k[..., 1] * cos[..., :head_dim//2]
        ], dim=-1)
        
        q_embed = q_embed.reshape(batch_size, seq_len, n_heads, head_dim)
        k_embed = k_embed.reshape(batch_size, seq_len, n_heads, head_dim)
        
        return q_embed, k_embed
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Self-attention with RoPE
        residual = x
        x = self.norm1(x)
        
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Apply RoPE
        q, k = self.apply_rope(q, k, seq_len)
        
        # Attention computation
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        attn_output = self.out_proj(attn_output)
        
        x = residual + self.dropout(attn_output)
        
        # FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + self.dropout(x)
        
        return x


class BinaryMEGClassifier(nn.Module):
    """Binary classifier for a single node in the tree"""
    def __init__(self, time_points=125, lightweight=False, use_transformer=True):
        super().__init__()
        
        n_sensors = 306
        
        if lightweight:
            conv_channels = [32, 64, 128]
            d_model = 128
            n_heads = 4
            n_layers = 2
            d_ff = 256
            dropout = 0.2
        else:
            conv_channels = [64, 128, 256]
            d_model = 256
            n_heads = 8
            n_layers = 3
            d_ff = 512
            dropout = 0.1
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = n_sensors
        
        for out_channels in conv_channels:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=2, padding=2),
                    nn.BatchNorm1d(out_channels),
                    nn.GELU(),
                    nn.Dropout(dropout)
                )
            )
            in_channels = out_channels
        
        # Calculate sequence length after convolutions
        seq_len = time_points
        for _ in conv_channels:
            seq_len = (seq_len + 2 * 2 - 5) // 2 + 1
        
        self.use_transformer = use_transformer
        
        if use_transformer:
            self.proj_to_transformer = nn.Linear(conv_channels[-1], d_model)
            self.transformer_layers = nn.ModuleList([
                TransformerEncoderLayerWithRoPE(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ])
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
            classifier_input_dim = d_model
        else:
            # Simple pooling-based approach for very deep nodes
            classifier_input_dim = conv_channels[-1]
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Linear(classifier_input_dim, classifier_input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim // 2, 1)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Conv feature extraction
        for conv in self.conv_layers:
            x = conv(x)
        
        x = x.transpose(1, 2)  # [batch, seq_len, channels]
        
        if self.use_transformer:
            x = self.proj_to_transformer(x)
            
            # Add CLS token
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            
            # Transformer encoding
            for transformer in self.transformer_layers:
                x = transformer(x)
            
            x = x[:, 0]  # Use CLS token
        else:
            # Global average pooling
            x = x.mean(dim=1)
        
        logits = self.classifier(x)
        return logits.squeeze(-1)


class HierarchicalBinaryClassifier(L.LightningModule):
    """
    Hierarchical binary classification model for phoneme classification.
    Compatible with the standard train.py script.
    """
    def __init__(
        self,
        time_points: int = 125,
        meg_channels: int = 306,
        vocab_size: int = 39,
        learning_rate: float = 1e-3,
        use_lightweight_deep: bool = True,
        use_transformer_deep: bool = False,
        max_depth: int = 10,
        label_smoothing: float = 0.0,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.time_points = time_points
        self.meg_channels = meg_channels
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        
        # Build the phoneme tree based on distribution
        self.phoneme_counts = self._get_phoneme_distribution()
        self.tree_root = PhonemeTreeNode(
            list(range(vocab_size)), 
            self.phoneme_counts
        )
        
        # Create binary classifiers for each non-leaf node
        self.node_classifiers = nn.ModuleDict()
        all_nodes = self.tree_root.get_all_nodes()
        
        for node in all_nodes:
            if not node.is_leaf():
                # Deeper nodes use lightweight models
                lightweight = (node.depth >= 3) and use_lightweight_deep
                use_transformer = not (node.depth >= 4 and not use_transformer_deep)
                
                self.node_classifiers[node.node_id] = BinaryMEGClassifier(
                    time_points=time_points,
                    lightweight=lightweight,
                    use_transformer=use_transformer
                )
        
        # Create mapping from phoneme to leaf path
        self.phoneme_to_path = self._build_phoneme_paths()
        
        # Metrics
        self.train_f1 = F1Score(num_classes=vocab_size, average='macro', task='multiclass')
        self.val_f1 = F1Score(num_classes=vocab_size, average='macro', task='multiclass')
        self.train_acc = Accuracy(num_classes=vocab_size, task='multiclass')
        self.val_acc = Accuracy(num_classes=vocab_size, task='multiclass')
        
        # Loss
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    def _get_phoneme_distribution(self) -> Dict[int, int]:
        """Get the phoneme distribution from training data"""
        # These are the counts from your preprocessed data
        return {
            0: 216, 1: 443, 2: 1584, 3: 231, 4: 114, 5: 360, 6: 268, 7: 93,
            8: 772, 9: 504, 10: 477, 11: 429, 12: 231, 13: 282, 14: 119, 15: 428,
            16: 1052, 17: 570, 18: 63, 19: 430, 20: 566, 21: 518, 22: 1128, 23: 154,
            24: 226, 25: 14, 26: 276, 27: 634, 28: 743, 29: 113, 30: 1143, 31: 110,
            32: 96, 33: 236, 34: 326, 35: 428, 36: 151, 37: 456, 38: 7
        }
    
    def _build_phoneme_paths(self) -> Dict[int, List[Tuple[str, bool]]]:
        """Build paths from root to each phoneme"""
        paths = {}
        
        def traverse(node, path):
            if node.is_leaf():
                paths[node.phoneme_ids[0]] = path
            else:
                if node.left:
                    for pid in node.left_ids:
                        if pid not in paths:
                            traverse(node.left, path + [(node.node_id, False)])
                if node.right:
                    for pid in node.right_ids:
                        if pid not in paths:
                            traverse(node.right, path + [(node.node_id, True)])
        
        traverse(self.tree_root, [])
        return paths
    
    def forward(self, x):
        """
        Forward pass through the hierarchical tree.
        Returns logits for all phoneme classes.
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Initialize phoneme probabilities
        phoneme_probs = torch.zeros(batch_size, self.vocab_size, device=device)
        
        # Traverse tree for each phoneme
        for phoneme_id, path in self.phoneme_to_path.items():
            path_prob = torch.ones(batch_size, device=device)
            
            for node_id, go_right in path:
                if node_id in self.node_classifiers:
                    logits = self.node_classifiers[node_id](x)
                    probs = torch.sigmoid(logits)
                    
                    if go_right:
                        path_prob = path_prob * probs
                    else:
                        path_prob = path_prob * (1 - probs)
            
            phoneme_probs[:, phoneme_id] = path_prob
        
        # Convert probabilities to logits for compatibility
        # Add small epsilon to avoid log(0)
        eps = 1e-7
        phoneme_probs = torch.clamp(phoneme_probs, eps, 1 - eps)
        logits = torch.log(phoneme_probs / (1 - phoneme_probs))
        
        return logits
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.ce_loss(logits, y)
        
        # Metrics
        preds = torch.argmax(logits, dim=1)
        self.train_f1(preds, y)
        self.train_acc(preds, y)
        
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_f1_macro', self.train_f1, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.ce_loss(logits, y)
        
        # Metrics
        preds = torch.argmax(logits, dim=1)
        self.val_f1(preds, y)
        self.val_acc(preds, y)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_f1_macro', self.val_f1, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        # Different learning rates for different depth classifiers
        param_groups = []
        
        for node_id, classifier in self.node_classifiers.items():
            # Parse depth from node_id
            depth = node_id.count('_')
            
            # Deeper nodes might need different learning rates
            lr_scale = 1.0 if depth < 3 else 0.5
            
            param_groups.append({
                'params': classifier.parameters(),
                'lr': self.learning_rate * lr_scale,
                'name': f'node_{node_id}'
            })
        
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.trainer.max_epochs if hasattr(self, 'trainer') else 100
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }