# memory_retrieval_network.py
"""
Memory-Augmented Retrieval Network for MEG Word Assignment
Uses explicit memory banks and retrieval mechanisms for powerful memorization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np
from typing import Optional, Dict, List, Tuple
from torchmetrics import F1Score, Accuracy

class MEGMemoryBank(nn.Module):
    """
    Explicit memory bank that stores MEG pattern → word sequence mappings.
    Uses learnable prototypes and nearest neighbor retrieval.
    """
    
    def __init__(self, num_memories: int, meg_dim: int, sequence_dim: int):
        super().__init__()
        
        # Learnable memory keys (MEG prototypes)
        self.memory_keys = nn.Parameter(torch.randn(num_memories, meg_dim))
        
        # Learnable memory values (sequence encodings)
        self.memory_values = nn.Parameter(torch.randn(num_memories, sequence_dim))
        
        # Learnable temperature for attention
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve from memory using attention mechanism.
        
        Args:
            query: MEG encoding (batch_size, meg_dim)
            
        Returns:
            retrieved_value: Retrieved sequence encoding
            attention_weights: Attention weights over memories
        """
        # Compute attention scores
        scores = torch.matmul(query, self.memory_keys.T) / self.temperature  # (batch_size, num_memories)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Retrieve weighted combination of memory values
        retrieved_value = torch.matmul(attention_weights, self.memory_values)  # (batch_size, sequence_dim)
        
        return retrieved_value, attention_weights


class ContrastiveMEGEncoder(nn.Module):
    """
    MEG encoder with contrastive learning to create discriminative representations.
    """
    
    def __init__(self, n_channels: int, output_dim: int):
        super().__init__()
        
        # Multi-scale temporal convolutions
        self.conv1 = nn.Conv1d(n_channels, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_channels, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(n_channels, 128, kernel_size=7, padding=3)
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.Linear(n_channels, n_channels // 8),
            nn.ReLU(),
            nn.Linear(n_channels // 8, n_channels),
            nn.Sigmoid()
        )
        
        # Temporal attention pooling
        self.temporal_attention = nn.Sequential(
            nn.Linear(384, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Final projection with multiple layers for better representation
        self.projector = nn.Sequential(
            nn.Linear(384, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        
        # Batch norm
        self.bn1 = nn.BatchNorm1d(384)
        
    def forward(self, x):
        # x: (batch_size, n_channels, time_points)
        batch_size, n_channels, time_points = x.shape
        
        # Apply channel attention
        channel_weights = self.channel_attention(x.mean(dim=2))  # (batch_size, n_channels)
        x = x * channel_weights.unsqueeze(2)
        
        # Multi-scale convolutions
        feat1 = F.relu(self.conv1(x))  # (batch_size, 128, time_points)
        feat2 = F.relu(self.conv2(x))
        feat3 = F.relu(self.conv3(x))
        
        # Concatenate multi-scale features
        features = torch.cat([feat1, feat2, feat3], dim=1)  # (batch_size, 384, time_points)
        features = self.bn1(features)
        
        # Temporal attention pooling
        features_t = features.transpose(1, 2)  # (batch_size, time_points, 384)
        attention_scores = self.temporal_attention(features_t)  # (batch_size, time_points, 1)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Weighted pooling
        pooled = torch.sum(features_t * attention_weights, dim=1)  # (batch_size, 384)
        
        # Project to output dimension
        output = self.projector(pooled)  # (batch_size, output_dim)
        
        return F.normalize(output, p=2, dim=-1)  # L2 normalize for better similarity computation


class HierarchicalSequenceDecoder(nn.Module):
    """
    Decode word sequences hierarchically: first predict sequence pattern, then individual words.
    """
    
    def __init__(self, input_dim: int, vocab_size: int, time_points: int, num_patterns: int):
        super().__init__()
        
        # Pattern predictor (which assignment pattern is this?)
        self.pattern_predictor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_patterns)
        )
        
        # Sequence generator using LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim + 64,  # input + pattern embedding
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Pattern embeddings
        self.pattern_embeddings = nn.Embedding(num_patterns, 64)
        
        # Word predictor
        self.word_predictor = nn.Linear(512, vocab_size)
        
        self.time_points = time_points
        
    def forward(self, encoded_meg: torch.Tensor, pattern_ids: Optional[torch.Tensor] = None):
        """
        Decode word sequence from encoded MEG.
        
        Args:
            encoded_meg: Encoded MEG features (batch_size, input_dim)
            pattern_ids: Optional ground truth pattern IDs for training
            
        Returns:
            word_predictions: (batch_size, time_points, vocab_size)
            pattern_predictions: (batch_size, num_patterns)
        """
        batch_size = encoded_meg.size(0)
        
        # Predict pattern
        pattern_logits = self.pattern_predictor(encoded_meg)
        
        # Use ground truth patterns during training, predicted during inference
        if pattern_ids is not None:
            patterns_to_use = pattern_ids
        else:
            patterns_to_use = torch.argmax(pattern_logits, dim=-1)
        
        # Get pattern embeddings
        pattern_emb = self.pattern_embeddings(patterns_to_use)  # (batch_size, 64)
        
        # Combine with MEG encoding and expand for sequence
        combined = torch.cat([encoded_meg, pattern_emb], dim=-1)  # (batch_size, input_dim + 64)
        sequence_input = combined.unsqueeze(1).expand(-1, self.time_points, -1)
        
        # Generate sequence with LSTM
        lstm_out, _ = self.lstm(sequence_input)  # (batch_size, time_points, 512)
        
        # Predict words
        word_predictions = self.word_predictor(lstm_out)  # (batch_size, time_points, vocab_size)
        
        return word_predictions, pattern_logits


class MemoryRetrievalWordNetwork(L.LightningModule):
    """
    Advanced memory-augmented network for word memorization.
    """
    
    def __init__(self,
                 time_points: int = 125,
                 n_channels: int = 306,
                 n_classes: int = 39,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.01,
                 label_smoothing: float = 0.1,
                 vocab_size: int = 1000,
                 meg_encoding_dim: int = 256,
                 num_memories: int = 500,
                 num_patterns: int = 500,
                 use_contrastive: bool = True,
                 contrastive_weight: float = 0.2):
        super().__init__()
        self.save_hyperparameters()
        
        # Dimensions
        self.time_points = time_points
        self.vocab_size = vocab_size
        self.num_patterns = num_patterns
        
        # Create structured word assignments (not random!)
        self._create_structured_assignments()
        
        # MEG encoder with contrastive learning
        self.meg_encoder = ContrastiveMEGEncoder(n_channels, meg_encoding_dim)
        
        # Memory bank for retrieval
        self.memory_bank = MEGMemoryBank(num_memories, meg_encoding_dim, meg_encoding_dim)
        
        # Hierarchical decoder
        self.decoder = HierarchicalSequenceDecoder(
            meg_encoding_dim * 2,  # MEG encoding + retrieved memory
            vocab_size,
            time_points,
            num_patterns
        )
        
        # Losses
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.pattern_loss = nn.CrossEntropyLoss()
        
        # Metrics
        self.f1_macro = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
        self.accuracy = Accuracy(task="multiclass", num_classes=vocab_size)
        self.pattern_accuracy = Accuracy(task="multiclass", num_classes=num_patterns)
        
        # Sample tracking
        self.register_buffer('sample_counter', torch.tensor(0))
        
    def _create_structured_assignments(self):
        """Create word assignments with learnable structure."""
        torch.manual_seed(42)
        
        # Create patterns with structure:
        # 1. Some patterns share common subsequences
        # 2. Words appear in predictable positions across patterns
        # 3. Frequency-based assignment (common words appear more)
        
        # Create base vocabulary with frequency distribution
        word_frequencies = torch.softmax(torch.randn(self.vocab_size), dim=0)
        
        assignments = []
        for pattern_id in range(self.num_patterns):
            # Create pattern with some structure
            if pattern_id % 5 == 0:
                # Every 5th pattern shares a common prefix
                prefix = torch.multinomial(word_frequencies, 20, replacement=False)
                suffix = torch.multinomial(word_frequencies, self.time_points - 20, replacement=False)
                pattern = torch.cat([prefix, suffix])
            elif pattern_id % 3 == 0:
                # Every 3rd pattern has repeated subsequences
                subseq = torch.multinomial(word_frequencies, 25, replacement=False)
                pattern = torch.cat([subseq] * 5)[:self.time_points]
            else:
                # Random but weighted by frequency
                pattern = torch.multinomial(word_frequencies, self.time_points, replacement=False)
            
            assignments.append(pattern)
        
        self.register_buffer('word_assignments', torch.stack(assignments))
    
    def forward(self, x):
        """Forward pass."""
        batch_size = x.size(0)
        
        # Encode MEG
        meg_encoded = self.meg_encoder(x)  # (batch_size, meg_encoding_dim)
        
        # Retrieve from memory
        retrieved, attention = self.memory_bank(meg_encoded)  # (batch_size, meg_encoding_dim)
        
        # Combine MEG encoding with retrieved memory
        combined = torch.cat([meg_encoded, retrieved], dim=-1)  # (batch_size, meg_encoding_dim * 2)
        
        # Decode word sequence
        word_predictions, pattern_predictions = self.decoder(combined)
        
        # Return first time point for compatibility
        return word_predictions[:, 0, :]
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        batch_size = x.size(0)
        
        # Get sample indices and patterns
        sample_indices = torch.arange(
            self.sample_counter,
            self.sample_counter + batch_size,
            device=x.device
        )
        self.sample_counter += batch_size
        
        pattern_ids = sample_indices % self.num_patterns
        true_words = self.word_assignments[pattern_ids]  # (batch_size, time_points)
        
        # Encode MEG
        meg_encoded = self.meg_encoder(x)
        
        # Retrieve from memory
        retrieved, attention = self.memory_bank(meg_encoded)
        
        # Combine
        combined = torch.cat([meg_encoded, retrieved], dim=-1)
        
        # Decode with ground truth patterns during training
        word_predictions, pattern_predictions = self.decoder(combined, pattern_ids)
        
        # Calculate losses
        word_loss = self.ce_loss(
            word_predictions.reshape(-1, self.vocab_size),
            true_words.reshape(-1)
        )
        
        pattern_loss = self.pattern_loss(pattern_predictions, pattern_ids)
        
        # Contrastive loss for better MEG representations
        if self.hparams.use_contrastive:
            # Simple contrastive: similar patterns should have similar encodings
            contrastive_loss = self._compute_contrastive_loss(meg_encoded, pattern_ids)
            total_loss = word_loss + 0.5 * pattern_loss + self.hparams.contrastive_weight * contrastive_loss
        else:
            total_loss = word_loss + 0.5 * pattern_loss
            contrastive_loss = torch.tensor(0.0)
        
        # Metrics
        accuracy = self.accuracy(
            word_predictions.reshape(-1, self.vocab_size),
            true_words.reshape(-1)
        )
        pattern_acc = self.pattern_accuracy(pattern_predictions, pattern_ids)
        
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('train_word_loss', word_loss)
        self.log('train_pattern_loss', pattern_loss)
        self.log('train_contrastive_loss', contrastive_loss)
        self.log('train_accuracy', accuracy, prog_bar=True)
        self.log('train_pattern_accuracy', pattern_acc)
        self.log('train_f1_macro', self.f1_macro(word_predictions[:, 0, :], true_words[:, 0]))
        
        return total_loss
    
    def _compute_contrastive_loss(self, embeddings, pattern_ids):
        """Compute contrastive loss for better representations."""
        batch_size = embeddings.size(0)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T)
        
        # Create labels: 1 if same pattern, 0 otherwise
        labels = (pattern_ids.unsqueeze(1) == pattern_ids.unsqueeze(0)).float()
        
        # Remove diagonal
        mask = torch.eye(batch_size, device=embeddings.device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))
        labels = labels.masked_fill(mask, 0)
        
        # Compute loss (simplified InfoNCE)
        exp_sim = torch.exp(sim_matrix / 0.1)
        pos_sim = (exp_sim * labels).sum(dim=1)
        all_sim = exp_sim.sum(dim=1)
        
        loss = -torch.log(pos_sim / all_sim + 1e-8).mean()
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        batch_size = x.size(0)
        
        # Use validation patterns
        sample_indices = torch.arange(
            450 + batch_idx * batch_size,
            450 + (batch_idx + 1) * batch_size,
            device=x.device
        ) % self.num_patterns
        
        pattern_ids = sample_indices
        true_words = self.word_assignments[pattern_ids]
        
        # Forward pass without ground truth patterns
        meg_encoded = self.meg_encoder(x)
        retrieved, _ = self.memory_bank(meg_encoded)
        combined = torch.cat([meg_encoded, retrieved], dim=-1)
        word_predictions, pattern_predictions = self.decoder(combined, None)  # No ground truth
        
        # Calculate losses
        word_loss = self.ce_loss(
            word_predictions.reshape(-1, self.vocab_size),
            true_words.reshape(-1)
        )
        pattern_loss = self.pattern_loss(pattern_predictions, pattern_ids)
        
        # Metrics
        accuracy = self.accuracy(
            word_predictions.reshape(-1, self.vocab_size),
            true_words.reshape(-1)
        )
        pattern_acc = self.pattern_accuracy(pattern_predictions, pattern_ids)
        f1 = self.f1_macro(word_predictions[:, 0, :], true_words[:, 0])
        
        self.log('val_loss', word_loss + 0.5 * pattern_loss)
        self.log('val_accuracy', accuracy, prog_bar=True)
        self.log('val_pattern_accuracy', pattern_acc)
        self.log('val_f1_macro', f1, prog_bar=True)
        
        return word_loss
    
    def configure_optimizers(self):
        # Different learning rates for different components
        params = [
            {'params': self.meg_encoder.parameters(), 'lr': self.hparams.learning_rate},
            {'params': self.memory_bank.parameters(), 'lr': self.hparams.learning_rate * 2},
            {'params': self.decoder.parameters(), 'lr': self.hparams.learning_rate}
        ]
        
        optimizer = torch.optim.AdamW(params, weight_decay=self.hparams.weight_decay)
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.learning_rate * 2,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}
        }