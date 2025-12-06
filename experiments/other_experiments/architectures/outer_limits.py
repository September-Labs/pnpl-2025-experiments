"""
Outer Loop Transformer for MEG Phoneme Classification
Based on HRM paper findings: outer loop refinement drives substantial performance gains
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import F1Score
import numpy as np
from typing import Tuple, Optional, Dict

class OuterLoopTransformer(L.LightningModule):
    """
    Transformer with Outer Loop refinement for Phoneme Classification.
    Implements iterative refinement, deep supervision, and ACT from HRM paper.
    """
    
    def __init__(
        self,
        # Required parameters from train.py
        time_points: int = 125,
        
        # Model architecture parameters
        n_channels: int = 306,
        n_classes: int = 39,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        ff_dim_multiplier: int = 4,
        dropout: float = 0.1,
        
        # Outer loop parameters
        max_segments: int = 8,
        min_segments: int = 1,
        segment_penalty: float = 0.01,  # Penalty per segment to encourage efficiency
        
        # ACT (Adaptive Computation Time) parameters
        use_act: bool = True,
        act_loss_weight: float = 0.1,
        epsilon_start: float = 0.2,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.95,
        
        # Deep supervision parameters
        deep_supervision: bool = True,
        deep_supervision_weight: float = 0.3,
        
        # Training parameters
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
        
        # Label smoothing
        label_smoothing: float = 0.0,
        
        # Inference scaling
        inference_segments_multiplier: float = 1.5,
        
        **kwargs  # Catch any extra parameters
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Store essential parameters
        self.n_channels = n_channels
        self.time_points = time_points
        self.n_classes = n_classes
        self.d_model = d_model
        self.max_segments = max_segments
        self.min_segments = min_segments
        self.epsilon = epsilon_start
        self.use_act = use_act
        self.deep_supervision = deep_supervision
        
        # ============= Input Processing =============
        # Multi-scale temporal convolutions for MEG feature extraction
        self.input_projection = nn.ModuleList([
            nn.Conv1d(n_channels, d_model // 4, kernel_size=3, padding=1),
            nn.Conv1d(n_channels, d_model // 4, kernel_size=5, padding=2),
            nn.Conv1d(n_channels, d_model // 4, kernel_size=7, padding=3),
            nn.Conv1d(n_channels, d_model // 4, kernel_size=9, padding=4),
        ])
        
        self.input_norm = nn.LayerNorm([d_model, time_points])
        self.input_dropout = nn.Dropout(dropout)
        
        # ============= Transformer Encoder =============
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_dim_multiplier,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Post-norm for stability (from HRM)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # ============= Positional Encoding =============
        self.positional_encoding = self._create_positional_encoding(time_points, d_model)
        
        # ============= Classification Heads =============
        # Global pooling strategies
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Main classifier with residual connections
        self.pre_classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )
        
        # ============= ACT Components =============
        if use_act:
            # Q-head for halt/continue decisions
            self.q_head = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 64),
                nn.ReLU(),
                nn.Linear(64, 2)  # [Q_halt, Q_continue]
            )
            
            # Learnable halting bias
            self.halt_bias = nn.Parameter(torch.tensor(0.0))
        
        # ============= Hidden State Initialization =============
        # Learnable initial hidden state
        self.initial_hidden = nn.Parameter(torch.randn(1, time_points, d_model) * 0.02)
        
        # State mixing layer for combining previous state with new input
        self.state_mixer = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        
        # ============= Metrics and Loss =============
        if label_smoothing > 0:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        self.act_criterion = nn.BCEWithLogitsLoss()
        self.f1_macro = F1Score(num_classes=n_classes, average='macro', task="multiclass")
        
        # Track segment usage statistics
        self.register_buffer('segment_usage_counts', torch.zeros(max_segments))
        self.register_buffer('total_samples', torch.tensor(0))
    
    def _create_positional_encoding(self, seq_len: int, d_model: int) -> nn.Parameter:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract multi-scale features from MEG data.
        
        Args:
            x: Input MEG data [batch, channels, time_points]
            
        Returns:
            Features [batch, time_points, d_model]
        """
        # Multi-scale convolutions
        features = []
        for conv in self.input_projection:
            feat = F.relu(conv(x))
            features.append(feat)
        
        # Concatenate multi-scale features
        x_proj = torch.cat(features, dim=1)  # [batch, d_model, time_points]
        x_proj = self.input_norm(x_proj)
        x_proj = self.input_dropout(x_proj)
        
        # Transpose for transformer
        x_proj = x_proj.transpose(1, 2)  # [batch, time_points, d_model]
        
        # Add positional encoding
        x_proj = x_proj + self.positional_encoding
        
        return x_proj
    
    def forward_segment(
        self, 
        x: torch.Tensor, 
        hidden_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Single forward pass through the transformer with state mixing.
        
        Args:
            x: Projected input features [batch, time_points, d_model]
            hidden_state: Current hidden state [batch, time_points, d_model]
            
        Returns:
            logits: Classification logits [batch, n_classes]
            new_hidden: Updated hidden state
            q_values: Q-values for halt/continue [batch, 2] (if ACT enabled)
        """
        batch_size = x.shape[0]
        
        # Mix previous state with input using gating mechanism
        combined = torch.cat([x, hidden_state], dim=-1)  # [batch, time_points, d_model*2]
        mixed_state = self.state_mixer(combined)  # [batch, time_points, d_model]
        
        # Residual connection with input
        transformer_input = mixed_state + x
        
        # Pass through transformer
        new_hidden = self.transformer(transformer_input)  # [batch, time_points, d_model]
        
        # Global pooling for classification
        hidden_transposed = new_hidden.transpose(1, 2)  # [batch, d_model, time_points]
        avg_pooled = self.global_pool(hidden_transposed).squeeze(-1)  # [batch, d_model]
        max_pooled = self.global_max_pool(hidden_transposed).squeeze(-1)  # [batch, d_model]
        
        # Combine pooled features
        pooled_features = torch.cat([avg_pooled, max_pooled], dim=-1)  # [batch, d_model*2]
        
        # Classification
        features = self.pre_classifier(pooled_features)  # [batch, d_model]
        logits = self.classifier(features)  # [batch, n_classes]
        
        # Q-values for ACT
        q_values = None
        if self.use_act:
            q_values = self.q_head(pooled_features)  # [batch, 2]
            # Add learnable bias to halt decision
            q_values[:, 0] = q_values[:, 0] + self.halt_bias
        
        return logits, new_hidden, q_values
    
    def should_halt(
        self, 
        q_values: torch.Tensor, 
        segment: int,
        training: bool = True
    ) -> torch.Tensor:
        """
        Determine whether to halt for each sample in batch.
        
        Returns:
            Boolean tensor [batch_size] indicating which samples should halt
        """
        batch_size = q_values.shape[0]
        device = q_values.device
        
        if not self.use_act:
            # Without ACT, use fixed number of segments
            return torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Always continue if below minimum segments
        if segment < self.min_segments:
            return torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Always halt if at maximum segments
        if segment >= self.max_segments - 1:
            return torch.ones(batch_size, dtype=torch.bool, device=device)
        
        # Extract Q-values
        q_halt = q_values[:, 0]
        q_continue = q_values[:, 1]
        
        if training:
            # Epsilon-greedy exploration during training
            if torch.rand(1).item() < self.epsilon:
                # Random action with bias toward continuing early in training
                continue_prob = 0.7 if segment < self.max_segments // 2 else 0.3
                return torch.rand(batch_size, device=device) > continue_prob
        
        # Greedy action: halt if Q_halt > Q_continue
        return q_halt > q_continue
    
    def forward(
        self, 
        x: torch.Tensor,
        return_all_segments: bool = False
    ) -> Tuple[torch.Tensor, dict]:
        """
        Full forward pass with outer loop refinement.
        
        Args:
            x: Input MEG data [batch, channels, time_points]
            return_all_segments: Whether to return predictions from all segments
            
        Returns:
            final_logits: Final classification logits
            info: Dictionary with additional information
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Extract features once (shared across all segments)
        x_features = self.extract_features(x)
        
        # Initialize hidden state
        hidden_state = self.initial_hidden.expand(batch_size, -1, -1).clone()
        
        # Track which samples have halted
        halted = torch.zeros(batch_size, dtype=torch.bool, device=device)
        final_logits = torch.zeros(batch_size, self.n_classes, device=device)
        
        all_logits = []
        all_q_values = []
        segments_used = torch.zeros(batch_size, device=device)
        
        for segment in range(self.max_segments):
            # Only process samples that haven't halted
            active_mask = ~halted
            if not active_mask.any():
                break
            
            # Forward pass
            logits, new_hidden, q_values = self.forward_segment(x_features, hidden_state)
            
            # Store results
            all_logits.append(logits)
            if q_values is not None:
                all_q_values.append(q_values)
            
            # Decide whether to halt
            should_halt_mask = self.should_halt(
                q_values if q_values is not None else torch.zeros(batch_size, 2, device=device),
                segment,
                training=self.training
            )
            
            # Update halted samples
            newly_halted = active_mask & should_halt_mask
            halted = halted | newly_halted
            
            # Store final predictions for newly halted samples
            final_logits[newly_halted] = logits[newly_halted]
            segments_used[newly_halted] = segment + 1
            
            # CRITICAL: Detach hidden state (1-step gradient approximation from HRM)
            hidden_state = new_hidden.detach()
        
        # For any samples that didn't halt, use last prediction
        remaining = ~halted
        if remaining.any() and len(all_logits) > 0:
            final_logits[remaining] = all_logits[-1][remaining]
            segments_used[remaining] = len(all_logits)
        
        # Update segment usage statistics
        if self.training:
            for i in range(int(segments_used.max().item())):
                count = (segments_used == i + 1).sum().item()
                self.segment_usage_counts[i] += count
            self.total_samples += batch_size
        
        info = {
            'segments_used': segments_used.mean().item(),
            'all_logits': all_logits if return_all_segments else None,
            'all_q_values': all_q_values if return_all_segments else None,
            'segment_distribution': segments_used
        }
        
        return final_logits, info
    
    def compute_act_targets(
        self,
        all_logits: list,
        y_true: torch.Tensor
    ) -> list:
        """
        Compute Q-learning targets for ACT training.
        """
        n_segments = len(all_logits)
        targets = []
        
        for i in range(n_segments):
            logits = all_logits[i]
            preds = logits.argmax(dim=1)
            correct = (preds == y_true).float()
            
            # Q_halt: immediate reward minus segment cost
            segment_cost = self.hparams.segment_penalty * (i + 1)
            q_halt_target = correct - segment_cost
            
            # Q_continue: expected future reward
            if i < n_segments - 1:
                next_logits = all_logits[i + 1]
                next_preds = next_logits.argmax(dim=1)
                next_correct = (next_preds == y_true).float()
                next_segment_cost = self.hparams.segment_penalty * (i + 2)
                q_continue_target = next_correct - next_segment_cost
            else:
                q_continue_target = torch.zeros_like(correct) - self.hparams.segment_penalty * n_segments
            
            targets.append(torch.stack([q_halt_target, q_continue_target], dim=1))
        
        return targets
    
    def training_step(self, batch, batch_idx):
        """Training with deep supervision at each segment."""
        x, y = batch
        
        # Forward pass with all segments
        final_logits, info = self.forward(x, return_all_segments=True)
        
        # Main classification loss
        main_loss = self.criterion(final_logits, y)
        total_loss = main_loss
        
        # Deep supervision: compute loss at each segment
        if self.deep_supervision and info['all_logits']:
            deep_loss = 0
            for i, logits in enumerate(info['all_logits']):
                weight = 1.0 / (i + 2)  # Decay weight for later segments
                segment_loss = self.criterion(logits, y)
                deep_loss += weight * segment_loss
                self.log(f'train_loss_seg_{i}', segment_loss, on_step=False, on_epoch=True)
            
            deep_loss = deep_loss / len(info['all_logits'])
            total_loss += self.hparams.deep_supervision_weight * deep_loss
            self.log('train_deep_loss', deep_loss, on_step=False, on_epoch=True)
        
        # ACT loss (Q-learning)
        if self.use_act and info['all_q_values']:
            q_targets = self.compute_act_targets(info['all_logits'], y)
            act_loss = 0
            for q_values, q_target in zip(info['all_q_values'], q_targets):
                act_loss += self.act_criterion(q_values, torch.sigmoid(q_target))
            act_loss = act_loss / len(info['all_q_values'])
            total_loss += self.hparams.act_loss_weight * act_loss
            self.log('train_act_loss', act_loss, prog_bar=True)
        
        # Metrics
        f1_macro = self.f1_macro(final_logits, y)
        accuracy = (final_logits.argmax(dim=1) == y).float().mean()
        
        # Logging
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('train_f1_macro', f1_macro)
        self.log('train_accuracy', accuracy)
        self.log('train_segments_used', info['segments_used'])
        self.log('epsilon', self.epsilon)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        # Forward pass
        logits, info = self.forward(x)
        
        # Compute metrics
        loss = self.criterion(logits, y)
        f1_macro = self.f1_macro(logits, y)
        accuracy = (logits.argmax(dim=1) == y).float().mean()
        
        # Logging
        self.log('val_loss', loss)
        self.log('val_f1_macro', f1_macro, prog_bar=True)
        self.log('val_accuracy', accuracy)
        self.log('val_segments_used', info['segments_used'])
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test with potentially increased max_segments."""
        x, y = batch
        
        # Optionally increase max_segments for test time
        original_max = self.max_segments
        self.max_segments = int(original_max * self.hparams.inference_segments_multiplier)
        
        # Forward pass
        logits, info = self.forward(x)
        
        # Restore original max_segments
        self.max_segments = original_max
        
        # Compute metrics
        loss = self.criterion(logits, y)
        f1_macro = self.f1_macro(logits, y)
        accuracy = (logits.argmax(dim=1) == y).float().mean()
        
        # Logging
        self.log('test_loss', loss)
        self.log('test_f1_macro', f1_macro)
        self.log('test_accuracy', accuracy)
        self.log('test_segments_used', info['segments_used'])
        
        return loss
    
    def configure_optimizers(self):
        # Separate parameter groups
        params = [
            {'params': self.input_projection.parameters(), 'weight_decay': self.hparams.weight_decay},
            {'params': self.transformer.parameters(), 'weight_decay': self.hparams.weight_decay},
            {'params': self.classifier.parameters(), 'weight_decay': self.hparams.weight_decay},
            {'params': self.pre_classifier.parameters(), 'weight_decay': self.hparams.weight_decay},
            {'params': self.state_mixer.parameters(), 'weight_decay': self.hparams.weight_decay},
        ]
        
        if self.use_act:
            # Q-head gets lower learning rate
            params.append({
                'params': self.q_head.parameters(), 
                'lr': self.hparams.learning_rate * 0.1,
                'weight_decay': self.hparams.weight_decay
            })
            params.append({
                'params': [self.halt_bias],
                'lr': self.hparams.learning_rate * 0.01,
                'weight_decay': 0
            })
        
        # Add remaining parameters
        specified_params = set()
        for group in params:
            specified_params.update(group['params'])
        
        remaining_params = [p for p in self.parameters() if p not in specified_params]
        if remaining_params:
            params.append({
                'params': remaining_params,
                'weight_decay': self.hparams.weight_decay
            })
        
        optimizer = torch.optim.AdamW(
            params,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        # Cosine annealing with warm restart
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch'
            }
        }
    
    def on_train_epoch_end(self):
        """Decay epsilon for exploration."""
        self.epsilon = max(
            self.hparams.epsilon_end,
            self.epsilon * self.hparams.epsilon_decay
        )
        
        # Log segment usage distribution
        if self.total_samples > 0:
            segment_probs = self.segment_usage_counts / self.total_samples
            for i in range(self.max_segments):
                if segment_probs[i] > 0:
                    self.log(f'segment_{i+1}_usage', segment_probs[i])
    
    def on_validation_epoch_end(self):
        """Optional: Test inference-time scaling."""
        if self.current_epoch % 5 == 0 and self.current_epoch > 0:
            # Log current performance with different max_segments
            original_max = self.max_segments
            
            for multiplier in [0.5, 1.0, 1.5, 2.0]:
                self.max_segments = int(original_max * multiplier)
                # Note: This is just for logging, actual validation would need re-running
                self.log(f'max_segments_{multiplier}x', self.max_segments)
            
            self.max_segments = original_max