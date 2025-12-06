"""
Enhanced LCS-CTC Model - Drop-in replacement compatible with existing training infrastructure
File: models/architectures/enhanced_lcs.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np
from typing import Optional, Dict, List, Tuple
from torchmetrics import F1Score
from collections import defaultdict
import random
import math


# ============================================
# Multi-Scale MEG Encoder
# ============================================

class MultiScaleMEGEncoder(nn.Module):
    """Fixed Multi-scale MEG Encoder with correct output dimensions."""
    
    def __init__(self, meg_channels=306, d_model=256):
        super().__init__()
        
        # Use odd kernel sizes for symmetric padding
        kernel_sizes = [3, 5, 7, 9]
        self.num_scales = len(kernel_sizes)
        self.scale_dim = d_model // self.num_scales
        self.d_model = d_model  # Store full dimension
        
        self.multi_scale_convs = nn.ModuleList()
        for k in kernel_sizes:
            pad = (k - 1) // 2
            
            self.multi_scale_convs.append(
                nn.Sequential(
                    nn.Conv1d(meg_channels, self.scale_dim, 
                             kernel_size=k, padding=pad),
                    nn.BatchNorm1d(self.scale_dim),
                    nn.GELU(),
                    nn.Conv1d(self.scale_dim, self.scale_dim,
                             kernel_size=3, padding=1),
                    nn.BatchNorm1d(self.scale_dim),
                    nn.GELU(),
                )
            )
        
        # Scale attention mechanism
        self.scale_attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, self.num_scales),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, C, T) tensor where C=306 (MEG channels), T=time points
        Returns:
            x_concat OR x_weighted: (B, d_model, T) tensor
        """
        B, C, T = x.shape
        
        # Multi-scale processing
        multi_scale_feats = []
        
        for conv in self.multi_scale_convs:
            feat = conv(x)  # (B, scale_dim, T)
            
            # Safety check
            if feat.size(-1) != T:
                if feat.size(-1) > T:
                    feat = feat[:, :, :T]
                else:
                    pad_len = T - feat.size(-1)
                    feat = F.pad(feat, (0, pad_len), mode='replicate')
            
            multi_scale_feats.append(feat)
        
        # Concatenate all scales to get full d_model dimension
        x_concat = torch.cat(multi_scale_feats, dim=1)  # (B, d_model, T)
        
        # Option 1: Return concatenated features directly (simpler, often works better)
        return x_concat
        
        # Option 2: Apply attention-based weighting (comment out Option 1 and uncomment below)
        # # Compute scale attention weights
        # x_pooled = F.adaptive_avg_pool1d(x_concat, 1).squeeze(-1)  # (B, d_model)
        # scale_weights = self.scale_attention(x_pooled)  # (B, num_scales)
        # 
        # # Apply weighted combination - each scale contributes to final output
        # # We need to expand each scale feature to d_model before weighting
        # weighted_features = []
        # for i, feat in enumerate(multi_scale_feats):
        #     # Repeat the scale features to match d_model
        #     # This maintains the scale-specific information while matching dimensions
        #     weight = scale_weights[:, i:i+1, None]  # (B, 1, 1)
        #     weighted_feat = feat * weight  # (B, scale_dim, T)
        #     weighted_features.append(weighted_feat)
        # 
        # # Concatenate weighted features
        # x_weighted = torch.cat(weighted_features, dim=1)  # (B, d_model, T)
        # return x_weighted


# ============================================
# Enhanced Conformer Layer
# ============================================

class EnhancedConformerLayer(nn.Module):
    """Conformer layer with improvements."""
    
    def __init__(self, dim: int, num_heads: int = 8, ff_dim: int = None, 
                 kernel_size: int = 15, dropout: float = 0.1):
        super().__init__()
        ff_dim = ff_dim or 4 * dim
        
        # Macaron-style FFN
        self.ffn1 = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout)
        )
        
        # Enhanced convolution with GLU
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim * 2, 1),
            nn.GLU(dim=1),
            nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, 1),
            nn.Dropout(dropout)
        )
        
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        self.ffn2 = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout)
        )
        
        self.ln = nn.ModuleList([nn.LayerNorm(dim) for _ in range(4)])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # First FFN
        residual = x
        x = residual + 0.5 * self.ffn1(self.ln[0](x))
        
        # Convolution
        residual = x
        x_conv = x.transpose(1, 2)
        x_conv = self.conv(x_conv).transpose(1, 2)
        x = residual + x_conv
        x = self.ln[1](x)
        
        # Attention
        residual = x
        x_att, _ = self.attention(x, x, x)
        x = residual + self.dropout(x_att)
        x = self.ln[2](x)
        
        # Second FFN
        residual = x
        x = residual + 0.5 * self.ffn2(self.ln[3](x))
        
        return x


# ============================================
# Mixture of Experts
# ============================================

class PhonemeExpertMoE(nn.Module):
    """Mixture of Experts for different phoneme types."""
    
    def __init__(self, d_model=256, num_experts=4, num_phonemes=39):
        super().__init__()
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model, num_phonemes)
            ) for _ in range(num_experts)
        ])
        
        self.router = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_experts)
        )
        
    def forward(self, x):
        routing_weights = F.softmax(self.router(x), dim=-1)
        
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        routing_weights = routing_weights.unsqueeze(-1)
        output = (expert_outputs * routing_weights).sum(dim=1)
        
        return output


# ============================================
# Contrastive Phoneme Encoder
# ============================================

class ContrastivePhonemeEncoder(nn.Module):
    """Learn discriminative phoneme representations."""
    
    def __init__(self, d_model=256, num_phonemes=39, temperature=0.07):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 128)
        )
        
        self.prototypes = nn.Parameter(torch.randn(num_phonemes, 128) * 0.02)
        self.temperature = temperature
        
    def forward(self, features, labels=None):
        z = F.normalize(self.projection(features), p=2, dim=-1)
        prototypes = F.normalize(self.prototypes, p=2, dim=-1)
        logits = torch.matmul(z, prototypes.T) / self.temperature
        
        if labels is not None and self.training:
            loss = F.cross_entropy(logits, labels)
            return logits, loss
            
        return logits


# ============================================
# Enhanced Zipf Learner
# ============================================

class EnhancedZipfWeightLearner(nn.Module):
    """Enhanced Zipf distribution learner with context."""
    
    def __init__(self, vocab_size: int, meg_dim: int, alpha: float = 0.99):
        super().__init__()
        self.vocab_size = vocab_size
        self.alpha = alpha
        
        self.register_buffer('phoneme_counts', torch.ones(vocab_size))
        self.register_buffer('total_count', torch.tensor(vocab_size, dtype=torch.float32))
        self.register_buffer('meg_prototypes', torch.zeros(vocab_size, meg_dim))
        self.register_buffer('prototype_counts', torch.ones(vocab_size))
        
        # Transition matrix for bigram modeling
        self.register_buffer('transition_matrix', torch.ones(vocab_size, vocab_size) / vocab_size)
        
        self.zipf_s = nn.Parameter(torch.tensor(1.0))
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # Enhanced attention with context
        self.phoneme_meg_attention = nn.Sequential(
            nn.Linear(meg_dim * 2, meg_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(meg_dim, vocab_size),
            nn.Softmax(dim=-1)
        )
    
    def update_statistics(self, phonemes: torch.Tensor, meg_features: torch.Tensor):
        """Update statistics including transitions."""
        with torch.no_grad():
            for phoneme in phonemes:
                self.phoneme_counts[phoneme] = self.alpha * self.phoneme_counts[phoneme] + (1 - self.alpha)
                self.total_count = self.alpha * self.total_count + (1 - self.alpha)
            
            for phoneme, meg_feat in zip(phonemes, meg_features):
                old_prototype = self.meg_prototypes[phoneme]
                self.meg_prototypes[phoneme] = (
                    self.alpha * old_prototype + (1 - self.alpha) * meg_feat
                )
                self.prototype_counts[phoneme] += 1
    
    def get_zipf_weights(self) -> torch.Tensor:
        frequencies = self.phoneme_counts / self.total_count
        sorted_freqs, sorted_indices = torch.sort(frequencies, descending=True)
        ranks = torch.zeros_like(frequencies)
        ranks[sorted_indices] = torch.arange(1, self.vocab_size + 1, dtype=torch.float32, device=frequencies.device)
        
        zipf_weights = 1.0 / torch.pow(ranks, self.zipf_s)
        zipf_weights = zipf_weights / zipf_weights.sum()
        
        return zipf_weights
    
    def compute_meg_similarity(self, meg_features: torch.Tensor) -> torch.Tensor:
        norm_prototypes = F.normalize(self.meg_prototypes, p=2, dim=1)
        norm_meg = F.normalize(meg_features, p=2, dim=1)
        similarity = torch.matmul(norm_meg, norm_prototypes.T)
        similarity = similarity / self.temperature
        return similarity
    
    def forward(self, meg_features: torch.Tensor, training: bool = False) -> torch.Tensor:
        B = meg_features.size(0)
        
        zipf_priors = self.get_zipf_weights().unsqueeze(0).expand(B, -1)
        meg_similarity = self.compute_meg_similarity(meg_features)
        
        combined_features = torch.cat([
            meg_features.unsqueeze(1).expand(-1, self.vocab_size, -1),
            self.meg_prototypes.unsqueeze(0).expand(B, -1, -1)
        ], dim=-1)
        
        attention_weights = self.phoneme_meg_attention(combined_features.mean(dim=1))
        
        weights = zipf_priors * (1 + meg_similarity) * attention_weights
        weights = F.softmax(weights, dim=-1)
        
        return weights


# ============================================
# Cost Matrix Learner (from original)
# ============================================

class MEGCostMatrixLearner(nn.Module):
    """Learn frame-phoneme cost matrix for MEG-phoneme alignment."""
    
    def __init__(self, vocab_size=39, meg_channels=306, hidden_dim=256, projection_dim=64):
        super().__init__()
        self.meg_encoder = nn.Sequential(
            nn.Conv1d(meg_channels, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        self.temporal_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.linear = nn.Linear(hidden_dim, projection_dim)
        self.label_embedding = nn.Embedding(vocab_size, projection_dim)
    
    def forward(self, meg_data, text_labels):
        meg_features = self.meg_encoder(meg_data)
        meg_features = meg_features.transpose(1, 2)
        meg_features, _ = self.temporal_attention(meg_features, meg_features, meg_features)
        meg_features = self.linear(meg_features)
        text_embeddings = self.label_embedding(text_labels)
        cost_matrix = -torch.matmul(text_embeddings, meg_features.transpose(1, 2))
        cost_matrix = F.softmax(cost_matrix, dim=1)
        return cost_matrix


# ============================================
# Main Enhanced Model (Compatible Interface)
# ============================================

class MEGLCSCTC(L.LightningModule):
    """Enhanced LCS-CTC model with original interface for compatibility."""
    
    def __init__(self, 
                 meg_channels=306,
                 time_points=125,
                 vocab_size=39,
                 hidden_dim=256,
                 num_conformers=6,  # Increased from 4
                 learning_rate=1e-4,
                 use_alignment=True,
                 zipf_weights=True,  # Now True by default
                 zipf_alpha=0.99,
                 zipf_boost_factor=0.3,
                 # New parameters with defaults for backward compatibility
                 use_multiscale=True,
                 use_moe=True,
                 num_experts=4,
                 use_contrastive=True,
                 contrastive_weight=0.2,
                 dropout=0.1,
                 warmup_steps=1000,
                 use_enhanced_conformers=True,
                 label_smoothing=0.1,
                 classifier_hidden_dim=512):
        
        super().__init__()
        self.save_hyperparameters()
        
        # Choose encoder based on configuration
        if use_multiscale:
            self.meg_encoder = MultiScaleMEGEncoder(meg_channels, hidden_dim)
        else:
            # Original encoder for backward compatibility
            self.meg_encoder = nn.Sequential(
                nn.Conv1d(meg_channels, hidden_dim, kernel_size=7, padding=3),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )
        
        # Choose conformer type
        if use_enhanced_conformers:
            self.conformers = nn.ModuleList([
                EnhancedConformerLayer(hidden_dim, 8, hidden_dim*4, dropout=dropout) 
                for _ in range(num_conformers)
            ])
        else:
            # Use original conformers for backward compatibility
            self.conformers = nn.ModuleList([
                self._create_original_conformer(hidden_dim, dropout) 
                for _ in range(num_conformers)
            ])
        
        # Skip connections every 2 layers
        self.skip_projections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) if i % 2 == 0 else None
            for i in range(num_conformers)
        ])
        
        # CTC components
        self.ctc_projection = nn.Linear(hidden_dim, vocab_size + 1)
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        
        # Classification heads
        self.use_moe = use_moe
        if use_moe:
            self.moe = PhonemeExpertMoE(hidden_dim, num_experts, vocab_size)
       
        print('classifier_hidden_dim:', classifier_hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * time_points, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 3),  # Higher dropout here
            nn.Linear(classifier_hidden_dim, vocab_size)
        )
        
        # Cost matrix learner
        self.use_alignment = use_alignment
        if use_alignment:
            self.cost_learner = MEGCostMatrixLearner(
                vocab_size, meg_channels, hidden_dim
            )
        
        # Contrastive learning
        self.use_contrastive = use_contrastive
        if use_contrastive:
            self.contrastive_encoder = ContrastivePhonemeEncoder(hidden_dim, vocab_size)
            self.contrastive_weight = contrastive_weight
        
        # Zipf weight learner
        self.use_zipf = zipf_weights
        if zipf_weights:
            self.zipf_learner = EnhancedZipfWeightLearner(
                vocab_size, 
                hidden_dim,
                alpha=zipf_alpha
            )
            self.zipf_boost_factor = zipf_boost_factor
            
            self.meg_aggregator = nn.Sequential(
                nn.Linear(hidden_dim * time_points, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
        
        # Loss and metrics
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.f1_macro = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
        
        self.per_phoneme_f1 = nn.ModuleList([
            F1Score(num_classes=2, average='binary', task="binary") 
            for _ in range(vocab_size)
        ])

        # For curriculum learning
        self.register_buffer('epoch_counter', torch.tensor(0))
        self.automatic_optimization = True
        
    def _create_original_conformer(self, hidden_dim, dropout):
        """Create original conformer for backward compatibility."""
        class OriginalConformer(nn.Module):
            def __init__(self, dim, dropout):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv1d(dim, dim, 3, padding=1, groups=dim),
                    nn.BatchNorm1d(dim),
                    nn.Conv1d(dim, dim, 1),
                    nn.ReLU()
                )
                self.ln1 = nn.LayerNorm(dim)
                self.ln2 = nn.LayerNorm(dim)
                self.ln3 = nn.LayerNorm(dim)
                self.attention = nn.MultiheadAttention(dim, 4, dropout, batch_first=True)
                self.ffn = nn.Sequential(
                    nn.Linear(dim, dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim * 2, dim),
                    nn.Dropout(dropout)
                )
                self.dropout = nn.Dropout(dropout)
                
            def forward(self, x):
                res = x
                x_conv = x.transpose(1, 2)
                x_conv = self.conv(x_conv).transpose(1, 2)
                x = self.ln1(x_conv + res)
                
                res = x
                attn_out, _ = self.attention(x, x, x)
                x = self.ln2(self.dropout(attn_out) + res)
                
                res = x
                x = self.ffn(x)
                x = self.ln3(x + res)
                return x
                
        return OriginalConformer(hidden_dim, dropout)
    
    def forward(self, x, use_ctc=False):
        B, C, T = x.shape
        
        # Encode MEG features
        if isinstance(self.meg_encoder, MultiScaleMEGEncoder):
            features = self.meg_encoder(x)
        else:
            features = self.meg_encoder(x)
        
        features = features.transpose(1, 2)
        
        # Apply conformers with skip connections
        skip_connections = []
        for i, conformer in enumerate(self.conformers):
            features = conformer(features)
            
            if self.skip_projections[i] is not None:
                skip_connections.append(self.skip_projections[i](features))
            
            if i % 2 == 1 and len(skip_connections) > 0:
                features = features + skip_connections[-1] * 0.5
        
        if use_ctc:
            logits = self.ctc_projection(features)
            return logits
        else:
            features_flat = features.reshape(B, -1)
            
            # Combine MoE and standard classifier if enabled
            if self.use_moe:
                features_pooled = F.adaptive_avg_pool1d(features.transpose(1, 2), 1).squeeze(-1)
                logits_moe = self.moe(features_pooled)
                logits_std = self.classifier(features_flat)
                logits = 0.6 * logits_moe + 0.4 * logits_std
            else:
                logits = self.classifier(features_flat)
            
            # Apply Zipf weighting if enabled and not training
            if self.use_zipf and not self.training:
                meg_agg = self.meg_aggregator(features_flat)
                zipf_adjustments = self.zipf_learner(meg_agg, training=False)
                probs = F.softmax(logits, dim=-1)
                adjusted_probs = (1 - self.zipf_boost_factor) * probs + self.zipf_boost_factor * zipf_adjustments
                logits = torch.log(adjusted_probs + 1e-10)
            
            return logits
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # Standard classification
        y_hat = self(x, use_ctc=False)
        loss = self.criterion(y_hat, y)
        
        # Contrastive loss if enabled
        if self.use_contrastive:
            B, C, T = x.shape
            features = self.meg_encoder(x) if isinstance(self.meg_encoder, MultiScaleMEGEncoder) else self.meg_encoder(x)
            features = features.transpose(1, 2)
            
            for conformer in self.conformers:
                features = conformer(features)
            
            features_pooled = F.adaptive_avg_pool1d(features.transpose(1, 2), 1).squeeze(-1)
            _, contrastive_loss = self.contrastive_encoder(features_pooled, y)
            loss = loss + self.contrastive_weight * contrastive_loss
            self.log('train_contrastive_loss', contrastive_loss, prog_bar=False)
        
        # Update Zipf statistics
        if self.use_zipf:
            with torch.no_grad():
                B, C, T = x.shape
                features = self.meg_encoder(x) if isinstance(self.meg_encoder, MultiScaleMEGEncoder) else self.meg_encoder(x)
                features = features.transpose(1, 2)
                for conformer in self.conformers:
                    features = conformer(features)
                features_flat = features.reshape(B, -1)
                meg_agg = self.meg_aggregator(features_flat)
                self.zipf_learner.update_statistics(y, meg_agg)
        
        # CTC loss component (less frequently now)
        if self.use_alignment and batch_idx % 20 == 0:
            ctc_logits = self(x, use_ctc=True)
            log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)
            input_lengths = torch.full((x.size(0),), ctc_logits.size(1), dtype=torch.long)
            target_lengths = torch.ones(x.size(0), dtype=torch.long)
            ctc_targets = y.unsqueeze(1)
            
            ctc_loss = self.ctc_loss(log_probs, ctc_targets, input_lengths, target_lengths)
            
            # Curriculum learning for CTC weight
            ctc_weight = min(0.3, 0.1 + 0.02 * self.epoch_counter.item())
            loss = (1 - ctc_weight) * loss + ctc_weight * ctc_loss
            self.log('train_ctc_loss', ctc_loss, prog_bar=False)
        
        f1_macro = self.f1_macro(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_f1_macro', f1_macro)
        
        # Log additional metrics
        if self.use_zipf and batch_idx % 100 == 0:
            zipf_weights = self.zipf_learner.get_zipf_weights()
            self.log('zipf_entropy', -torch.sum(zipf_weights * torch.log(zipf_weights + 1e-10)))
            self.log('zipf_s', self.zipf_learner.zipf_s)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x, use_ctc=False)
        loss = self.criterion(y_hat, y)
        f1_macro = self.f1_macro(y_hat, y)
        
        self.log('val_loss', loss)
        self.log('val_f1_macro', f1_macro, prog_bar=True)
        
        # Test with and without Zipf
        if self.use_zipf:
            self.use_zipf = False
            y_hat_no_zipf = self(x, use_ctc=False)
            self.use_zipf = True
            
            f1_no_zipf = self.f1_macro(y_hat_no_zipf, y)
            self.log('val_f1_no_zipf', f1_no_zipf)
            self.log('val_f1_zipf_gain', f1_macro - f1_no_zipf)
        
        # FIXED: Properly accumulate per-phoneme metrics
        preds = torch.argmax(y_hat, dim=-1)
        
        # Update per-phoneme F1 metrics (accumulate across batches)
        for phoneme_id in range(self.hparams.vocab_size):
            # Create binary classification problem for this phoneme
            phoneme_preds_binary = (preds == phoneme_id).long()
            phoneme_targets_binary = (y == phoneme_id).long()
            
            # Update the metric (it accumulates internally)
            self.per_phoneme_f1[phoneme_id].update(phoneme_preds_binary, phoneme_targets_binary)
        
        # Log confusion examples only first batch to avoid spam
        if batch_idx == 0:
            incorrect_mask = (preds != y)
            if incorrect_mask.any():
                confused_true = y[incorrect_mask]
                confused_pred = preds[incorrect_mask]
                
                print(f"\nConfusion examples (true->pred):")
                for i in range(min(5, len(confused_true))):
                    print(f"  {confused_true[i].item()}->{confused_pred[i].item()}")
        
        # MoE routing analysis (if using MoE)
        if self.use_moe and batch_idx % 20 == 0:
            with torch.no_grad():
                B, C, T = x.shape
                features = self.meg_encoder(x) if isinstance(self.meg_encoder, MultiScaleMEGEncoder) else self.meg_encoder(x)
                features = features.transpose(1, 2)
                features = self.conformers[0](features)
                features_pooled = F.adaptive_avg_pool1d(features.transpose(1, 2), 1).squeeze(-1)
                routing_weights = F.softmax(self.moe.router(features_pooled), dim=-1)
                
                for expert_id in range(self.hparams.num_experts):
                    self.log(f'moe_expert_{expert_id}_usage', 
                            routing_weights[:, expert_id].mean(),
                            prog_bar=False)

        return loss

    def on_validation_epoch_end(self):
        """Compute and log per-phoneme metrics after full validation epoch"""
        
        # Compute per-phoneme F1 scores (now with accumulated data)
        phoneme_scores = []
        for i in range(self.hparams.vocab_size):
            f1_score = self.per_phoneme_f1[i].compute()
            phoneme_scores.append((i, f1_score.item() if torch.is_tensor(f1_score) else f1_score))
            
            # Log individual phoneme F1
            self.log(f'val_f1_phoneme_{i}', f1_score, prog_bar=False)
            
            # Reset for next epoch
            self.per_phoneme_f1[i].reset()
        
        # Sort and identify worst/best
        phoneme_scores.sort(key=lambda x: x[1])
        worst_20 = phoneme_scores[:20]
        best_20 = phoneme_scores[-20:]
        
        # Print summary
        worst_str = ', '.join([f'ph{i}:{s:.3f}' for i, s in worst_20])
        best_str = ', '.join([f'ph{i}:{s:.3f}' for i, s in best_20])
        
        print(f"\n{'='*50}")
        print(f"Worst performing phonemes: {worst_str}")
        print(f"Best performing phonemes: {best_str}")
        
        # Also show which phonemes are never predicted
        never_predicted = [i for i, s in phoneme_scores if s == 0.0]
        if never_predicted:
            print(f"Never predicted phonemes: {len(never_predicted)} phones, in detail: {never_predicted}...")  # Show first 10
        
        print(f"{'='*50}\n")
        
        # Reset main metric
        self.f1_macro.reset()
        
    def on_train_epoch_end(self):
        self.epoch_counter += 1
        
    def configure_optimizers(self):
        # Different learning rates for different components
        params = [
            {'params': self.meg_encoder.parameters(), 'lr': self.hparams.learning_rate},
            {'params': self.conformers.parameters(), 'lr': self.hparams.learning_rate},
            {'params': self.classifier.parameters(), 'lr': self.hparams.learning_rate},
        ]
        
        if self.use_alignment:
            params.append({'params': self.cost_learner.parameters(), 'lr': self.hparams.learning_rate})
        
        if self.use_moe:
            params.append({'params': self.moe.parameters(), 'lr': self.hparams.learning_rate * 0.5})
            
        if self.use_contrastive:
            params.append({'params': self.contrastive_encoder.parameters(), 'lr': self.hparams.learning_rate * 0.5})
        
        if self.use_zipf:
            params.append({'params': self.zipf_learner.parameters(), 'lr': self.hparams.learning_rate * 0.1})
            params.append({'params': self.meg_aggregator.parameters(), 'lr': self.hparams.learning_rate})
        
        optimizer = torch.optim.AdamW(params, weight_decay=0.01)
        
        # Use OneCycleLR if we have access to trainer
        if hasattr(self, 'trainer') and hasattr(self.trainer, 'estimated_stepping_batches'):
            from torch.optim.lr_scheduler import OneCycleLR
            scheduler = OneCycleLR(
                optimizer,
                max_lr=[group['lr'] for group in params],
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.05,
                anneal_strategy='cos'
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
            }
        else:
            # Fallback to cosine annealing
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=50,
                eta_min=1e-6
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler
            }


# ============================================
# Backward Compatibility Functions
# ============================================

def create_meg_lcs_ctc_model(dataset_info, use_zipf=True):
    """
    Create LCS-CTC model configured for LibriBrain dataset.
    Maintains backward compatibility with original function signature.
    """
    model = MEGLCSCTC(
        meg_channels=dataset_info.get('meg_channels', 306),
        time_points=dataset_info.get('time_points', 125),
        vocab_size=dataset_info.get('num_phonemes', 39),
        hidden_dim=256,
        num_conformers=6,  # Increased
        learning_rate=1e-4,
        use_alignment=True,
        zipf_weights=use_zipf,
        zipf_alpha=0.99,
        zipf_boost_factor=0.3,
        # New features enabled by default
        use_multiscale=True,
        use_moe=True,
        use_contrastive=True,
        use_enhanced_conformers=True
    )
    
    return model


def integrate_with_libribrain(train_dataset, val_dataset, use_zipf=True):
    """
    Example integration with LibriBrain competition code.
    Maintains backward compatibility.
    """
    from torch.utils.data import DataLoader
    import lightning as L
    
    dataset_info = {
        'meg_channels': 306,
        'time_points': 125,
        'num_phonemes': 39
    }
    
    model = create_meg_lcs_ctc_model(dataset_info, use_zipf=use_zipf)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
    
    callbacks = [
        ModelCheckpoint(
            monitor='val_f1_macro',
            mode='max',
            save_top_k=3,
            filename='enhanced-meg-lcs-ctc-{epoch:02d}-{val_f1_macro:.3f}'
        ),
        EarlyStopping(
            monitor='val_f1_macro',
            mode='max',
            patience=10
        )
    ]
   
    print("Building trainer...")

    trainer = L.Trainer(
        devices="auto",
        max_epochs=30,
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,
        callbacks=callbacks,
        enable_progress_bar=True
    )
    
    trainer.fit(model, train_loader, val_loader)
    
    return model
