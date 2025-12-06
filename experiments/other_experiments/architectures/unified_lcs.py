"""
Unified MEG-Phoneme Classification Model - Three-Stage Training Version
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np
from typing import Dict, Optional, Tuple, Literal
from torchmetrics import F1Score, Accuracy, ConfusionMatrix
from collections import defaultdict
import warnings
from pathlib import Path

# ============================================
# Components from meg_lcs_ctc.py
# ============================================

class ZipfWeightLearner(nn.Module):
    """Zipf distribution learner from main model"""
    def __init__(self, vocab_size: int, meg_dim: int, alpha: float = 0.99):
        super().__init__()
        self.vocab_size = vocab_size
        self.alpha = alpha
        
        self.register_buffer('phoneme_counts', torch.ones(vocab_size))
        self.register_buffer('total_count', torch.tensor(vocab_size, dtype=torch.float32))
        self.register_buffer('meg_prototypes', torch.zeros(vocab_size, meg_dim))
        self.register_buffer('prototype_counts', torch.ones(vocab_size))
        
        self.zipf_s = nn.Parameter(torch.tensor(1.0))
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        self.phoneme_meg_attention = nn.Sequential(
            nn.Linear(meg_dim * 2, meg_dim),
            nn.ReLU(),
            nn.Linear(meg_dim, 1),
            nn.Sigmoid()
        )
    
    def update_statistics(self, phonemes: torch.Tensor, meg_features: torch.Tensor):
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
        
        attention_weights = self.phoneme_meg_attention(combined_features.reshape(B * self.vocab_size, -1))
        attention_weights = attention_weights.squeeze(-1).reshape(B, self.vocab_size)
        
        weights = zipf_priors * (1 + meg_similarity) * attention_weights
        weights = F.softmax(weights, dim=-1)
        return weights

class MEGConformerLayer(nn.Module):
    """Conformer layer from main model"""
    def __init__(self, dim: int, num_heads: int = 4, ff_dim: int = None, 
                 kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        ff_dim = ff_dim or 2 * dim
        
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

class StrugglingPhonemeMultiTaskExpert(nn.Module):
    """Multi-task expert for struggling phonemes - the 65% performer"""
    def __init__(self, input_channels, hidden_dim=256, n_classes=14):
        super().__init__()
        
        # Shared encoder
        self.shared_encoder = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim*2),
            nn.ReLU()
        )
        
        # Task-specific heads
        self.phoneme_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim*2, n_classes)
        )
        
        self.manner_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim*2, 7)
        )
        
        self.place_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim*2, 8)
        )
        
        self.voicing_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim*2, 2)
        )
    
    def forward(self, x, return_all_heads=False):
        shared_features = self.shared_encoder(x)
        
        if return_all_heads:
            return {
                'phoneme': self.phoneme_head(shared_features),
                'manner': self.manner_head(shared_features),
                'place': self.place_head(shared_features),
                'voicing': self.voicing_head(shared_features)
            }
        else:
            return self.phoneme_head(shared_features)

# ============================================
# Three-Stage Unified Model
# ============================================

class UnifiedMEGPhonemeClassifier(L.LightningModule):
    """
    Three-stage training unified model:
    Stage 1: Main model only
    Stage 2: Specialist + Router (main frozen)
    Stage 3: Joint fine-tuning
    """
    
    def __init__(self,
             meg_channels=306,
             time_points=125,
             vocab_size=39,
             hidden_dim=256,
             num_conformers=4,
             learning_rate=1e-4,
             # Zipf parameters
             use_zipf=True,
             zipf_alpha=0.99,
             zipf_boost_factor=0.3,
             # Specialist parameters
             specialist_hidden_dim=512,
             routing_temperature=1.0,
             # Training stage control
             training_stage="stage1",
             stage1_epochs=20,
             stage2_epochs=15,
             stage3_epochs=10,
             # Stage-specific learning rates
             stage1_lr=1e-4,
             stage2_lr=5e-5,
             stage3_lr=1e-5,
             # Loss weights
             specialist_weight=0.3,
             main_model_weight=0.7,
             # Checkpoint paths for loading pretrained components
             pretrained_main_path: Optional[str] = None,
             pretrained_specialist_path: Optional[str] = None,
             # Paths for specialist's sensor analysis (required for full specialist)
             sensor_importance_path: Optional[str] = None,
             region_importance_path: Optional[str] = None,
             phoneme_summary_path: Optional[str] = None,
             phoneme_counts_path: Optional[str] = None):
        super().__init__()
        self.save_hyperparameters()
        
        # Training stage management
        self.current_stage = training_stage
        self.stage_epochs = {
            "stage1": stage1_epochs,
            "stage2": stage2_epochs,
            "stage3": stage3_epochs
        }
        self.stage_lrs = {
            "stage1": stage1_lr,
            "stage2": stage2_lr,
            "stage3": stage3_lr
        }
        
        # Define struggling phonemes
        self.struggling_indices = [0, 3, 11, 14, 18, 25, 31, 32, 36, 38, 2, 12, 34, 16]
        self.struggling_set = set(self.struggling_indices)
        self.struggling_to_reduced = {idx: i for i, idx in enumerate(self.struggling_indices)}
        self.reduced_to_struggling = {i: idx for idx, i in self.struggling_to_reduced.items()}
        
        print("\n" + "="*80)
        print(f"INITIALIZING THREE-STAGE UNIFIED MODEL")
        print(f"Current Stage: {self.current_stage.upper()}")
        print(f"Stage Configuration:")
        print(f"  Stage 1: {stage1_epochs} epochs @ LR={stage1_lr} (Main Model Only)")
        print(f"  Stage 2: {stage2_epochs} epochs @ LR={stage2_lr} (Specialist + Router)")
        print(f"  Stage 3: {stage3_epochs} epochs @ LR={stage3_lr} (Joint Fine-tuning)")
        print("="*80 + "\n")
        
        # ============================================
        # Main Model Components
        # ============================================
        
        self.meg_encoder = nn.Sequential(
            nn.Conv1d(meg_channels, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        self.conformers = nn.ModuleList([
            MEGConformerLayer(hidden_dim, 4, hidden_dim*2) 
            for _ in range(num_conformers)
        ])
        
        self.main_classifier = nn.Sequential(
            nn.Linear(hidden_dim * time_points, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, vocab_size)
        )
        
        # Zipf components
        self.use_zipf = use_zipf
        if use_zipf:
            self.zipf_learner = ZipfWeightLearner(vocab_size, hidden_dim, alpha=zipf_alpha)
            self.meg_aggregator = nn.Sequential(
                nn.Linear(hidden_dim * time_points, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
        
        # ============================================
        # FULL Specialist Model (65% performer)
        # ============================================

        # Check if we're loading a pretrained specialist or creating new
        if pretrained_specialist_path and Path(pretrained_specialist_path).exists():
            print(f"Loading pretrained specialist from: {pretrained_specialist_path}")
            
            # Check if this is a best_expert checkpoint or full model
            checkpoint = torch.load(pretrained_specialist_path, map_location='cpu')
            
            if 'expert_name' in checkpoint and checkpoint['expert_name'] == 'expert_multitask':
                # Loading just the best expert
                print(f"Loading best expert only: {checkpoint['expert_name']}")
                
                # Create just the MultiTaskExpert
                from models.architectures._13_phonemes_why import MultiTaskExpert
                
                # FIXED: Use hidden_dim directly, not divided by 2
                # The MultiTaskExpert was trained with hidden_dim=512
                self.specialist_model = MultiTaskExpert(
                    input_channels=meg_channels,
                    hidden_dim=checkpoint['hidden_dim'],  # Use 512, not 256!
                    n_classes=14
                )
                
                # Load the state dict
                self.specialist_model.load_state_dict(checkpoint['state_dict'])
                
                # Store the mappings
                self.struggling_indices = checkpoint['struggling_indices']
                self.struggling_to_reduced = checkpoint['struggling_to_reduced']
                
                # Create a simple wrapper to make it compatible with the forward calls
                class SpecialistWrapper(nn.Module):
                    def __init__(self, expert_model, struggling_indices, struggling_to_reduced):
                        super().__init__()
                        self.expert = expert_model
                        self.struggling_indices = struggling_indices
                        self.struggling_to_reduced = struggling_to_reduced
                        
                    def forward(self, x, return_all_experts=False):
                        outputs = self.expert(x)
                        # Return just phoneme predictions and dummy gate logits
                        return outputs['phoneme'], torch.zeros(x.size(0), 2, device=x.device)
                
                self.specialist_model = SpecialistWrapper(
                    self.specialist_model, 
                    self.struggling_indices,
                    self.struggling_to_reduced
                )
                
                print("✓ Loaded best expert model (31% F1 on struggling phonemes)")
                
                # Freeze specialist if in stage 2
                if self.current_stage == "stage2":
                    for param in self.specialist_model.parameters():
                        param.requires_grad = False
                    print("  Expert frozen for Stage 2 inference")
            else:
                # This is a full specialist checkpoint - load the whole model
                from models.architectures._13_phonemes_why import StrugglingPhonemeSpecialist
                
                self.specialist_model = StrugglingPhonemeSpecialist.load_from_checkpoint(
                    pretrained_specialist_path,
                    meg_channels=meg_channels,
                    time_points=time_points,
                    vocab_size=vocab_size,
                    hidden_dim=specialist_hidden_dim,
                    learning_rate=learning_rate,
                    sensor_importance_path=sensor_importance_path,
                    region_importance_path=region_importance_path,
                    phoneme_summary_path=phoneme_summary_path,
                    phoneme_counts_path=phoneme_counts_path,
                    test_all_strategies=False,
                    use_multitask=True,
                    use_heavy_augmentation=True,
                    strict=False
                )
                print("✓ Loaded full specialist model")
                
                if self.current_stage == "stage2":
                    for param in self.specialist_model.parameters():
                        param.requires_grad = False
                    print("  Specialist frozen for Stage 2 inference")
        else:
            # Create new specialist model (simplified version)
            print("Creating new specialist model...")
            from models.architectures._13_phonemes_why import MultiTaskExpert
            
            expert = MultiTaskExpert(
                input_channels=meg_channels,
                hidden_dim=specialist_hidden_dim // 2,
                n_classes=14
            )
            
            class SpecialistWrapper(nn.Module):
                def __init__(self, expert_model, struggling_indices, struggling_to_reduced):
                    super().__init__()
                    self.expert = expert_model
                    self.struggling_indices = struggling_indices
                    self.struggling_to_reduced = struggling_to_reduced
                    
                def forward(self, x, return_all_experts=False):
                    outputs = self.expert(x)
                    return outputs['phoneme'], torch.zeros(x.size(0), 2, device=x.device)
            
            self.specialist_model = SpecialistWrapper(
                expert,
                self.struggling_indices,
                self.struggling_to_reduced
            )
            
            print("✓ Created new multitask expert for struggling phonemes")
            
        # ============================================
        # Routing Components
        # ============================================
        
        self.router = nn.Sequential(
            nn.Conv1d(meg_channels, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3)  # main only, specialist only, blend
        )
        
        # Confidence networks for intelligent blending
        self.main_confidence = nn.Sequential(
            nn.Linear(hidden_dim * time_points, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Specialist confidence (adapted for full model)
        if self.specialist_model is not None:
            # Full specialist has different architecture
            specialist_feature_dim = specialist_hidden_dim * 2
        else:
            # Simple expert
            specialist_feature_dim = specialist_hidden_dim * 2
        
        # Load pretrained main model if provided
        if pretrained_main_path and self.current_stage != "stage1":
            self._load_pretrained_main(pretrained_main_path)
        
        # Apply stage-specific freezing
        self._apply_stage_freezing()
        
        # Loss functions
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics for comprehensive tracking
        self._setup_metrics()
        
        # Track training progress
        self.current_epoch_in_stage = 0
        
        print("\n" + "="*80)
        print("MODEL CONFIGURATION SUMMARY:")
        print(f"  Main Model: {'Loaded' if pretrained_main_path else 'New'}")
        print(f"  Specialist: {'Full Ensemble (65%)' if self.specialist_model else 'Simple Expert'}")
        print(f"  Stage: {self.current_stage}")
        print(f"  Device will be set by Lightning")
        print("="*80 + "\n")

    def _setup_metrics(self):
        """Setup comprehensive metrics for all stages"""
        # Main metrics
        self.train_f1 = F1Score(num_classes=39, average='macro', task='multiclass')
        self.val_f1 = F1Score(num_classes=39, average='macro', task='multiclass')
        self.train_acc = Accuracy(num_classes=39, task='multiclass')
        self.val_acc = Accuracy(num_classes=39, task='multiclass')
        
        # Specialist metrics (14 classes)
        self.specialist_train_f1 = F1Score(num_classes=14, average='macro', task='multiclass')
        self.specialist_val_f1 = F1Score(num_classes=14, average='macro', task='multiclass')
        
        # Per-phoneme tracking for struggling ones
        self.struggling_train_f1 = F1Score(num_classes=39, average='none', task='multiclass')
        self.struggling_val_f1 = F1Score(num_classes=39, average='none', task='multiclass')
        
        # Router metrics
        self.router_train_acc = Accuracy(num_classes=3, task='multiclass')
        self.router_val_acc = Accuracy(num_classes=3, task='multiclass')
    
    def _apply_stage_freezing(self):
        """Apply stage-specific parameter freezing"""
        if self.current_stage == "stage1":
            # Stage 1: Only train main model
            print("Stage 1: Training main model only")
            for param in self.specialist_model.parameters():
                param.requires_grad = False
            for param in self.router.parameters():
                param.requires_grad = False
            for param in self.main_confidence.parameters():
                param.requires_grad = False
            for param in self.specialist_confidence.parameters():
                param.requires_grad = False
                
        elif self.current_stage == "stage2":
            # Stage 2: Freeze main, train specialist + router
            print("Stage 2: Training specialist + router (main model frozen)")
            for param in self.meg_encoder.parameters():
                param.requires_grad = False
            for param in self.conformers.parameters():
                param.requires_grad = False
            for param in self.main_classifier.parameters():
                param.requires_grad = False
            if self.use_zipf:
                for param in self.zipf_learner.parameters():
                    param.requires_grad = False
                for param in self.meg_aggregator.parameters():
                    param.requires_grad = False
                    
        else:  # stage3
            # Stage 3: Train everything with lower LR
            print("Stage 3: Fine-tuning all components jointly")
            for param in self.parameters():
                param.requires_grad = True
    
    def _load_pretrained_main(self, checkpoint_path):
        """Load pretrained main model weights"""
        print(f"Loading pretrained main model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        # Debug: Check what's in the checkpoint
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"Found state_dict with {len(state_dict)} keys")
        else:
            state_dict = checkpoint
            print(f"Direct state dict with {len(checkpoint)} keys")
        
        # Debug: Print first 10 keys to see naming pattern
        print("First 10 keys in checkpoint:")
        for i, key in enumerate(list(state_dict.keys())[:10]):
            print(f"  {key}")
        
        # Check how many keys match our components
        main_components = ['meg_encoder', 'conformers', 'main_classifier', 'zipf_learner', 'meg_aggregator']
        matched_keys = [key for key in state_dict.keys() if any(comp in key for comp in main_components)]
        print(f"Found {len(matched_keys)} matching main model keys")
        
        # Keep only main model components
        keys_to_delete = []
        for key in state_dict.keys():
            if not any(comp in key for comp in main_components):
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            del state_dict[key]
        
        # Try to load
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"✓ Loaded {len(state_dict)} parameters")
        print(f"  Missing keys: {len(missing)}")
        print(f"  Unexpected keys: {len(unexpected)}")
        
        if len(missing) > 0:
            print(f"  First 5 missing: {missing[:5]}")
    def _load_pretrained_specialist(self, checkpoint_path):
        """Load pretrained specialist weights"""
        print(f"Loading pretrained specialist from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        
        # Keep only specialist components
        specialist_components = ['specialist_expert', 'router', 'main_confidence', 'specialist_confidence']
        keys_to_delete = []
        for key in state_dict.keys():
            if not any(comp in key for comp in specialist_components):
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            del state_dict[key]
        
        self.load_state_dict(state_dict, strict=False)
        print(f"✓ Specialist weights loaded: {len(state_dict)} parameters")
    
    def _forward_main_only(self, x):
        """Stage 1: Main model forward pass"""
        B, C, T = x.shape
        
        features = self.meg_encoder(x)
        features = features.transpose(1, 2)
        
        for conformer in self.conformers:
            features = conformer(features)
        
        features_flat = features.reshape(B, -1)
        logits = self.main_classifier(features_flat)
        
        if self.use_zipf and not self.training:
            meg_agg = self.meg_aggregator(features_flat)
            zipf_adjustments = self.zipf_learner(meg_agg, training=False)
            probs = F.softmax(logits, dim=-1)
            adjusted_probs = (1 - self.hparams.zipf_boost_factor) * probs + \
                           self.hparams.zipf_boost_factor * zipf_adjustments
            logits = torch.log(adjusted_probs + 1e-10)
        
        return logits
    
    def _forward_with_specialist_training(self, x):
        """Stage 2: Gradually introduce specialist"""
        B, C, T = x.shape
        
        # Get both predictions
        with torch.no_grad():
            main_logits = self._forward_main_only(x)
        
        # Get specialist predictions (now just returns phoneme logits directly)
        specialist_logits_reduced, _ = self.specialist_model(x, return_all_experts=False)
        
        # Convert to full space
        full_specialist_logits = torch.full((B, 39), -100.0, device=x.device)
        for reduced_idx in range(14):
            full_idx = self.reduced_to_struggling[reduced_idx]
            full_specialist_logits[:, full_idx] = specialist_logits_reduced[:, reduced_idx]
        
        # GRADUAL BLENDING based on epoch
        blend_factor = min(self.current_epoch / 10.0, 1.0)
        
        final_logits = main_logits.clone()
        for b in range(B):
            for idx in self.struggling_indices:
                if full_specialist_logits[b, idx] > -99:
                    final_logits[b, idx] = (1 - blend_factor) * main_logits[b, idx] + \
                                        blend_factor * full_specialist_logits[b, idx]
        
        return final_logits

    def _forward_unified(self, x):
        """Stage 3: Full unified forward with learned routing"""
        B, C, T = x.shape
        
        # Get main model features and predictions
        features = self.meg_encoder(x)
        features = features.transpose(1, 2)
        for conformer in self.conformers:
            features = conformer(features)
        features_flat = features.reshape(B, -1)
        main_logits = self.main_classifier(features_flat)
        
        # Apply Zipf if not training
        if self.use_zipf and not self.training:
            meg_agg = self.meg_aggregator(features_flat)
            zipf_adjustments = self.zipf_learner(meg_agg, training=False)
            probs = F.softmax(main_logits, dim=-1)
            adjusted_probs = (1 - self.hparams.zipf_boost_factor) * probs + \
                        self.hparams.zipf_boost_factor * zipf_adjustments
            main_logits = torch.log(adjusted_probs + 1e-10)
        
        # Get specialist predictions
        specialist_logits_reduced, _ = self.specialist_model(x, return_all_experts=False)
        full_specialist_logits = torch.full((B, 39), -100.0, device=x.device)
        for reduced_idx in range(14):
            full_idx = self.reduced_to_struggling[reduced_idx]
            full_specialist_logits[:, full_idx] = specialist_logits_reduced[:, reduced_idx]
        
        # Get routing and confidence
        routing_logits = self.router(x)
        routing_probs = F.softmax(routing_logits / self.hparams.routing_temperature, dim=-1)
        
        main_conf = self.main_confidence(features_flat).squeeze(-1)
        specialist_conf = torch.sigmoid(specialist_logits_reduced.max(dim=1)[0] / 10.0)
        
        # ===== ADD CONSERVATIVE CONSTRAINTS HERE =====
        # Conservative blending parameters
        MAX_SPECIALIST_WEIGHT = 0.4  # Never more than 40% specialist
        MIN_SPECIALIST_CONF = 0.6    # Only use specialist if confident
        
        # Intelligent blending
        final_logits = torch.zeros_like(main_logits)
        for b in range(B):
            main_pred = main_logits[b].argmax().item()
            is_struggling = main_pred in self.struggling_set
            
            if is_struggling and specialist_conf[b] > MIN_SPECIALIST_CONF:  # ← ADDED THRESHOLD
                # Blend for struggling phonemes (but conservatively)
                for idx in self.struggling_indices:
                    if full_specialist_logits[b, idx] > -99:
                        # Calculate weights from router
                        w_spec = routing_probs[b, 1] + routing_probs[b, 2] * specialist_conf[b]
                        w_main = routing_probs[b, 0] + routing_probs[b, 2] * main_conf[b]
                        total = w_spec + w_main + 1e-8
                        
                        # ===== APPLY CONSERVATIVE CAP =====
                        spec_weight = min(w_spec/total, MAX_SPECIALIST_WEIGHT)
                        main_weight = 1.0 - spec_weight
                        
                        final_logits[b, idx] = spec_weight * full_specialist_logits[b, idx] + \
                                            main_weight * main_logits[b, idx]
                    else:
                        final_logits[b, idx] = main_logits[b, idx]
                
                # Non-struggling use main
                for idx in range(39):
                    if idx not in self.struggling_set:
                        final_logits[b, idx] = main_logits[b, idx]
            else:
                # Not struggling OR specialist not confident → use main only
                final_logits[b] = main_logits[b]
        
        return final_logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # Print stage info every 100 batches
        if batch_idx % 100 == 0:
            print(f"\n[Stage {self.current_stage[-1]}, Epoch {self.current_epoch+1}/{self.stage_epochs[self.current_stage]}] "
                  f"Batch {batch_idx}")
        
        if self.current_stage == "stage1":
            return self._training_step_stage1(x, y, batch_idx)
        elif self.current_stage == "stage2":
            return self._training_step_stage2(x, y, batch_idx)
        else:
            return self._training_step_stage3(x, y, batch_idx)
    
    def _training_step_stage1(self, x, y, batch_idx):
        """Stage 1: Train main model only"""
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        
        # Update Zipf statistics
        if self.use_zipf:
            with torch.no_grad():
                features = self.meg_encoder(x).transpose(1, 2)
                for conformer in self.conformers:
                    features = conformer(features)
                features_flat = features.reshape(x.size(0), -1)
                meg_agg = self.meg_aggregator(features_flat)
                self.zipf_learner.update_statistics(y, meg_agg)
        
        # Metrics
        self.train_f1.update(logits, y)
        self.train_acc.update(logits, y)
        
        # Calculate struggling phoneme performance
        with torch.no_grad():
            struggling_f1_scores = self.struggling_train_f1(logits, y)
            struggling_avg = struggling_f1_scores[self.struggling_indices].mean()
        
        # Logging
        self.log('stage1/train_loss', loss, prog_bar=True)
        self.log('stage1/train_f1', self.train_f1, prog_bar=True)
        self.log('stage1/train_acc', self.train_acc)
        self.log('stage1/train_struggling_f1', struggling_avg)
        
        return loss
    
    def _training_step_stage2(self, x, y, batch_idx):
        """Stage 2: Train specialist + router with frozen main"""
        logits = self.forward(x)
        
        # Find struggling phoneme samples
        struggling_mask = torch.tensor([y_i.item() in self.struggling_set for y_i in y], 
                                    device=self.device)
        
        total_loss = 0
        
        # Main loss (for stability)
        main_loss = self.criterion(logits, y)
        total_loss += 0.1 * main_loss
        
        # Specialist loss on struggling samples
        if struggling_mask.sum() > 0:
            x_struggling = x[struggling_mask]
            y_struggling = y[struggling_mask]
            y_reduced = torch.tensor([self.struggling_to_reduced[y_i.item()] 
                                    for y_i in y_struggling],
                                    device=self.device, dtype=torch.long)
            
            # Use full specialist model
            specialist_logits, _ = self.specialist_model(x_struggling, return_all_experts=False)
            specialist_loss = self.criterion(specialist_logits, y_reduced)
            total_loss += specialist_loss
            
            # Update specialist metrics
            self.specialist_train_f1.update(specialist_logits, y_reduced)
            self.log('stage2/specialist_train_f1', self.specialist_train_f1, prog_bar=True)
        
        # Router loss - predict if sample is struggling
        router_logits = self.router(x)
        router_targets = torch.zeros(len(y), dtype=torch.long, device=self.device)
        router_targets[struggling_mask] = 1
        router_loss = self.criterion(router_logits, router_targets)
        total_loss += 0.5 * router_loss
        
        # Metrics
        self.train_f1.update(logits, y)
        self.router_train_acc.update(router_logits, router_targets)
        
        # Logging
        self.log('stage2/train_loss', total_loss, prog_bar=True)
        self.log('stage2/train_f1', self.train_f1, prog_bar=True)
        self.log('stage2/router_acc', self.router_train_acc)
        self.log('stage2/router_loss', router_loss)
        
        return total_loss
        
    def _training_step_stage3(self, x, y, batch_idx):
        """Stage 3: Joint fine-tuning"""
        logits = self.forward(x)
        main_loss = self.criterion(logits, y)
        
        # Find struggling samples
        struggling_mask = torch.tensor([y_i.item() in self.struggling_set for y_i in y],
                                    device=self.device)
        
        # Dynamic loss weighting
        struggling_ratio = struggling_mask.float().mean()
        specialist_weight = self.hparams.specialist_weight * (1 + struggling_ratio)
        main_weight = self.hparams.main_model_weight * (1 - struggling_ratio * 0.5)
        
        total_loss = main_weight * main_loss
        
        # Add specialist loss if we have struggling samples
        if struggling_mask.sum() > 0:
            x_struggling = x[struggling_mask]
            y_struggling = y[struggling_mask]
            y_reduced = torch.tensor([self.struggling_to_reduced[y_i.item()]
                                    for y_i in y_struggling],
                                    device=self.device, dtype=torch.long)
            
            # Use full specialist model
            specialist_logits, _ = self.specialist_model(x_struggling, return_all_experts=False)
            specialist_loss = self.criterion(specialist_logits, y_reduced)
            total_loss += specialist_weight * specialist_loss
            
            self.specialist_train_f1.update(specialist_logits, y_reduced)
            self.log('stage3/specialist_train_f1', self.specialist_train_f1, prog_bar=True)
        
        # Metrics
        self.train_f1.update(logits, y)
        self.train_acc.update(logits, y)
        
        # Per-phoneme tracking
        with torch.no_grad():
            struggling_f1_scores = self.struggling_train_f1(logits, y)
            struggling_avg = struggling_f1_scores[self.struggling_indices].mean()
        
        # Comprehensive logging
        self.log('stage3/train_loss', total_loss, prog_bar=True)
        self.log('stage3/train_f1', self.train_f1, prog_bar=True)
        self.log('stage3/train_acc', self.train_acc)
        self.log('stage3/train_struggling_f1', struggling_avg)
        self.log('stage3/specialist_weight', specialist_weight)
        self.log('stage3/main_weight', main_weight)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        
        # Update main metrics
        self.val_f1.update(logits, y)
        self.val_acc.update(logits, y)
        
        # Stage-specific validation
        stage_prefix = f"{self.current_stage}/"
        
        # Calculate struggling phoneme performance
        struggling_f1_scores = self.struggling_val_f1(logits, y)
        struggling_avg = struggling_f1_scores[self.struggling_indices].mean()
        
        # Specialist evaluation if in stage 2 or 3
        # In the specialist evaluation section:
        if self.current_stage in ["stage2", "stage3"]:
            struggling_mask = torch.tensor([y_i.item() in self.struggling_set for y_i in y],
                                        device=self.device)
            
            if struggling_mask.sum() > 0:
                x_struggling = x[struggling_mask]
                y_struggling = y[struggling_mask]
                y_reduced = torch.tensor([self.struggling_to_reduced[y_i.item()]
                                        for y_i in y_struggling],
                                        device=self.device, dtype=torch.long)
                
                # Use full specialist model
                specialist_logits, _ = self.specialist_model(x_struggling, return_all_experts=False)
                self.specialist_val_f1.update(specialist_logits, y_reduced)
                self.log(f'{stage_prefix}val_specialist_f1', self.specialist_val_f1, prog_bar=True)
        
        # Logging
        self.log(f'{stage_prefix}val_loss', loss)
        self.log(f'{stage_prefix}val_f1', self.val_f1, prog_bar=True)
        self.log(f'{stage_prefix}val_acc', self.val_acc)
        self.log(f'{stage_prefix}val_struggling_f1', struggling_avg)
        
        # Track best metric for checkpointing
        self.log('val_f1_macro', self.val_f1, prog_bar=True)
        
        return loss
    
    def on_train_epoch_end(self):
        """Reset metrics and print stage progress"""
        self.current_epoch_in_stage += 1
        
        print(f"\n{'='*80}")
        print(f"STAGE {self.current_stage[-1]} - Epoch {self.current_epoch_in_stage}/{self.stage_epochs[self.current_stage]} Complete")
        print(f"Main Model F1: {self.train_f1.compute():.4f}")
        if self.current_stage in ["stage2", "stage3"]:
            print(f"Specialist F1: {self.specialist_train_f1.compute():.4f}")
        print(f"{'='*80}\n")
        
        # Reset metrics
        self.train_f1.reset()
        self.train_acc.reset()
        self.specialist_train_f1.reset()
        self.struggling_train_f1.reset()
        self.router_train_acc.reset()
    
    def on_validation_epoch_end(self):
        """Print comprehensive validation results"""
        val_f1 = self.val_f1.compute()
        val_acc = self.val_acc.compute()
        
        print(f"\n{'='*80}")
        print(f"VALIDATION RESULTS - Stage {self.current_stage[-1]}")
        print(f"{'='*80}")
        print(f"Overall F1: {val_f1:.4f}")
        print(f"Overall Accuracy: {val_acc:.4f}")
        
        if self.current_stage in ["stage2", "stage3"]:
            specialist_f1 = self.specialist_val_f1.compute()
            print(f"Specialist F1 (struggling phonemes): {specialist_f1:.4f}")
        
        # Show per struggling phoneme
        struggling_scores = self.struggling_val_f1.compute()
        print("\nStruggling Phoneme F1 Scores:")
        phoneme_names = ['aa', 'ao', 'er', 'g', 'jh', 'oy', 'th', 'uh', 'y', 'zh', 'ah', 'ey', 'v', 'ih']
        for i, idx in enumerate(self.struggling_indices):
            score = struggling_scores[idx].item()
            print(f"  {phoneme_names[i]:4s}: {score:.4f}")
        
        print(f"{'='*80}\n")
        
        # Reset metrics
        self.val_f1.reset()
        self.val_acc.reset()
        self.specialist_val_f1.reset()
        self.struggling_val_f1.reset()
        self.router_val_acc.reset()
    
    def configure_optimizers(self):
        """Stage-specific optimizer configuration"""
        current_lr = self.stage_lrs[self.current_stage]
        
        # Collect parameters that require gradients
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        
        print(f"Configuring optimizer for {self.current_stage} with LR={current_lr}")
        print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        if self.current_stage == "stage3":
            # Use different LRs for different components
            optimizer = torch.optim.AdamW([
                {'params': self.meg_encoder.parameters(), 'lr': 1e-5},  # Main: very low
                {'params': self.conformers.parameters(), 'lr': 1e-5},   # Main: very low
                {'params': self.main_classifier.parameters(), 'lr': 1e-5},  # Main: very low
                {'params': self.specialist_model.parameters(), 'lr': 5e-5},  # Specialist: higher
                {'params': self.router.parameters(), 'lr': 5e-5},  # Router: higher
            ], weight_decay=0.01)
        else:
            optimizer = torch.optim.AdamW(trainable_params, lr=current_lr, weight_decay=0.01)
        
        # Stage-specific scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.stage_epochs[self.current_stage],
            eta_min=current_lr * 0.1
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }
    
    def set_inference_mode(self):
        """Set model to inference mode with proper configuration"""
        self.eval()
        
        # Ensure all submodules are in eval mode
        self.meg_encoder.eval()
        for conformer in self.conformers:
            conformer.eval()
        self.main_classifier.eval()
        
        if self.specialist_model is not None:
            self.specialist_model.eval()
            if hasattr(self.specialist_model, 'expert'):
                self.specialist_model.expert.eval()
        
        self.router.eval()
        self.main_confidence.eval()
        
        if self.use_zipf:
            self.zipf_learner.eval()
            self.meg_aggregator.eval()
        
        # Force to stage3 for best performance
        self.current_stage = "stage3"
        
        print("Model set to inference mode (Stage 3)")
        print(f"  Main model: eval mode")
        print(f"  Specialist: eval mode")
        print(f"  Router: eval mode")
        
        return self
    
    def forward_inference(self, x, debug=False):
        """Simplified forward pass for inference with debugging"""
        B, C, T = x.shape
        
        if debug:
            print(f"\nInference forward pass:")
            print(f"  Input shape: {x.shape}")
        
        # Get main model predictions
        features = self.meg_encoder(x)
        features = features.transpose(1, 2)
        
        for conformer in self.conformers:
            features = conformer(features)
        
        features_flat = features.reshape(B, -1)
        main_logits = self.main_classifier(features_flat)
        
        # Apply Zipf adjustments
        if self.use_zipf:
            meg_agg = self.meg_aggregator(features_flat)
            zipf_adjustments = self.zipf_learner(meg_agg, training=False)
            probs = F.softmax(main_logits, dim=-1)
            adjusted_probs = (1 - self.hparams.zipf_boost_factor) * probs + \
                        self.hparams.zipf_boost_factor * zipf_adjustments
            main_logits = torch.log(adjusted_probs + 1e-10)
        
        # Get specialist predictions
        specialist_logits_reduced, _ = self.specialist_model(x, return_all_experts=False)
        
        # Convert to full space
        full_specialist_logits = torch.full((B, 39), -float('inf'), device=x.device)
        for reduced_idx in range(14):
            full_idx = self.reduced_to_struggling[reduced_idx]
            full_specialist_logits[:, full_idx] = specialist_logits_reduced[:, reduced_idx]
        
        # OVERRIDE: Force specialist usage for struggling phonemes
        # Since router isn't working well, use a simpler heuristic
        
        final_logits = main_logits.clone()
        specialist_used_count = 0
        
        for b in range(B):
            # Get main model's prediction
            main_pred = main_logits[b].argmax().item()
            
            # If main predicts a struggling phoneme, blend with specialist
            if main_pred in self.struggling_set:
                for idx in self.struggling_indices:
                    if full_specialist_logits[b, idx] > -float('inf'):
                        # Use fixed blending: 60% main, 40% specialist
                        # (since specialist had 65% F1 and main had 62%)
                        final_logits[b, idx] = 0.6 * main_logits[b, idx] + 0.4 * full_specialist_logits[b, idx]
                        specialist_used_count += 1
            
            # Alternative: Always blend for struggling phonemes regardless of main prediction
            # Uncomment this if the above doesn't work well
            """
            for idx in self.struggling_indices:
                if full_specialist_logits[b, idx] > -float('inf'):
                    # Get specialist confidence for this phoneme
                    reduced_idx = self.struggling_to_reduced[idx]
                    spec_conf = torch.sigmoid(specialist_logits_reduced[b, reduced_idx] / 5.0)
                    
                    if spec_conf > 0.3:  # Low threshold
                        blend_weight = min(spec_conf * 0.5, 0.4)  # Max 40% specialist
                        final_logits[b, idx] = (1 - blend_weight) * main_logits[b, idx] + \
                                            blend_weight * full_specialist_logits[b, idx]
                        specialist_used_count += 1
            """
        
        if debug:
            print(f"  Main logits range: [{main_logits.min():.2f}, {main_logits.max():.2f}]")
            print(f"  Specialist confidence: {torch.sigmoid(specialist_logits_reduced.max(dim=1)[0] / 10.0).mean():.2%}")
            print(f"  Specialist used for {specialist_used_count}/{B*14} struggling predictions")
            print(f"  Final logits range: [{final_logits.min():.2f}, {final_logits.max():.2f}]")
        
        return final_logits

    def forward(self, x):
        """Forward pass adapts based on training stage"""
        # Use simplified inference path if not training
        if not self.training and hasattr(self, 'use_inference_forward'):
            return self.forward_inference(x, debug=False)
        
        # Original forward logic
        B, C, T = x.shape
        
        if self.current_stage == "stage1":
            return self._forward_main_only(x)
        elif self.current_stage == "stage2":
            return self._forward_with_specialist_training(x)
        else:  # stage3
            return self._forward_unified(x)