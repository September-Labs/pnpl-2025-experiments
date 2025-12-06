"""
MEG Phoneme Classifier with Hierarchical Consistency Enhancement (HCE)
Adapted for Lightning training pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import F1Score
from collections import defaultdict
import math
from typing import Dict, List, Optional

# ============================================================================
# PHONEME LINGUISTIC HIERARCHY
# ============================================================================

# class PhonemeHierarchy:
#     """Defines the linguistic hierarchy for phonemes"""
    
#     def __init__(self, vocab_size: int = 39):
#         self.vocab_size = vocab_size
        
#         # Define the phoneme mappings (adjust based on your actual phoneme set)
#         # Level 1: Broad categories
#         self.broad_categories = {
#             'vowel': set(range(0, 15)),  # Phoneme IDs 0-14 are vowels
#             'consonant': set(range(15, 38)),  # Phoneme IDs 15-37 are consonants
#             'special': {38}  # Phoneme ID 38 is silence/special
#         }
        
#         # Level 2: Manner of articulation (for consonants)
#         self.manner_categories = {
#             'stop': {15, 16, 17, 18, 19, 20},  # p, b, t, d, k, g
#             'fricative': {21, 22, 23, 24, 25, 26, 27, 28},  # f, v, θ, ð, s, z, ʃ, ʒ
#             'affricate': {29, 30},  # tʃ, dʒ
#             'nasal': {31, 32, 33},  # m, n, ŋ
#             'liquid': {34, 35},  # l, r
#             'glide': {36, 37}  # w, j
#         }
        
#         # Level 2: Vowel categories
#         self.vowel_height = {
#             'high': {0, 1, 2, 3},  # i, ɪ, u, ʊ
#             'mid': {4, 5, 6, 7, 8},  # e, ɛ, ə, o, ɔ
#             'low': {9, 10, 11, 12, 13, 14}  # æ, a, ɑ, etc.
#         }
        
#         self.vowel_backness = {
#             'front': {0, 1, 4, 5, 9},  # i, ɪ, e, ɛ, æ
#             'central': {6, 10, 11},  # ə, ʌ, a
#             'back': {2, 3, 7, 8, 12, 13, 14}  # u, ʊ, o, ɔ, ɑ
#         }
        
#         # Level 3: Place of articulation (for consonants)
#         self.place_categories = {
#             'bilabial': {15, 16, 31, 36},  # p, b, m, w
#             'labiodental': {21, 22},  # f, v
#             'dental': {23, 24},  # θ, ð
#             'alveolar': {17, 18, 32, 25, 26, 34, 35},  # t, d, n, s, z, l, r
#             'postalveolar': {27, 28, 29, 30},  # ʃ, ʒ, tʃ, dʒ
#             'palatal': {37},  # j
#             'velar': {19, 20, 33},  # k, g, ŋ
#             'glottal': {28}  # h
#         }
        
#         self._build_mappings()
# Apparenlty above was completely incorrect, below is the correct one
class PhonemeHierarchy:
    """Defines the linguistic hierarchy for phonemes"""
    
    def __init__(self, vocab_size: int = 39):
        self.vocab_size = vocab_size
        
        # Level 1: Broad categories
        self.broad_categories = {
            'vowel': {0, 1, 2, 3, 4, 5, 10, 11, 12, 16, 17, 24, 25, 32, 33},  # All vowels
            'consonant': {6, 7, 8, 9, 13, 14, 15, 18, 19, 20, 21, 22, 23, 26, 27, 28, 29, 30, 31, 34, 35, 36, 37, 38},
            'special': set()  # No special tokens in your dataset
        }
        
        # Level 2: Manner of articulation (for consonants)
        self.manner_categories = {
            'stop': {6, 8, 14, 19, 26, 30},  # B, D, G, K, P, T
            'fricative': {9, 13, 15, 28, 29, 31, 34, 37, 38},  # DH, F, HH, S, SH, TH, V, Z, ZH
            'affricate': {7, 18},  # CH, JH
            'nasal': {21, 22, 23},  # M, N, NG
            'liquid': {20, 27},  # L, R
            'glide': {35, 36}  # W, Y
        }
        
        # Level 2: Vowel categories
        self.vowel_height = {
            'high': {16, 17, 32, 33},  # IH, IY, UH, UW
            'mid': {10, 11, 12, 24, 25},  # EH, ER, EY, OW, OY
            'low': {0, 1, 2, 3, 4, 5}  # AA, AE, AH, AO, AW, AY
        }
        
        self.vowel_backness = {
            'front': {1, 10, 12, 16, 17},  # AE, EH, EY, IH, IY
            'central': {2, 11},  # AH, ER
            'back': {0, 3, 4, 5, 24, 25, 32, 33}  # AA, AO, AW, AY, OW, OY, UH, UW
        }
        
        # Level 3: Place of articulation (for consonants)
        self.place_categories = {
            'bilabial': {6, 21, 26, 35},  # B, M, P, W
            'labiodental': {13, 34},  # F, V
            'dental': {9, 31},  # DH, TH
            'alveolar': {8, 20, 22, 27, 28, 30, 37},  # D, L, N, R, S, T, Z
            'postalveolar': {7, 18, 29, 38},  # CH, JH, SH, ZH
            'palatal': {36},  # Y
            'velar': {14, 19, 23},  # G, K, NG
            'glottal': {15}  # HH
        }
        
        # Additional features for vowels (optional but helpful)
        self.vowel_tenseness = {
            'tense': {12, 17, 24, 33},  # EY, IY, OW, UW
            'lax': {0, 1, 2, 3, 10, 16, 32},  # AA, AE, AH, AO, EH, IH, UH
            'diphthong': {4, 5, 25}  # AW, AY, OY
        }
        
        self._build_mappings()
        
    def _build_mappings(self):
        """Build mappings from phoneme ID to categories"""
        self.phoneme_to_broad = {}
        self.phoneme_to_manner = {}
        self.phoneme_to_place = {}
        self.phoneme_to_height = {}
        self.phoneme_to_backness = {}
        
        for category, phonemes in self.broad_categories.items():
            for p in phonemes:
                self.phoneme_to_broad[p] = category
        
        for manner, phonemes in self.manner_categories.items():
            for p in phonemes:
                self.phoneme_to_manner[p] = manner
        
        for place, phonemes in self.place_categories.items():
            for p in phonemes:
                self.phoneme_to_place[p] = place
        
        for height, phonemes in self.vowel_height.items():
            for p in phonemes:
                self.phoneme_to_height[p] = height
        
        for backness, phonemes in self.vowel_backness.items():
            for p in phonemes:
                self.phoneme_to_backness[p] = backness
    
    def get_hierarchical_labels(self, phoneme_id: int) -> Dict[str, int]:
        """Get all hierarchical labels for a phoneme"""
        labels = {}
        
        broad = self.phoneme_to_broad.get(phoneme_id, 'special')
        labels['broad'] = ['vowel', 'consonant', 'special'].index(broad)
        
        if broad == 'consonant':
            manner = self.phoneme_to_manner.get(phoneme_id, 'stop')
            labels['manner'] = ['stop', 'fricative', 'affricate', 'nasal', 'liquid', 'glide'].index(manner)
            
            place = self.phoneme_to_place.get(phoneme_id, 'alveolar')
            labels['place'] = ['bilabial', 'labiodental', 'dental', 'alveolar', 
                              'postalveolar', 'palatal', 'velar', 'glottal'].index(place)
        
        if broad == 'vowel':
            height = self.phoneme_to_height.get(phoneme_id, 'mid')
            labels['height'] = ['high', 'mid', 'low'].index(height)
            
            backness = self.phoneme_to_backness.get(phoneme_id, 'central')
            labels['backness'] = ['front', 'central', 'back'].index(backness)
        
        return labels


# ============================================================================
# MODEL COMPONENTS
# ============================================================================

class SensorFusionBlock(nn.Module):
    """Early fusion of MEG sensors using 2D convolutions"""
    
    def __init__(self, in_channels=1, out_channels=32, sensor_groups=4, dropout=0.1):
        super().__init__()
        
        self.spatial_temporal_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 
                     kernel_size=(7, 5),  # (sensors, time)
                     stride=(2, 1),
                     padding=(3, 2)),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout)
        )
        
        self.sensor_grouping = nn.Conv2d(
            out_channels, out_channels * sensor_groups,
            kernel_size=(3, 3),
            stride=(2, 1),
            padding=(1, 1),
            groups=out_channels
        )
        
        self.fusion = nn.Sequential(
            nn.BatchNorm2d(out_channels * sensor_groups),
            nn.GELU(),
            nn.Conv2d(out_channels * sensor_groups, out_channels * 2,
                     kernel_size=(3, 3),
                     padding=(1, 1)),
            nn.BatchNorm2d(out_channels * 2),
            nn.GELU(),
            nn.Dropout2d(dropout)
        )
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        x = self.spatial_temporal_conv(x)
        x = self.sensor_grouping(x)
        x = self.fusion(x)
        
        return x


class MultiScaleConvBlock(nn.Module):
    """Multi-scale convolution block processing different receptive fields"""
    
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.1):
        super().__init__()
        
        branch_channels = out_channels // 4
        
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(branch_channels),
            nn.GELU()
        )
        
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3, 
                     stride=stride, padding=1),
            nn.BatchNorm2d(branch_channels),
            nn.GELU()
        )
        
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3,
                     stride=1, padding=1),
            nn.BatchNorm2d(branch_channels),
            nn.GELU(),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3,
                     stride=stride, padding=1),
            nn.BatchNorm2d(branch_channels),
            nn.GELU()
        )
        
        self.branch_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
            nn.GELU()
        )
        
        self.combine = nn.Sequential(
            nn.Conv2d(branch_channels * 4, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout)
        )
        
        self.residual = nn.Identity() if (in_channels == out_channels and stride == 1) else \
                       nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
    
    def forward(self, x):
        branch1 = self.branch1x1(x)
        branch2 = self.branch3x3(x)
        branch3 = self.branch5x5(x)
        branch4 = self.branch_pool(x)
        
        branch4 = F.interpolate(branch4, size=(branch1.shape[2], branch1.shape[3]), mode='nearest')
        
        multi_scale = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        out = self.combine(multi_scale)
        return out + self.residual(x)


class RotaryPositionEmbedding(nn.Module):
    """RoPE for temporal encoding"""
    
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
    
    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


class TransformerBlock(nn.Module):
    """Transformer encoder block with RoPE"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, use_rope=True):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.use_rope = use_rope
        if use_rope:
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
    
    def apply_rope(self, q, k):
        batch_size, seq_len, n_heads, head_dim = q.shape
        
        cos, sin = self.rope(q, seq_dim=1)
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
        
        q_rot = q[..., :head_dim//2]
        q_pass = q[..., head_dim//2:]
        
        q_rot_new = torch.cat([
            q_rot * cos[..., :head_dim//2] - q_pass * sin[..., :head_dim//2],
            q_rot * sin[..., :head_dim//2] + q_pass * cos[..., :head_dim//2]
        ], dim=-1)
        
        k_rot = k[..., :head_dim//2]
        k_pass = k[..., head_dim//2:]
        
        k_rot_new = torch.cat([
            k_rot * cos[..., :head_dim//2] - k_pass * sin[..., :head_dim//2],
            k_rot * sin[..., :head_dim//2] + k_pass * cos[..., :head_dim//2]
        ], dim=-1)
        
        return q_rot_new, k_rot_new
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        residual = x
        x = self.norm1(x)
        
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        if self.use_rope:
            q, k = self.apply_rope(q, k)
        
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
        
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + self.dropout(x)
        
        return x


# ============================================================================
# MAIN MODEL (Lightning Module)
# ============================================================================

class MEGPhonemeClassifierHCE(L.LightningModule):
    """MEG phoneme classifier with HCE and early sensor fusion"""
    
    def __init__(
        self,
        time_points=125,
        meg_channels=306,
        vocab_size=39,
        initial_channels=32,
        conv_channels=[64, 128, 256],
        d_model=256,
        n_heads=8,
        n_transformer_layers=4,
        d_ff=1024,
        dropout=0.1,
        hce_phoneme_weight=1.0,
        hce_broad_weight=0.3,
        hce_manner_weight=0.2,
        hce_place_weight=0.2,
        hce_height_weight=0.15,
        hce_backness_weight=0.15,
        consistency_weight=0.1,
        learning_rate=1e-3,
        weight_decay=1e-4,
        label_smoothing=0.0,
        warmup_epochs=5,
        total_epochs=100,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.vocab_size = vocab_size
        self.hierarchy = PhonemeHierarchy(vocab_size)
        
        # HCE weights
        self.hce_weights = {
            'phoneme': hce_phoneme_weight,
            'broad': hce_broad_weight,
            'manner': hce_manner_weight,
            'place': hce_place_weight,
            'height': hce_height_weight,
            'backness': hce_backness_weight
        }
        self.consistency_weight = consistency_weight
        
        # Early sensor fusion
        self.sensor_fusion = SensorFusionBlock(
            in_channels=1,
            out_channels=initial_channels,
            sensor_groups=4,
            dropout=dropout
        )
        
        # Convolutional layers
        self.conv_blocks = nn.ModuleList()
        in_ch = initial_channels * 2
        for i, out_ch in enumerate(conv_channels):
            stride = 2 if i % 2 == 0 else 1
            self.conv_blocks.append(
                MultiScaleConvBlock(in_ch, out_ch, stride, dropout)
            )
            in_ch = out_ch
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(conv_channels[-1], conv_channels[-1], kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels[-1]),
            nn.GELU(),
            nn.Dropout2d(dropout)
        )
        
        # Reshape for transformer
        self.to_sequence = nn.AdaptiveAvgPool2d((None, 1))
        self.proj_to_transformer = nn.Linear(conv_channels[-1], d_model)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, use_rope=True)
            for _ in range(n_transformer_layers)
        ])
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Feature projection
        self.feature_norm = nn.LayerNorm(d_model)
        self.feature_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Classification heads
        self.heads = nn.ModuleDict({
            'phoneme': nn.Linear(d_model, vocab_size),
            'broad': nn.Linear(d_model, 3),
            'manner': nn.Linear(d_model, 6),
            'place': nn.Linear(d_model, 8),
            'height': nn.Linear(d_model, 3),
            'backness': nn.Linear(d_model, 3)
        })
        
        # Initialize heads
        for head in self.heads.values():
            nn.init.xavier_uniform_(head.weight, gain=0.5)
            if head.bias is not None:
                nn.init.constant_(head.bias, 0)
        
        # Metrics
        self.train_f1 = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
        self.val_f1 = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
        self.test_f1 = F1Score(num_classes=vocab_size, average='macro', task="multiclass")
        self.phoneme_f1_scores = defaultdict(float)
        self.phoneme_counts = defaultdict(int)
    
    def encode(self, x):
        batch_size = x.shape[0]
        
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Early sensor fusion
        x = self.sensor_fusion(x)
        
        # Convolutional processing
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        x = self.final_conv(x)
        
        # Reshape to sequence
        x = self.to_sequence(x)
        x = x.squeeze(-1).transpose(1, 2)
        x = self.proj_to_transformer(x)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Transformer processing
        for transformer in self.transformer_layers:
            x = transformer(x)
        
        # Extract CLS representation
        cls_representation = x[:, 0]
        
        # Feature projection
        features = self.feature_norm(cls_representation)
        features = self.feature_projection(features)
        
        return features
    
    def forward(self, x):
        # x: (B, C, T) - MEG channels x time points
        features = self.encode(x)
        
        predictions = {}
        for head_name, head_layer in self.heads.items():
            predictions[head_name] = head_layer(features)
        
        return predictions['phoneme']  # Return phoneme predictions for compatibility
    
    def compute_hce_loss(self, predictions, phoneme_labels):
        """Compute HCE loss"""
        device = phoneme_labels.device
        batch_size = phoneme_labels.size(0)
        ce_loss = nn.CrossEntropyLoss(label_smoothing=self.hparams.label_smoothing)
        
        # Get hierarchical labels - initialize with -1 for invalid entries
        hierarchical_labels = {
            'broad': torch.full((batch_size,), -1, device=device),
            'manner': torch.full((batch_size,), -1, device=device),
            'place': torch.full((batch_size,), -1, device=device),
            'height': torch.full((batch_size,), -1, device=device),
            'backness': torch.full((batch_size,), -1, device=device)
        }
        
        # Fill in valid hierarchical labels
        for i, phoneme_id in enumerate(phoneme_labels.cpu().numpy()):
            labels = self.hierarchy.get_hierarchical_labels(int(phoneme_id))
            for level, label in labels.items():
                if level in hierarchical_labels:
                    hierarchical_labels[level][i] = label
        
        # Compute losses
        total_loss = 0
        loss_dict = {}
        
        # Phoneme loss
        if 'phoneme' in predictions:
            loss = ce_loss(predictions['phoneme'], phoneme_labels)
            loss_dict['phoneme'] = loss.item()
            total_loss += self.hce_weights.get('phoneme', 1.0) * loss
        
        # Hierarchical losses
        for level in ['broad', 'manner', 'place', 'height', 'backness']:
            if level in predictions and level in hierarchical_labels:
                valid_mask = hierarchical_labels[level] >= 0
                if valid_mask.any():
                    valid_preds = predictions[level][valid_mask]
                    valid_labels = hierarchical_labels[level][valid_mask]
                    
                    if len(valid_labels) > 0:
                        loss = ce_loss(valid_preds, valid_labels)
                        loss_dict[level] = loss.item()
                        total_loss += self.hce_weights.get(level, 0.1) * loss
        
        # Consistency loss
        if self.consistency_weight > 0:
            consistency_loss = self.compute_consistency_loss(predictions, phoneme_labels)
            loss_dict['consistency'] = consistency_loss.item()
            total_loss += self.consistency_weight * consistency_loss
        
        return total_loss, loss_dict
    
    def compute_consistency_loss(self, predictions, phoneme_labels):
        """Compute consistency between hierarchical predictions"""
        loss = 0
        
        phoneme_probs = F.softmax(predictions['phoneme'], dim=-1)
        
        if 'broad' in predictions:
            broad_probs = F.softmax(predictions['broad'], dim=-1)
            
            for broad_idx, (category, phoneme_set) in enumerate(self.hierarchy.broad_categories.items()):
                phoneme_indices = list(phoneme_set)
                if phoneme_indices:
                    category_prob_from_phonemes = phoneme_probs[:, phoneme_indices].sum(dim=1)
                    category_prob_direct = broad_probs[:, broad_idx]
                    loss += F.mse_loss(category_prob_from_phonemes, category_prob_direct)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        x, y = batch  # x: (B, C, T), y: (B,)
        
        # Get all predictions
        features = self.encode(x)
        predictions = {}
        for head_name, head_layer in self.heads.items():
            predictions[head_name] = head_layer(features)
        
        # Compute HCE loss
        loss, loss_dict = self.compute_hce_loss(predictions, y)
        
        # Metrics
        with torch.no_grad():
            preds = predictions['phoneme'].argmax(dim=-1)
            acc = (preds == y).float().mean()
            f1 = self.train_f1(predictions['phoneme'], y)
            
            # Track per-phoneme performance
            for i in range(len(y)):
                phoneme_id = y[i].item()
                self.phoneme_counts[phoneme_id] += 1
                if preds[i] == y[i]:
                    self.phoneme_f1_scores[phoneme_id] += 1
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_f1', f1, prog_bar=True)
        self.log('train_acc', acc)
        
        # Log individual loss components
        for k, v in loss_dict.items():
            self.log(f'train_loss_{k}', v)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        # Get all predictions
        features = self.encode(x)
        predictions = {}
        for head_name, head_layer in self.heads.items():
            predictions[head_name] = head_layer(features)
        
        # Compute HCE loss
        loss, loss_dict = self.compute_hce_loss(predictions, y)
        
        # Metrics
        preds = predictions['phoneme'].argmax(dim=-1)
        acc = (preds == y).float().mean()
        f1 = self.val_f1(predictions['phoneme'], y)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_f1_macro', f1, prog_bar=True)
        self.log('val_acc', acc)
        
        # Log individual loss components
        for k, v in loss_dict.items():
            self.log(f'val_loss_{k}', v)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        
        # Get all predictions
        features = self.encode(x)
        predictions = {}
        for head_name, head_layer in self.heads.items():
            predictions[head_name] = head_layer(features)
        
        # Compute HCE loss
        loss, loss_dict = self.compute_hce_loss(predictions, y)
        
        # Metrics
        preds = predictions['phoneme'].argmax(dim=-1)
        acc = (preds == y).float().mean()
        f1 = self.test_f1(predictions['phoneme'], y)
        
        self.log('test_loss', loss)
        self.log('test_f1_macro', f1)
        self.log('test_acc', acc)
        
        return loss
    
    def on_train_epoch_end(self):
        """Log per-phoneme performance statistics."""
        if self.current_epoch % 5 == 0:  # Log every 5 epochs
            print(f"\n{'='*50}")
            print(f"Epoch {self.current_epoch} - Per-Phoneme Performance:")
            
            # Calculate per-phoneme accuracy
            phoneme_accuracies = {}
            for phoneme_id in self.phoneme_counts:
                if self.phoneme_counts[phoneme_id] > 0:
                    accuracy = self.phoneme_f1_scores[phoneme_id] / self.phoneme_counts[phoneme_id]
                    phoneme_accuracies[phoneme_id] = accuracy
            
            # Find best and worst performing phonemes
            sorted_phonemes = sorted(phoneme_accuracies.items(), key=lambda x: x[1])
            
            print("Worst performing phonemes:")
            for pid, acc in sorted_phonemes[:5]:
                print(f"  Phoneme {pid}: {acc:.3f} (count: {self.phoneme_counts[pid]})")
            
            print("Best performing phonemes:")
            for pid, acc in sorted_phonemes[-5:]:
                print(f"  Phoneme {pid}: {acc:.3f} (count: {self.phoneme_counts[pid]})")
            
            print(f"{'='*50}\n")
            
            # Reset counters
            self.phoneme_f1_scores = defaultdict(float)
            self.phoneme_counts = defaultdict(int)
    
    def configure_optimizers(self):
        """Configure optimizer with different learning rates for different components."""
        params = []
        
        # Encoder parameters with base learning rate
        encoder_params = list(self.sensor_fusion.parameters()) + \
                        list(self.conv_blocks.parameters()) + \
                        list(self.final_conv.parameters()) + \
                        list(self.transformer_layers.parameters())
        
        params.append({
            'params': encoder_params,
            'lr': self.hparams.learning_rate
        })
        
        # Classification heads with higher learning rate
        heads_params = list(self.heads.parameters())
        params.append({
            'params': heads_params,
            'lr': self.hparams.learning_rate * 2  # Higher LR for heads
        })
        
        optimizer = torch.optim.AdamW(params, weight_decay=self.hparams.weight_decay)
        
        # Learning rate scheduling with warmup
        def lr_lambda(epoch):
            if epoch < self.hparams.warmup_epochs:
                # Linear warmup
                return (epoch + 1) / self.hparams.warmup_epochs
            else:
                # Cosine annealing
                progress = (epoch - self.hparams.warmup_epochs) / (self.hparams.total_epochs - self.hparams.warmup_epochs)
                return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }