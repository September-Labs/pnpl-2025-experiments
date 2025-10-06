"""
MEG-DeBERTa Model Components

This package contains the core building blocks for the MEG phoneme classifier:
- DeBERTa-style attention mechanism
- MEG Conformer layers
- Specialized loss functions
- IPA phonetic features
"""

from .deberta_attention import DisentangledSelfAttention
from .conformer import MEGConformerLayer, normalize
from .losses import focal_loss_mean_over_present_classes, supervised_nt_xent
from .ipa_features import get_ipa_feature_matrix

__all__ = [
    "DisentangledSelfAttention",
    "MEGConformerLayer",
    "normalize",
    "focal_loss_mean_over_present_classes",
    "supervised_nt_xent",
    "get_ipa_feature_matrix",
]