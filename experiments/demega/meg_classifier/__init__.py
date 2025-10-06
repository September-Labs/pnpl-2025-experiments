"""
DeMEGa Classifier

A state-of-the-art MEG phoneme classification model using disentangled attention
and class-balanced focal loss.
"""

from .models import SingleStageMEGClassifier, BalancedPhonemePretrainer

__version__ = "0.1.0"

__all__ = [
    "SingleStageMEGClassifier",
    "BalancedPhonemePretrainer",
    "__version__",
]