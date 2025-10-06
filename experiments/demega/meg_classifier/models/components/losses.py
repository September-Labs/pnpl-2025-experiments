"""
Loss Functions for MEG Phoneme Classification

This module implements specialized loss functions including class-balanced focal loss
and supervised contrastive loss for handling imbalanced phoneme datasets.
"""

import torch
import torch.nn.functional as F


def focal_loss_mean_over_present_classes(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: torch.Tensor | float | None = None,
    label_smoothing: float = 0.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Focal loss that averages over present classes in the batch, with optional label smoothing.

    This loss function helps with class imbalance by:
    1. Focusing on hard-to-classify examples (focal loss)
    2. Averaging loss per class rather than per sample
    3. Optional class weighting and label smoothing

    Args:
        logits: [N, C] unnormalized scores
        targets: [N] class indices (int64)
        gamma: Focusing parameter for focal loss (higher = more focus on hard examples)
        alpha: None (no weighting), float (same weight for all), or Tensor[C] (per-class weights)
        label_smoothing: Smoothing parameter in [0, 1)
        eps: Numerical stability constant

    Returns:
        Scalar tensor: mean loss over classes present in targets
    """
    if logits.ndim != 2:
        raise ValueError("logits must be [N, C]")
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError("targets must be [N] and match batch size of logits")

    N, C = logits.shape
    if not (0.0 <= label_smoothing < 1.0):
        raise ValueError("label_smoothing must be in [0, 1)")

    # Log-probs and probs
    log_probs = F.log_softmax(logits, dim=1)
    probs = log_probs.exp().clamp(min=eps, max=1 - eps)

    # Build smoothed target distribution [N, C]
    if label_smoothing > 0.0:
        off_value = label_smoothing / (C - 1) if C > 1 else 0.0
        y = torch.full_like(probs, off_value)
        y.scatter_(1, targets.view(-1, 1), 1.0 - label_smoothing)
    else:
        # Exact one-hot for efficiency
        y = torch.zeros_like(probs)
        y.scatter_(1, targets.view(-1, 1), 1.0)

    # Alpha handling -> [C] then broadcast to [N, C]
    if alpha is None:
        alpha_vec = torch.ones(C, device=logits.device, dtype=logits.dtype)
    elif isinstance(alpha, float):
        alpha_vec = torch.full((C,), float(alpha), device=logits.device, dtype=logits.dtype)
    else:
        if alpha.numel() != C:
            raise ValueError(f"alpha tensor must have shape [C]={C}, got {tuple(alpha.shape)}")
        alpha_vec = alpha.to(device=logits.device, dtype=logits.dtype)
    alpha_mat = alpha_vec.unsqueeze(0).expand(N, C)

    # Focal factor per class
    focal = (1.0 - probs) ** gamma

    # Per-sample loss summed over classes using smoothed targets
    # loss_ij = - y_ij * alpha_j * (1 - p_ij)^gamma * log p_ij
    loss_ij = -y * alpha_mat * focal * log_probs
    loss_i = loss_ij.sum(dim=1)  # [N]

    # Average over present classes (attribute each sample's loss to its true class)
    uniq, inv = targets.unique(sorted=False, return_inverse=True)
    K = uniq.numel()
    sums = torch.zeros(K, device=logits.device, dtype=loss_i.dtype)
    cnts = torch.zeros(K, device=logits.device, dtype=loss_i.dtype)

    sums.index_add_(0, inv, loss_i)
    cnts.index_add_(0, inv, torch.ones_like(loss_i))

    class_means = sums / cnts.clamp_min(1)
    return class_means.mean()


def supervised_nt_xent(emb: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1, eps: float = 1e-8):
    """
    Supervised contrastive loss (NT-Xent variant) on a single view.

    This loss encourages embeddings of the same class to be close and
    embeddings of different classes to be far apart.

    Args:
        emb: [N, D] normalized embeddings
        labels: [N] class labels
        temperature: Temperature for similarity scaling (lower = sharper)
        eps: Small constant for numerical stability

    Returns:
        Scalar loss value
    """
    N, D = emb.shape
    if N <= 1:
        return emb.new_zeros(())

    # Compute similarity matrix
    sim = torch.matmul(emb, emb.T)
    sim = sim / temperature

    # Create mask for valid pairs (exclude self-similarity)
    logits_mask = torch.ones_like(sim, dtype=torch.bool)
    logits_mask.fill_(True)
    logits_mask.fill_diagonal_(False)

    # Create positive mask (same class pairs)
    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.T) & logits_mask

    # Compute numerically stable softmax
    sim_max, _ = torch.max(sim.masked_fill(~logits_mask, float("-inf")), dim=1, keepdim=True)
    sim = sim - sim_max.detach()

    # Compute denominator (all valid pairs)
    exp_sim = torch.exp(sim) * logits_mask
    denom = exp_sim.sum(dim=1) + eps

    # Compute numerator (positive pairs)
    pos_exp_sim = exp_sim * pos_mask
    num = pos_exp_sim.sum(dim=1)

    # Only compute loss for samples with positive pairs
    valid = pos_mask.any(dim=1)
    if valid.sum() == 0:
        return emb.new_zeros(())

    loss_i = -torch.log((num[valid] + eps) / denom[valid])
    return loss_i.mean()