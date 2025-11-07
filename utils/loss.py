from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def sim_loss_func(z: Tensor, p: Tensor, mask: Tensor) -> Tensor:
    """
    Similarity loss mirroring the TensorFlow implementation.
    """
    z_norm = F.normalize(z, p=2.0, dim=-1)
    p_norm = F.normalize(p, p=2.0, dim=-1)
    distance = torch.sum(z_norm * p_norm, dim=-1)

    if mask.dtype != torch.bool:
        mask_bool = mask > 0.5
    else:
        mask_bool = mask

    if mask_bool.dim() > distance.dim():
        mask_bool = mask_bool.squeeze(-1)

    if mask_bool.shape != distance.shape:
        raise ValueError("Mask shape must match distance shape after squeezing.")

    if mask_bool.sum().item() >= 7:
        masked_distance = distance.masked_select(mask_bool)
        loss = 1.0 - masked_distance.mean()
    else:
        loss = distance.new_tensor(0.001)

    return loss


def sqi_loss_func(p_score: Tensor, n_score: Tensor) -> Tensor:
    """
    Implements the original SQI loss using PyTorch operations.
    """
    loss = -torch.mean(torch.log(p_score + 1e-5) + torch.log(1.0 - n_score + 1e-5)) + torch.log(
        torch.tensor(2.0, dtype=p_score.dtype, device=p_score.device)
    )
    return loss


def cce_loss_func(y_pred: Tensor, labels: Tensor) -> Tensor:
    """
    Categorical cross-entropy assuming probability predictions and one-hot labels.
    """
    epsilon = 1e-7
    y_pred = torch.clamp(y_pred, epsilon, 1.0 - epsilon)
    loss = -(labels * torch.log(y_pred)).sum(dim=-1)
    return loss.mean()


def bce_loss_func(y_pred: Tensor, labels: Tensor) -> Tensor:
    """
    Binary cross-entropy for per-element predictions.
    """
    return F.binary_cross_entropy(y_pred, labels)


__all__ = ["sim_loss_func", "sqi_loss_func", "cce_loss_func", "bce_loss_func"]
