# kernels.py
from __future__ import annotations

import torch


def euclidean_distances_w(
    samples: torch.Tensor,
    centres: torch.Tensor,
    w: torch.Tensor,
    *,
    squared: bool = True,
) -> torch.Tensor:
    """
    Weighted (semi‑)Euclidean distance matrix.

    D_ij = (x_i − y_j)^⊤ diag(w) (x_i − y_j).

    Parameters
    ----------
    samples : (n, d)  tensor
    centres : (m, d)  tensor
    w       : (d,)    non‑negative coordinate weights
    squared : If False, return the square‑rooted distance.

    Returns
    -------
    (n, m) tensor of distances.
    """
    # ‖x‖_w² and ‖y‖_w²
    sample_norm = (samples * w) * samples
    sample_norm = sample_norm.sum(dim=1, keepdim=True)            # (n, 1)

    if samples is centres:
        centre_norm = sample_norm
    else:
        centre_norm = (centres * w) * centres
        centre_norm = centre_norm.sum(dim=1, keepdim=True)        # (m, 1)

    centre_norm = centre_norm.t()                                 # (1, m)

    # −2 xᵀ diag(w) y
    dists = samples @ (w.unsqueeze(1) * centres.t())              # (n, m)
    dists.mul_(-2.)
    dists.add_(sample_norm)
    dists.add_(centre_norm)

    if not squared:
        dists.clamp_(min=0.)
        dists.sqrt_()
    return dists


def laplacian_kernel_w(
    samples: torch.Tensor,
    centres: torch.Tensor,
    bandwidth: float,
    w: torch.Tensor,
) -> torch.Tensor:
    """
    Anisotropic Laplacian kernel

    K_ij = exp( −‖x_i − y_j‖_w / bandwidth ).

    The bandwidth *ℓ* > 0 controls the kernel’s width.
    """
    if bandwidth <= 0:
        raise ValueError("`bandwidth` must be positive.")
    dists = euclidean_distances_w(samples, centres, w, squared=False)
    dists.clamp_(min=0.)
    return torch.exp(-dists / bandwidth)