# weights.py
from __future__ import annotations

import torch
from torch import Tensor

from kernels import euclidean_distances_w, laplacian_kernel_w


@torch.no_grad()
def update_coordinate_weights(
    X: Tensor,
    sol: Tensor,          # shape (1, n)  – row vector from K⁻¹ y
    bandwidth: float,
    w: Tensor,
    s: float = 1e-2
) -> Tensor:
    """
    Recomputes anisotropic coordinate weights.

    The formula follows the one in your prototype:
        a = solᵀ
        U = diag(a) K' − diag(K' a)
        grad_j = ∑_i U_ij x_ij w_j
        w_j  = ‖grad_j‖² / n
    and finally renormalise so that Σ_j w_j = d.
    """
    # --- distance derivative -------------------------------------------------
    K = laplacian_kernel_w(X, X, bandwidth, w)
    dist = euclidean_distances_w(X, X, w, squared=False)
    K_prime = K / (dist + 1e-12) / bandwidth           # avoid /0

    # --- matrix U  -----------------------------------------------------------
    a = sol.t().contiguous()                           # (n, 1) column
    diag_a = torch.diag(a.view(-1))                    # (n, n)
    diag_Ka = torch.diag((K_prime @ a).view(-1))       # (n, n)
    U = diag_a @ K_prime - diag_Ka                     # (n, n)

    # --- gradient & new weights ---------------------------------------------
    grad = (U.t() @ X) * w.unsqueeze(0)                # (n, d)
    new_w = grad.norm(dim=0).pow(2) / X.size(0)

    # small ridge 
    new_w += s / X.size(1)
    new_w = new_w * X.size(1) / new_w.sum()            # Σ w_j = d
    return new_w