# weights.py
from __future__ import annotations

import torch
from torch import Tensor

from kernels import euclidean_distances_M, laplacian_kernel_M, gaussian_kernel_M


def matrix_sqrt_eigh(M):
    # M should be symmetric PSD
    eigvals, eigvecs = torch.linalg.eigh(M)
    eigvals_clamped = eigvals.clamp(min=0)  # numerical stability
    sqrt_eigvals = eigvals_clamped.sqrt()
    return eigvecs @ torch.diag(sqrt_eigvals) @ eigvecs.T


@torch.no_grad()
def empirical_weights_estimator(
        X: Tensor,
        sol: Tensor,
        L: float,
        P: Tensor,
        s: float = 1e-2,
        diag: bool = True
) -> Tensor:
    """
    Empirical estimator for the P-weighted gradient covariance matrix.

    Args:
        X: (n, d) sample matrix.
        sol: (1, n) vector of coefficients.
        L: Laplacian kernel length scale.
        P: (d, d) AGOP matrix (positive semi-definite).
        s: safe guard parameter.
        diag: whether to use diagonal matrix.
    Returns:
        (d, d) PSD matrix estimating AGOP matrix.
    """
    d = X.shape[1]

    K = laplacian_kernel_M(X, X, L, P)  # (n, n)
    dist = euclidean_distances_M(X, X, P, squared=False)  # (n, n)

    # Derivative magnitude: K' = K / (L * r), safe for r = 0
    K_deriv = torch.where(dist > 0, K / (L * dist), torch.zeros_like(K))

    a = sol.reshape(-1, 1)  # ensure column (n, 1)
    U = torch.diag(a.view(-1)) @ K_deriv - torch.diag((K_deriv @ a).view(-1))
    grad = U.t() @ (X @ P)  # (n, d)
    M = grad.t() @ grad / X.shape[0]  # (d, d)
    if diag:
        M = torch.diagonal(M)
        M += s / d * torch.ones(d).to(M.device)
        M = d * M / torch.sum(M)
    else:
        M += s / d * torch.eye(d).to(M.device)
        M = d * M / torch.trace(M)
    return M


@torch.no_grad()
def derivative_norm_estimator(
        X: Tensor,
        sol: Tensor,
        L: float,
        P: Tensor,
        s: float = 1e-2,
        diag: bool = True,
        r_eps: float = 1e-8,
        mask_tol: float = 1e-8,
) -> Tensor:
    """
    Estimates the derivative norm using a Laplacian-graph structure.

    Args:
        X: (n, d) data.
        sol: (1, n) vector of coefficients.
        L: Laplacian length scale.
        P: (d, d) AGOP matrix (positive semi-definite).
        s: safe guard parameter.
        diag: whether to use diagonal matrix.
        r_eps: regularization for small distances.
        mask_tol: sparsity threshold for small weights.
    Returns:
        (d, d) PSD matrix estimating the derivative of the norm.
    """
    d = X.shape[1]

    K = laplacian_kernel_M(X, X, L, P)
    r = euclidean_distances_M(X, X, P, squared=False)
    inv_r = torch.where(r > r_eps, 1.0 / r, torch.zeros_like(r))

    a = sol.reshape(-1)  # (n,)
    aa = a[:, None] * a[None, :]  # outer(a, a)
    W = (aa * K * inv_r) / L  # weights
    W.fill_diagonal_(0.0)
    W = 0.5 * (W + W.T)

    # (Optional) sparsify tiny weights to reduce dynamic range / summation error

    if mask_tol > 0:
        row_max = W.max(dim=1, keepdim=True).values.clamp_min(torch.finfo(W.dtype).tiny)
        W = torch.where(W >= mask_tol * row_max, W, torch.zeros_like(W))

    # Graph Laplacian
    D = torch.diag(W.sum(dim=1))
    Lw = D - W

    Ps = torch.sqrt(P) if diag else matrix_sqrt_eigh(P)
    M = 2.0 * X.T @ Lw @ X
    M = Ps.T @ M @ Ps / X.shape[0]

    if diag:
        M = torch.diagonal(M)
        M += s / d * torch.ones(d).to(M.device)
        M = d * M / torch.sum(M)
    else:
        M += s / d * torch.eye(d).to(M.device)
        M = d * M / torch.trace(M)
    return M
