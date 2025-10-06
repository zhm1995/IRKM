# train.py
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

from sympy.functions import betainc
import torch
import numpy as np

from kernels import laplacian_kernel_M
from targets import f_star, f_star_multi_index, random_rotation_matrix
from weights import empirical_weights_estimator, derivative_norm_estimator

np.set_printoptions(precision=3)


def sample_hypercube(n: int, d: int, *, device: torch.device) -> torch.Tensor:
    """±1 hyper‑cube samples."""
    x = np.random.normal(size=(n, d))
    return torch.from_numpy(np.sign(x)).double().to(device)


def sample_gaussian(n: int, d: int, *, device: torch.device) -> torch.Tensor:
    """Gaussian samples."""
    x = np.random.normal(size=(n, d))
    return torch.from_numpy(x).double().to(device)


def main(
        d: int,
        ratio: float,
        alpha: float = 0.0,
        s: float = 1e-2,
        diag: bool = True,
        *,
        n_test: int = 5_000,
        epochs: int = 10,
        seed: int = 0,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    L = math.sqrt(d)  # kernel bandwidth
    n_train = int(d ** ratio)
    # test set (fixed)
    ### sparse target function
    X_test = sample_hypercube(n_test, d, device=device)
    y_test = f_star(X_test)
    ### multi-index target function
    # Q = random_rotation_matrix(d)
    # X_test = sample_hypercube(n_test, d, device=device)
    # y_test = f_star_multi_index(X_test, Q)

    M = torch.eye(d, dtype=torch.float64, device=device)

    for epoch in range(epochs):
        tic = time.perf_counter()

        # 1.  training data
        X_train = sample_hypercube(n_train, d, device=device)
        y_train = f_star(X_train)
        # y_train = f_star_multi_index(X_train, Q)

        # 2.  solve Kα = y   (kernel ridge with tiny Tikhonov)
        K_train = laplacian_kernel_M(X_train, X_train, L, M)
        K_train += 1e-8 * torch.eye(n_train, dtype=torch.float64, device=device)
        beta = torch.linalg.solve(K_train, y_train).t()  # (1, n)

        # 3.  test error
        K_test = laplacian_kernel_M(X_train, X_test, L, M)
        preds = (beta @ K_test).t()
        mse = torch.mean((preds - y_test).pow(2)).item()

        # 4.  update weights
        M1 = empirical_weights_estimator(X_train, beta, L, M, s, diag)
        M2 = derivative_norm_estimator(X_train, beta, L, M, s, diag)
        M = (1 - alpha) * M1 + alpha * M2

        toc = time.perf_counter()
        if diag:
            top5 = M[:5].cpu().numpy()
            M = torch.diag(M)
        else:
            eigvals = torch.linalg.eigvalsh(M)
            eigvals = eigvals.flip(0)
            top5 = eigvals[:5].cpu().numpy()
        print(
            f"[epoch {epoch:02d}]  "
            f"MSE = {mse:.4e}   "
            f"top‑5 coords/eigenvals = {top5}   "
            f"time = {toc - tic:.2f}s"
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=200, help="input dimension")
    p.add_argument("--ratio", type=float, default=1.0,
                   help="n_train = d^ratio")
    p.add_argument("--epochs", type=int, default=20,
                   help="number of epochs")
    p.add_argument("--s", type=float, default=1e-2,
                   help="safe guard parameter")
    p.add_argument("--diag", type=bool, default=True,
                   help="whether to use diagonal matrix")
    p.add_argument("--alpha", type=float, default=0.0,
                   help="average parameter")
    args = p.parse_args()

    main(d=args.d, ratio=args.ratio, epochs=args.epochs, alpha=args.alpha, s=args.s, diag=args.diag)