# train.py
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import numpy as np

from kernels import laplacian_kernel_w
from targets import f_star
from weights import update_coordinate_weights

np.set_printoptions(precision=3)

def sample_hypercube(n: int, d: int, *, device: torch.device) -> torch.Tensor:
    """±1 hyper‑cube samples."""
    x = np.random.normal(size=(n, d))
    return torch.from_numpy(np.sign(x)).double().to(device)


def experiment(
    d: int,
    ratio: float,
    *,
    n_test: int = 5_000,
    epochs: int = 10,
    seed: int = 0,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    L = math.sqrt(d)                # kernel bandwidth
    n_train = int(d ** ratio)

    # test set (fixed)
    X_test = sample_hypercube(n_test, d, device=device)
    y_test = f_star(X_test)

    w = torch.ones(d, dtype=torch.float64, device=device)

    for epoch in range(epochs):
        tic = time.perf_counter()

        # 1.  training data
        X_train = sample_hypercube(n_train, d, device=device)
        y_train = f_star(X_train)

        # 2.  solve Kα = y   (kernel ridge with tiny Tikhonov)
        K_train = laplacian_kernel_w(X_train, X_train, L, w)
        K_train += 1e-8 * torch.eye(n_train, dtype=torch.float64, device=device)
        alpha = torch.linalg.solve(K_train, y_train).t()          # (1, n)

        # 3.  test error
        K_test = laplacian_kernel_w(X_train, X_test, L, w)
        preds = (alpha @ K_test).t()
        mse = torch.mean((preds - y_test).pow(2)).item()

        # 4.  update weights
        w = update_coordinate_weights(X_train, alpha, L, w)

        toc = time.perf_counter()
        top5 = w[:5].cpu().numpy()

        print(
            f"[epoch {epoch:02d}]  "
            f"MSE = {mse:.4e}   "
            f"top‑5 w = {top5}   "
            f"time = {toc - tic:.2f}s"
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=200, help="ambient dimension")
    p.add_argument("--ratio", type=float, default=0.7,
                   help="n_train = d^ratio")
    p.add_argument("--epochs", type=int, default=10,
                   help="number of epochs")
    args = p.parse_args()

    experiment(d=args.d, ratio=args.ratio, epochs=args.epochs)