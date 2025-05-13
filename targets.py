# targets.py
import torch
from torch import Tensor


def f_star(X: Tensor) -> Tensor:
    """
    The function used in the original script:

        f(x) = (x₁ + x₂ + x₁ x₂ x₃) / 2.
    """
    y = X[:, 0] + X[:, 1] + X[:, 0] * X[:, 1] * X[:, 2]
    return (y / 2.).unsqueeze(1)