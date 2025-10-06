# targets.py
import torch
from torch import Tensor
import numpy as np


def f_star(X: Tensor) -> Tensor:
    """
    The function used in the original script:

        f(x) = (x₁ + x₂ + x₁ x₂ x₃) / 2.
    """
    y = X[:, 0] + X[:, 1] + X[:, 0] * X[:, 1] * X[:, 2]
    return (y / 2.).unsqueeze(1)


def random_rotation_matrix(n=3):
    # Step 1: Generate a random n x n matrix with normally distributed entries
    A = np.random.randn(n, n)

    # Step 2: Compute the QR decomposition of A.
    # Q is an orthogonal matrix, and R is an upper-triangular matrix.
    Q, R = np.linalg.qr(A)

    # Step 3: Adjust Q by the sign of R's diagonal elements.
    # np.sign(np.diag(R)) returns a vector of signs for the diagonal of R.
    # Multiplying Q by this vector (broadcasted along the columns) corrects the sign.
    Q = Q * np.sign(np.diag(R))

    # Step 4: Ensure Q is a rotation matrix (determinant +1).
    # If det(Q) is -1, flip the sign of the first column.
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]

    return Q


def f_star_multi_index(X, Q):
    X = X @ Q
    y = X[:, 0] + X[:, 1] + X[:, 0] * X[:, 1] * X[:, 2]
    y = y.reshape(-1, 1)
    y = y / 2
    return y