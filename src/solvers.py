"""Classical sparse recovery solvers: OMP, ISTA, FISTA, ADMM.

All solvers expose the same calling convention: take a sensing matrix A
(or matvec/rmatvec callables) and measurements y, return the recovered
N-vector. Iteration counts and tolerances are kept explicit so that we
can fairly compare the same compute budget across solvers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Orthogonal Matching Pursuit
# ---------------------------------------------------------------------------

def omp(A: np.ndarray, y: np.ndarray, sparsity: int, tol: float = 1e-6) -> np.ndarray:
    """Greedy OMP for y = A x.

    At each iteration: pick the column most correlated with the current
    residual, then re-solve least squares on the active support.
    Stops at the requested sparsity or when the residual is below tol.
    """
    M, N = A.shape
    x = np.zeros(N)
    residual = y.copy()
    support: list[int] = []

    for _ in range(sparsity):
        correlations = A.T @ residual
        # avoid re-picking
        for j in support:
            correlations[j] = 0.0
        idx = int(np.argmax(np.abs(correlations)))
        support.append(idx)
        A_S = A[:, support]
        # least squares on active support
        x_S, *_ = np.linalg.lstsq(A_S, y, rcond=None)
        residual = y - A_S @ x_S
        if np.linalg.norm(residual) < tol:
            break

    x[support] = x_S
    return x


# ---------------------------------------------------------------------------
# Iterative shrinkage thresholding (ISTA / FISTA)
# ---------------------------------------------------------------------------

def soft_threshold(x: np.ndarray, t: float) -> np.ndarray:
    """Elementwise soft-threshold (the prox operator of t * ||.||_1)."""
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


@dataclass
class ISTAHistory:
    objective: list[float] = field(default_factory=list)
    nmse: list[float] = field(default_factory=list)


def ista(
    A: np.ndarray,
    y: np.ndarray,
    lam: float,
    n_iters: int = 500,
    x_true: np.ndarray | None = None,
) -> tuple[np.ndarray, ISTAHistory]:
    """Vanilla ISTA for min_x 0.5 ||Ax - y||^2 + lam ||x||_1.

    Step size 1/L where L is the largest eigenvalue of A^T A. We compute L
    via power iteration on A^T A so the routine also works with implicit
    operators if you swap the matvecs.
    """
    L = _spectral_norm_sq(A)
    step = 1.0 / L
    N = A.shape[1]
    x = np.zeros(N)
    hist = ISTAHistory()

    for _ in range(n_iters):
        grad = A.T @ (A @ x - y)
        x = soft_threshold(x - step * grad, step * lam)
        if x_true is not None:
            hist.objective.append(
                0.5 * float(np.sum((A @ x - y) ** 2)) + lam * float(np.sum(np.abs(x)))
            )
            denom = float(np.sum(x_true**2)) + 1e-12
            hist.nmse.append(float(np.sum((x - x_true) ** 2)) / denom)
    return x, hist


def fista(
    A: np.ndarray,
    y: np.ndarray,
    lam: float,
    n_iters: int = 500,
    x_true: np.ndarray | None = None,
) -> tuple[np.ndarray, ISTAHistory]:
    """FISTA — Nesterov-accelerated ISTA. O(1/k^2) on the objective gap."""
    L = _spectral_norm_sq(A)
    step = 1.0 / L
    N = A.shape[1]
    x = np.zeros(N)
    z = x.copy()
    t = 1.0
    hist = ISTAHistory()

    for _ in range(n_iters):
        grad = A.T @ (A @ z - y)
        x_new = soft_threshold(z - step * grad, step * lam)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x, t = x_new, t_new
        if x_true is not None:
            hist.objective.append(
                0.5 * float(np.sum((A @ x - y) ** 2)) + lam * float(np.sum(np.abs(x)))
            )
            denom = float(np.sum(x_true**2)) + 1e-12
            hist.nmse.append(float(np.sum((x - x_true) ** 2)) / denom)
    return x, hist


# ---------------------------------------------------------------------------
# ADMM for the constrained form (Basis Pursuit Denoising)
# ---------------------------------------------------------------------------

def admm_l1(
    A: np.ndarray,
    y: np.ndarray,
    lam: float,
    rho: float = 1.0,
    n_iters: int = 200,
) -> np.ndarray:
    """ADMM for min_x 0.5 ||Ax - y||^2 + lam ||x||_1.

    Splits x -> (x, z) with constraint x = z, then alternates between a
    least-squares x-update and a soft-threshold z-update. Caches the
    factorization of (A^T A + rho I) once up front.
    """
    M, N = A.shape
    AtA = A.T @ A
    Aty = A.T @ y
    # Cholesky of (A^T A + rho I) — symmetric positive definite for rho > 0.
    chol = np.linalg.cholesky(AtA + rho * np.eye(N))

    x = np.zeros(N)
    z = np.zeros(N)
    u = np.zeros(N)
    for _ in range(n_iters):
        rhs = Aty + rho * (z - u)
        # solve via cached Cholesky
        x = _chol_solve(chol, rhs)
        z = soft_threshold(x + u, lam / rho)
        u = u + x - z
    return z


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spectral_norm_sq(A: np.ndarray, n_iters: int = 50) -> float:
    """Largest eigenvalue of A^T A via power iteration."""
    N = A.shape[1]
    v = np.random.default_rng(0).standard_normal(N)
    v /= np.linalg.norm(v)
    for _ in range(n_iters):
        v = A.T @ (A @ v)
        nrm = np.linalg.norm(v)
        if nrm == 0.0:
            return 0.0
        v /= nrm
    return float(np.linalg.norm(A.T @ (A @ v)))


def _chol_solve(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve (L L^T) x = b given a lower-triangular Cholesky factor L."""
    from scipy.linalg import solve_triangular

    y = solve_triangular(L, b, lower=True)
    return solve_triangular(L.T, y, lower=False)
