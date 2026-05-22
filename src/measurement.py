"""Forward measurement model for compressive sensing.

Implements y = A x + w with several sensing operator options. The Gaussian
operator is the workhorse — its column-normalized form has well-understood RIP
behavior. The partial-DCT operator is used for image-domain experiments where
fast transforms matter.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dct, idct


def gaussian_sensing_matrix(M: int, N: int, rng: np.random.Generator) -> np.ndarray:
    """Column-normalized i.i.d. Gaussian sensing matrix A in R^{M x N}.

    Columns are normalized to unit l2 norm so that the per-atom inner products
    used by OMP and the soft-threshold step used by ISTA are on a common scale.
    """
    A = rng.standard_normal((M, N)) / np.sqrt(M)
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    return A


def partial_dct_operator(M: int, N: int, rng: np.random.Generator):
    """Returns (forward, adjoint) callables for a partial-DCT sensing operator.

    Picks M random frequency rows of the orthonormal DCT-II matrix and applies
    a random column sign flip (a structurally-random ensemble). Fast O(N log N)
    apply via scipy.fft.dct.
    """
    rows = rng.choice(N, size=M, replace=False)
    signs = rng.choice([-1.0, 1.0], size=N)
    scale = np.sqrt(N / M)

    def forward(x: np.ndarray) -> np.ndarray:
        z = dct(signs * x, type=2, norm="ortho")
        return scale * z[rows]

    def adjoint(y: np.ndarray) -> np.ndarray:
        z = np.zeros(N)
        z[rows] = scale * y
        return signs * idct(z, type=2, norm="ortho")

    return forward, adjoint, rows, signs


def add_measurement_noise(
    y_clean: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Adds Gaussian noise at a target signal-to-noise ratio.

    Returns the noisy measurements and the noise std dev (useful for setting
    epsilon in the L1-constrained recovery).
    """
    signal_power = float(np.mean(y_clean**2))
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    sigma = np.sqrt(noise_power)
    w = rng.standard_normal(y_clean.shape) * sigma
    return y_clean + w, sigma
