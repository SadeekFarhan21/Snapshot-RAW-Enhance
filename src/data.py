"""Data generation: synthetic sparse signals + grayscale test images."""

from __future__ import annotations

import numpy as np
from scipy.fft import dct, idct
from skimage import data as skdata, color, img_as_float, transform


def sparse_signal(N: int, S: int, rng: np.random.Generator) -> np.ndarray:
    """N-dimensional vector with S nonzero entries drawn from N(0, 1).

    Uniformly random support so each draw is an independent CS problem.
    """
    x = np.zeros(N)
    support = rng.choice(N, size=S, replace=False)
    x[support] = rng.standard_normal(S)
    return x


def standard_image(size: int = 128) -> np.ndarray:
    """Grayscale 'cameraman'-style test image, resized and normalized to [0, 1]."""
    img = skdata.camera()  # uint8, 512x512
    img = transform.resize(img, (size, size), anti_aliasing=True)
    return img_as_float(img).astype(np.float64)


def hdr_like_image(size: int = 128, dynamic_range: float = 50.0) -> np.ndarray:
    """Synthetic 'RAW-like' image: standard scene with a strong illumination gradient.

    Models the dynamic range that RAW sensors capture but a sequential
    pipeline tends to clip — the test bed for the joint vs sequential
    ablation. Values are not clipped to [0,1]; downstream code must handle
    that.
    """
    base = standard_image(size)
    # linear gradient across X simulates spatially-varying gain (e.g., low-light corner)
    grad = np.linspace(1.0, dynamic_range, size)[None, :]
    return base * grad


def dct_basis_apply(x: np.ndarray) -> np.ndarray:
    """2D DCT-II: image -> DCT coefficients (norm='ortho', flat vector)."""
    side = int(np.sqrt(x.size))
    img = x.reshape(side, side)
    coeffs = dct(dct(img, type=2, norm="ortho", axis=0), type=2, norm="ortho", axis=1)
    return coeffs.reshape(-1)


def dct_basis_inverse(c: np.ndarray) -> np.ndarray:
    """Inverse of dct_basis_apply."""
    side = int(np.sqrt(c.size))
    coeffs = c.reshape(side, side)
    img = idct(idct(coeffs, type=2, norm="ortho", axis=0), type=2, norm="ortho", axis=1)
    return img.reshape(-1)
