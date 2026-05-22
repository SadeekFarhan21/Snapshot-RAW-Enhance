"""Evaluation metrics for sparse recovery and image reconstruction."""

from __future__ import annotations

import numpy as np


def nmse(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    """Normalized mean squared error ||x_hat - x_true||^2 / ||x_true||^2."""
    denom = float(np.sum(x_true**2)) + 1e-12
    return float(np.sum((x_hat - x_true) ** 2)) / denom


def support_f1(x_hat: np.ndarray, x_true: np.ndarray, tol: float = 1e-6) -> float:
    """F1 score on the recovered support — useful for OMP-style methods."""
    sup_hat = np.abs(x_hat) > tol
    sup_true = np.abs(x_true) > tol
    tp = float(np.sum(sup_hat & sup_true))
    fp = float(np.sum(sup_hat & ~sup_true))
    fn = float(np.sum(~sup_hat & sup_true))
    if tp == 0.0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2.0 * precision * recall / (precision + recall)


def psnr(x_hat: np.ndarray, x_true: np.ndarray, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio in dB, assuming pixels in [0, data_range]."""
    mse = float(np.mean((x_hat - x_true) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * float(np.log10((data_range**2) / mse))


def recovery_success(x_hat: np.ndarray, x_true: np.ndarray, threshold: float = 1e-3) -> bool:
    """Standard CS phase-transition criterion: NMSE below threshold."""
    return nmse(x_hat, x_true) < threshold
