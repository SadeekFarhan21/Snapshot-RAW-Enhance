"""Experiment 5 — Noise robustness sweep (fast).

Fixes the measurement rate at delta = 0.4 and sweeps the measurement SNR
across 6 levels from 5 dB (very noisy) to 35 dB (essentially clean).
Reports per-solver PSNR mean +/- std across the 5-image natural-scene
test set, using a single 16x16 center patch per scene (same as Exp 6).

This directly probes the bounded-noise BPDN bound: under the RIP, the
recovered coefficients satisfy ||c_hat - c||_2 <= C * epsilon for some
constant C depending on the RIP constant. PSNR should therefore drop
roughly linearly in SNR (dB). Anything flatter indicates the solver is
bottlenecked by something other than measurement noise (basis mismatch,
support recovery failure, regularization mis-tuning, etc.).

Sized for ~30 s wall-clock single-threaded.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import json
import time

import numpy as np
import matplotlib.pyplot as plt

from src.measurement import gaussian_sensing_matrix, add_measurement_noise
from src.solvers import omp, soft_threshold
from src.data import test_image_set, dct_basis_inverse
from src.metrics import psnr


def build_psi_t(N: int) -> np.ndarray:
    eye = np.eye(N)
    Psi_T = np.zeros((N, N))
    for k in range(N):
        Psi_T[:, k] = dct_basis_inverse(eye[:, k])
    return Psi_T


def fista_fast(A, y, lam, n_iters, L):
    """FISTA with pre-computed L."""
    step = 1.0 / L
    N = A.shape[1]
    x = np.zeros(N); z = np.zeros(N); t = 1.0
    for _ in range(n_iters):
        grad = A.T @ (A @ z - y)
        x_new = soft_threshold(z - step * grad, step * lam)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x, t = x_new, t_new
    return x


def admm_l1_fast(A, y, lam, n_iters, rho, AtA, Aty, chol):
    """ADMM with pre-factorized Cholesky."""
    from scipy.linalg import solve_triangular
    N = A.shape[1]
    x = np.zeros(N); z = np.zeros(N); u = np.zeros(N)
    for _ in range(n_iters):
        rhs = Aty + rho * (z - u)
        w = solve_triangular(chol, rhs, lower=True)
        x = solve_triangular(chol.T, w, lower=False)
        z = soft_threshold(x + u, lam / rho)
        u = u + x - z
    return z


def spectral_norm_fast(A, n_iters=20):
    v = np.random.default_rng(0).standard_normal(A.shape[1])
    v /= np.linalg.norm(v)
    for _ in range(n_iters):
        v = A.T @ (A @ v)
        nrm = np.linalg.norm(v)
        if nrm == 0:
            return 0.0
        v /= nrm
    return float(np.linalg.norm(A.T @ (A @ v)))


def extract_patch(img, side=16):
    H, W = img.shape
    i0 = (H - side) // 2; j0 = (W - side) // 2
    patch = img[i0:i0+side, j0:j0+side].copy()
    return (patch - patch.min()) / (patch.max() - patch.min() + 1e-8)


def main():
    side = 16
    N = side * side
    delta = 0.4
    M = int(round(delta * N))
    snr_grid = [5.0, 10.0, 15.0, 20.0, 25.0, 35.0]
    # lam scales with sigma ~ 10^(-snr/20)
    lam_table = {5.0: 0.20, 10.0: 0.10, 15.0: 0.05,
                 20.0: 0.03, 25.0: 0.02, 35.0: 0.008}

    test_set = test_image_set(size=64)
    print(f"[noise] scenes={[n for n, _ in test_set]}  delta={delta}  M={M}/{N}",
          flush=True)

    Psi_T = build_psi_t(N)
    rng = np.random.default_rng(1)

    # Pre-draw one sensing matrix per scene; reuse across all SNR levels.
    # This isolates noise effects from sensing-matrix variability.
    scene_data = []
    for name, img in test_set:
        patch = extract_patch(img, side=side)
        rng_local = np.random.default_rng(rng.integers(0, 2**31-1))
        A_pix = gaussian_sensing_matrix(M, N, rng_local)
        A_dct = A_pix @ Psi_T
        L = spectral_norm_fast(A_dct, n_iters=20)
        AtA = A_dct.T @ A_dct
        rho = 1.0
        chol = np.linalg.cholesky(AtA + rho * np.eye(N))
        Aty = A_dct.T  # placeholder; will be A_dct.T @ y per SNR
        scene_data.append({
            "name": name, "patch": patch, "A_pix": A_pix, "A_dct": A_dct,
            "L": L, "AtA": AtA, "chol": chol,
        })

    results: dict[float, list[dict]] = {snr: [] for snr in snr_grid}
    t0 = time.time()
    for snr_db in snr_grid:
        lam = lam_table[snr_db]
        sparsity = max(4, int(0.4 * M))
        for sd in scene_data:
            patch_vec = sd["patch"].reshape(-1)
            y_clean = sd["A_pix"] @ patch_vec
            y_noisy, _ = add_measurement_noise(
                y_clean, snr_db=snr_db,
                rng=np.random.default_rng(rng.integers(0, 2**31-1)))
            # in DCT basis
            Aty = sd["A_dct"].T @ y_noisy

            # OMP
            c_omp = omp(sd["A_dct"], y_noisy, sparsity=sparsity)
            rec_o = (Psi_T @ c_omp)
            # FISTA
            c_fista = fista_fast(sd["A_dct"], y_noisy, lam=lam,
                                  n_iters=200, L=sd["L"])
            rec_f = (Psi_T @ c_fista)
            # ADMM with cached cholesky
            c_admm = admm_l1_fast(sd["A_dct"], y_noisy, lam=lam,
                                   n_iters=100, rho=1.0,
                                   AtA=sd["AtA"], Aty=Aty, chol=sd["chol"])
            rec_a = (Psi_T @ c_admm)

            results[snr_db].append({
                "scene": sd["name"],
                "psnr_omp":   psnr(np.clip(rec_o, 0, 1), patch_vec),
                "psnr_fista": psnr(np.clip(rec_f, 0, 1), patch_vec),
                "psnr_admm":  psnr(np.clip(rec_a, 0, 1), patch_vec),
            })
        rs = results[snr_db]
        om = np.array([r["psnr_omp"] for r in rs])
        fi = np.array([r["psnr_fista"] for r in rs])
        ad = np.array([r["psnr_admm"] for r in rs])
        print(f"  SNR={snr_db:4.1f} dB  lam={lam:.3f}   "
              f"omp {om.mean():5.2f}±{om.std():.2f}   "
              f"fista {fi.mean():5.2f}±{fi.std():.2f}   "
              f"admm {ad.mean():5.2f}±{ad.std():.2f}   "
              f"({time.time()-t0:.1f}s)", flush=True)

    summary = {
        "delta": delta, "M": M, "N": N,
        "snr_db": snr_grid, "lam_table": lam_table,
        "scenes": [n for n, _ in test_set],
        "psnr_omp_mean":   [float(np.mean([r["psnr_omp"]   for r in results[s]])) for s in snr_grid],
        "psnr_omp_std":    [float(np.std ([r["psnr_omp"]   for r in results[s]])) for s in snr_grid],
        "psnr_fista_mean": [float(np.mean([r["psnr_fista"] for r in results[s]])) for s in snr_grid],
        "psnr_fista_std":  [float(np.std ([r["psnr_fista"] for r in results[s]])) for s in snr_grid],
        "psnr_admm_mean":  [float(np.mean([r["psnr_admm"]  for r in results[s]])) for s in snr_grid],
        "psnr_admm_std":   [float(np.std ([r["psnr_admm"]  for r in results[s]])) for s in snr_grid],
        "per_scene": {f"{s:.1f}": results[s] for s in snr_grid},
    }
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "noise_robustness.json", "w") as f:
        json.dump(summary, f, indent=2)

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5), constrained_layout=True)
    ax.errorbar(snr_grid, summary["psnr_omp_mean"],   yerr=summary["psnr_omp_std"],
                fmt="o-", capsize=3, label="OMP")
    ax.errorbar(snr_grid, summary["psnr_fista_mean"], yerr=summary["psnr_fista_std"],
                fmt="s-", capsize=3, label="FISTA")
    ax.errorbar(snr_grid, summary["psnr_admm_mean"],  yerr=summary["psnr_admm_std"],
                fmt="^-", capsize=3, label="ADMM")
    ax.set_xlabel("Measurement SNR (dB)")
    ax.set_ylabel("Reconstruction PSNR (dB)")
    ax.set_title(f"Noise robustness at $\\delta={delta}$, mean ± std across 5 scenes")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.savefig(ROOT / "figures" / "noise_robustness.png", dpi=140)
    print(f"\n[noise] saved figures/noise_robustness.png", flush=True)


if __name__ == "__main__":
    main()
