"""Rate-distortion curves on a natural image.

Given a 64x64 grayscale image patch we treat the 2D-DCT coefficients as the
sparse representation. For each measurement rate delta = M/N we sample a
Gaussian sensing matrix, take noisy measurements at a fixed SNR, recover via
OMP / FISTA / ADMM, and report PSNR + SSIM. Block-based so the problem
dimensions stay tractable.
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
from skimage.metrics import structural_similarity as ssim

from src.measurement import gaussian_sensing_matrix, add_measurement_noise
from src.solvers import omp, fista, admm_l1
from src.data import standard_image, dct_basis_apply, dct_basis_inverse
from src.metrics import psnr


def build_psi_t(N: int) -> np.ndarray:
    """Dense Psi^T (DCT synthesis matrix). Cached at module level."""
    eye = np.eye(N)
    Psi_T = np.zeros((N, N))
    for k in range(N):
        Psi_T[:, k] = dct_basis_inverse(eye[:, k])
    return Psi_T


def reconstruct_block(
    block: np.ndarray,
    Psi_T: np.ndarray,
    delta: float,
    snr_db: float,
    rng: np.random.Generator,
    solver: str,
    omp_sparsity: int = 80,
):
    """Reconstruct one (sqrt(N) x sqrt(N)) image block via CS in DCT domain.

    Psi_T (block N x N DCT synthesis matrix) is supplied externally so we don't
    rebuild it per block.
    """
    side = block.shape[0]
    N = side * side
    M = max(2, int(round(delta * N)))
    x = block.reshape(-1)
    A_pix = gaussian_sensing_matrix(M, N, rng)
    y_clean = A_pix @ x
    y_noisy, _ = add_measurement_noise(y_clean, snr_db=snr_db, rng=rng)
    A_dct = A_pix @ Psi_T

    if solver == "omp":
        c_hat = omp(A_dct, y_noisy, sparsity=min(omp_sparsity, M))
    elif solver == "fista":
        c_hat, _ = fista(A_dct, y_noisy, lam=0.02, n_iters=400)
    elif solver == "admm":
        c_hat = admm_l1(A_dct, y_noisy, lam=0.02, rho=1.0, n_iters=200)
    else:
        raise ValueError(solver)

    x_hat = dct_basis_inverse(c_hat)
    return x_hat.reshape(side, side)


def reconstruct_image(
    img: np.ndarray,
    Psi_T: np.ndarray,
    block_size: int,
    delta: float,
    snr_db: float,
    rng: np.random.Generator,
    solver: str,
) -> np.ndarray:
    H, W = img.shape
    out = np.zeros_like(img)
    for i in range(0, H, block_size):
        for j in range(0, W, block_size):
            block = img[i : i + block_size, j : j + block_size]
            out[i : i + block_size, j : j + block_size] = reconstruct_block(
                block, Psi_T=Psi_T, delta=delta, snr_db=snr_db, rng=rng, solver=solver
            )
    return np.clip(out, 0.0, 1.0)


def main():
    img = standard_image(size=64)  # 64x64
    block_size = 16
    deltas = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    snr_db = 30.0
    solvers = ["omp", "fista", "admm"]
    rng = np.random.default_rng(0)

    print(f"Precomputing Psi_T ({block_size**2}x{block_size**2})...")
    Psi_T = build_psi_t(block_size * block_size)

    results = {s: {"psnr": [], "ssim": []} for s in solvers}
    recon_grid: dict[tuple[str, float], np.ndarray] = {}

    t0 = time.time()
    for solver in solvers:
        for delta in deltas:
            rng_local = np.random.default_rng(rng.integers(0, 2**31 - 1))
            recon = reconstruct_image(
                img, Psi_T=Psi_T, block_size=block_size, delta=delta,
                snr_db=snr_db, rng=rng_local, solver=solver,
            )
            p = psnr(recon, img)
            s = ssim(img, recon, data_range=1.0)
            results[solver]["psnr"].append(p)
            results[solver]["ssim"].append(s)
            recon_grid[(solver, delta)] = recon
            print(f"  {solver:6s} delta={delta:.2f}  PSNR={p:5.2f} dB  SSIM={s:.3f}  ({time.time()-t0:.1f}s)")

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "rate_distortion.json", "w") as f:
        json.dump({"deltas": deltas, "snr_db": snr_db, "results": results}, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for solver in solvers:
        axes[0].plot(deltas, results[solver]["psnr"], "o-", label=solver.upper())
        axes[1].plot(deltas, results[solver]["ssim"], "o-", label=solver.upper())
    axes[0].set_xlabel(r"measurement rate $\delta = M/N$")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title(f"Rate-distortion (PSNR), SNR={snr_db} dB")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].set_xlabel(r"measurement rate $\delta = M/N$")
    axes[1].set_ylabel("SSIM")
    axes[1].set_title(f"Rate-distortion (SSIM), SNR={snr_db} dB")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.savefig(ROOT / "figures" / "rate_distortion.png", dpi=140)
    print(f"Saved {ROOT/'figures'/'rate_distortion.png'}")

    # Qualitative grid: rows = solver, cols = delta
    fig2, axes2 = plt.subplots(
        len(solvers) + 1, len(deltas), figsize=(2.0 * len(deltas), 2.0 * (len(solvers) + 1))
    )
    for j, delta in enumerate(deltas):
        axes2[0, j].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes2[0, j].set_title(f"$\\delta={delta:.2f}$")
        axes2[0, j].axis("off")
    axes2[0, 0].set_ylabel("orig", rotation=0, labelpad=20)
    for i, solver in enumerate(solvers, start=1):
        for j, delta in enumerate(deltas):
            axes2[i, j].imshow(recon_grid[(solver, delta)], cmap="gray", vmin=0, vmax=1)
            axes2[i, j].axis("off")
        axes2[i, 0].set_ylabel(solver.upper(), rotation=0, labelpad=20)
    fig2.suptitle(f"Reconstructions vs measurement rate (SNR={snr_db} dB)")
    fig2.tight_layout()
    fig2.savefig(ROOT / "figures" / "rate_distortion_qualitative.png", dpi=140)
    print(f"Saved {ROOT/'figures'/'rate_distortion_qualitative.png'}")


if __name__ == "__main__":
    main()
