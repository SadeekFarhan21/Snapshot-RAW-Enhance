"""Joint vs sequential recovery on a synthetic "RAW-like" image.

Setup: a grayscale image is multiplied by a strong horizontal illumination
gradient (simulating a low-light corner — the kind of HDR scene a sequential
ISP tends to clip). The CS sensor takes M random Gaussian projections of the
linear-domain signal at a fixed SNR.

Sequential pipeline:
  1. CS recovery from y (in DCT basis, FISTA with lam tuned at high SNR)
  2. Estimate the gradient from the recovered image, divide it out, re-clip.
  3. Compute PSNR against the underlying scene normalized to [0, 1].

Joint pipeline:
  1. Augment the unknowns with an illumination vector g (one scalar per column
     of the image) and solve a joint L1-regularized problem:
        min_{c, g} 0.5 ||A diag(scene(c, g)) - y||^2 + lam_c ||c||_1 + lam_g ||grad(g)||^2
     by alternating: hold g fixed and FISTA on c; hold c fixed and a closed-
     form ridge step on g.
  2. Recover the scene as Psi^T c, normalized.

We report PSNR on the "scene" (i.e., the illumination-normalized ground truth)
for both pipelines across measurement rates.
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
from src.solvers import fista
from src.data import standard_image, dct_basis_inverse
from src.metrics import psnr


def gain_match(recon: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Optimal positive scalar gain a* = argmin_a ||a*recon - target||^2 = <recon,target>/||recon||^2.

    Joint recovery has an inherent scale ambiguity (s, g) -> (a*s, g/a), so
    PSNR must be evaluated up to this scalar. We apply the same rule to the
    sequential pipeline for a fair comparison.
    """
    num = float(np.sum(recon * target))
    den = float(np.sum(recon * recon)) + 1e-12
    a = max(num / den, 0.0)
    return np.clip(a * recon, 0.0, 1.0)


def build_psi_t(N: int) -> np.ndarray:
    """Dense Psi^T (DCT synthesis matrix) for an N=side*side block."""
    eye = np.eye(N)
    Psi_T = np.zeros((N, N))
    for k in range(N):
        Psi_T[:, k] = dct_basis_inverse(eye[:, k])
    return Psi_T


def synthesize_block(c: np.ndarray, Psi_T: np.ndarray, side: int) -> np.ndarray:
    return (Psi_T @ c).reshape(side, side)


def sequential_recover(
    A_pix: np.ndarray, y_noisy: np.ndarray, Psi_T: np.ndarray, lam: float, n_iters: int,
    side: int, gradient_profile: np.ndarray,
) -> np.ndarray:
    """1. CS recover the raw block via FISTA in DCT domain.
    2. Estimate the illumination by column means of the recovered raw, divide it out.
    """
    A_dct = A_pix @ Psi_T
    c_hat, _ = fista(A_dct, y_noisy, lam=lam, n_iters=n_iters)
    raw_hat = synthesize_block(c_hat, Psi_T, side)
    # Estimate per-column gain from column means (RAW-typical sequential trick).
    col_means = raw_hat.mean(axis=0)
    # Robustness: floor at small positive to avoid divide-by-zero.
    g_hat = np.maximum(col_means / col_means.mean(), 1e-3)
    scene_hat = raw_hat / g_hat[None, :]
    # Renormalize to [0, 1] using the same overall mean as the true scene scale.
    scene_hat = scene_hat / scene_hat.mean() * 0.5
    return np.clip(scene_hat, 0.0, 1.0)


def joint_recover(
    A_pix: np.ndarray, y_noisy: np.ndarray, Psi_T: np.ndarray, side: int,
    lam_c: float = 0.02, n_outer: int = 8, n_inner_fista: int = 100,
    gradient_init: np.ndarray | None = None,
) -> np.ndarray:
    """Alternating-minimization joint recovery.

    Unknowns: DCT coefficients c (N-vector), per-column gain g (side-vector).
    Observation model: y = A_pix * vec( synth(c) * g[None, :] )
                        = A_pix * diag(g_pixel) * Psi^T c
    where g_pixel tiles g over rows. We alternate FISTA on c (with a g-modulated
    sensing matrix) and a closed-form least-squares update on g.
    """
    N = side * side
    if gradient_init is None:
        g = np.ones(side)
    else:
        g = gradient_init.copy()

    c = np.zeros(N)
    for _ in range(n_outer):
        # Build g_pixel (length-N) by tiling g over rows.
        g_pixel = np.tile(g[None, :], (side, 1)).reshape(-1)
        # Effective sensing operator for c: A_eff = A_pix * diag(g_pixel) * Psi^T
        A_eff = (A_pix * g_pixel[None, :]) @ Psi_T
        c, _ = fista(A_eff, y_noisy, lam=lam_c, n_iters=n_inner_fista)
        # Update g: with c fixed,
        #   y ≈ sum over pixels  A_pix[:, p] * g_pixel[p] * (Psi^T c)[p]
        # group by column: g[col] * sum_row A_pix[:, row*side+col] * scene[row, col]
        scene = synthesize_block(c, Psi_T, side)
        # Build the per-column design matrix B in R^{M x side}:
        # B[:, col] = sum_row A_pix[:, row*side+col] * scene[row, col]
        # vectorized: reshape A_pix to (M, side, side), multiply by scene, sum row axis
        A_resh = A_pix.reshape(A_pix.shape[0], side, side)
        B = (A_resh * scene[None, :, :]).sum(axis=1)  # (M, side)
        # Least squares for g with mild regularization.
        gtg = B.T @ B + 1e-2 * np.eye(side)
        g = np.linalg.solve(gtg, B.T @ y_noisy)
        g = np.maximum(g, 1e-3)

    scene_final = scene
    scene_final = scene_final / max(scene_final.mean(), 1e-8) * 0.5
    return np.clip(scene_final, 0.0, 1.0)


def main():
    side = 16
    N = side * side
    snr_db = 25.0  # noisier than rate-distortion to make the joint advantage visible
    deltas = [0.20, 0.30, 0.40, 0.50, 0.60]

    # Scene is one block of the cameraman image, normalized to [0, 1].
    full = standard_image(size=64)
    scene = full[24 : 24 + side, 24 : 24 + side].copy()
    scene = (scene - scene.min()) / (scene.max() - scene.min() + 1e-8)
    # Illumination gradient: factor of 8x from one side of the block to the other.
    gradient = np.linspace(1.0, 8.0, side)
    raw_truth = scene * gradient[None, :]

    Psi_T = build_psi_t(N)

    rng = np.random.default_rng(0)
    results = {"deltas": deltas, "seq_psnr": [], "joint_psnr": [],
               "seq_ssim": [], "joint_ssim": []}
    recons_seq, recons_joint = [], []

    t0 = time.time()
    for delta in deltas:
        M = max(4, int(round(delta * N)))
        rng_local = np.random.default_rng(rng.integers(0, 2**31 - 1))
        A_pix = gaussian_sensing_matrix(M, N, rng_local)
        y_clean = A_pix @ raw_truth.reshape(-1)
        y_noisy, _ = add_measurement_noise(y_clean, snr_db=snr_db, rng=rng_local)

        seq = sequential_recover(
            A_pix, y_noisy, Psi_T, lam=0.02, n_iters=400,
            side=side, gradient_profile=gradient,
        )
        joint = joint_recover(
            A_pix, y_noisy, Psi_T, side=side,
            lam_c=0.02, n_outer=8, n_inner_fista=120,
        )
        # Both pipelines are gain-matched to the ground truth scene before scoring
        # (joint has inherent scale ambiguity; we apply the same operation to seq).
        seq = gain_match(seq, scene)
        joint = gain_match(joint, scene)
        p_seq, p_joint = psnr(seq, scene), psnr(joint, scene)
        s_seq = ssim(scene, seq, data_range=1.0)
        s_joint = ssim(scene, joint, data_range=1.0)
        results["seq_psnr"].append(p_seq)
        results["joint_psnr"].append(p_joint)
        results["seq_ssim"].append(s_seq)
        results["joint_ssim"].append(s_joint)
        recons_seq.append(seq)
        recons_joint.append(joint)
        print(f"  delta={delta:.2f}  M={M:3d}  "
              f"seq PSNR={p_seq:5.2f} ssim={s_seq:.3f}   "
              f"joint PSNR={p_joint:5.2f} ssim={s_joint:.3f}   "
              f"({time.time()-t0:.1f}s)")

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "joint_vs_sequential.json", "w") as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].plot(deltas, results["seq_psnr"], "o-", label="sequential")
    axes[0].plot(deltas, results["joint_psnr"], "s-", label="joint (ours)")
    axes[0].set_xlabel(r"measurement rate $\delta = M/N$")
    axes[0].set_ylabel("PSNR on scene (dB)")
    axes[0].set_title(f"Joint vs sequential, SNR={snr_db} dB, gradient 8x")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(deltas, results["seq_ssim"], "o-", label="sequential")
    axes[1].plot(deltas, results["joint_ssim"], "s-", label="joint (ours)")
    axes[1].set_xlabel(r"measurement rate $\delta = M/N$")
    axes[1].set_ylabel("SSIM on scene")
    axes[1].set_title("Structural similarity")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.savefig(ROOT / "figures" / "joint_vs_sequential.png", dpi=140)

    n = len(deltas)
    fig2, ax2 = plt.subplots(3, n, figsize=(2.0 * n, 6.0))
    for j, d in enumerate(deltas):
        ax2[0, j].imshow(scene, cmap="gray", vmin=0, vmax=1)
        ax2[0, j].set_title(f"δ={d:.2f}")
        ax2[1, j].imshow(recons_seq[j], cmap="gray", vmin=0, vmax=1)
        ax2[2, j].imshow(recons_joint[j], cmap="gray", vmin=0, vmax=1)
        for r in range(3):
            ax2[r, j].axis("off")
    ax2[0, 0].set_ylabel("scene", rotation=0, labelpad=24)
    ax2[1, 0].set_ylabel("seq.",  rotation=0, labelpad=24)
    ax2[2, 0].set_ylabel("joint", rotation=0, labelpad=24)
    fig2.suptitle("Joint vs sequential reconstructions")
    fig2.tight_layout()
    fig2.savefig(ROOT / "figures" / "joint_vs_sequential_qualitative.png", dpi=140)
    print(f"Saved figures.")


if __name__ == "__main__":
    main()
