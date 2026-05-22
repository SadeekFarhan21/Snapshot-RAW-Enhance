"""Joint vs sequential recovery, averaged over multiple real-image scenes.

For each scene (a 16x16 patch from a real natural image), we apply a strong
horizontal illumination gradient (simulating a low-light corner / HDR
condition that a sequential ISP tends to clip), take Gaussian CS
measurements at SNR=25 dB, and recover via two pipelines:

  Sequential:
    1. CS recovery from y (FISTA in DCT basis)
    2. Estimate the gradient from the recovered image (column means)
    3. Divide it out, renormalize.

  Joint:
    1. Augment unknowns with an illumination vector g (one scalar per column)
    2. Solve a joint L1 + ridge-on-grad-g problem by alternating
       FISTA-on-c and ridge-on-g.

PSNR / SSIM on the illumination-normalized scene are reported as mean ± std
across the scene set, so the headline gain isn't a cameraman-specific
artifact.
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
from src.data import test_image_set, dct_basis_inverse
from src.metrics import psnr


def gain_match(recon: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Optimal positive scalar gain a* = <recon,target>/||recon||^2.

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
    side: int,
) -> np.ndarray:
    A_dct = A_pix @ Psi_T
    c_hat, _ = fista(A_dct, y_noisy, lam=lam, n_iters=n_iters)
    raw_hat = synthesize_block(c_hat, Psi_T, side)
    col_means = raw_hat.mean(axis=0)
    g_hat = np.maximum(col_means / col_means.mean(), 1e-3)
    scene_hat = raw_hat / g_hat[None, :]
    scene_hat = scene_hat / scene_hat.mean() * 0.5
    return np.clip(scene_hat, 0.0, 1.0)


def joint_recover(
    A_pix: np.ndarray, y_noisy: np.ndarray, Psi_T: np.ndarray, side: int,
    lam_c: float = 0.02, n_outer: int = 8, n_inner_fista: int = 100,
) -> np.ndarray:
    N = side * side
    g = np.ones(side)
    c = np.zeros(N)
    scene = synthesize_block(c, Psi_T, side)
    for _ in range(n_outer):
        g_pixel = np.tile(g[None, :], (side, 1)).reshape(-1)
        A_eff = (A_pix * g_pixel[None, :]) @ Psi_T
        c, _ = fista(A_eff, y_noisy, lam=lam_c, n_iters=n_inner_fista)
        scene = synthesize_block(c, Psi_T, side)
        A_resh = A_pix.reshape(A_pix.shape[0], side, side)
        B = (A_resh * scene[None, :, :]).sum(axis=1)
        gtg = B.T @ B + 1e-2 * np.eye(side)
        g = np.linalg.solve(gtg, B.T @ y_noisy)
        g = np.maximum(g, 1e-3)
    scene_final = scene / max(scene.mean(), 1e-8) * 0.5
    return np.clip(scene_final, 0.0, 1.0)


def extract_patch(img: np.ndarray, side: int = 16) -> np.ndarray:
    """Center patch, [0,1]-normalized."""
    H, W = img.shape
    i0 = (H - side) // 2
    j0 = (W - side) // 2
    patch = img[i0 : i0 + side, j0 : j0 + side].copy()
    return (patch - patch.min()) / (patch.max() - patch.min() + 1e-8)


def main():
    side = 16
    N = side * side
    snr_db = 25.0
    deltas = [0.20, 0.30, 0.40, 0.50, 0.60]
    gradient = np.linspace(1.0, 8.0, side)

    test_set = test_image_set(size=64)
    print(f"Test scenes: {[name for name, _ in test_set]}")
    Psi_T = build_psi_t(N)

    rng = np.random.default_rng(0)
    # per_delta -> list of dicts with per-scene results
    per_delta: dict[float, list[dict]] = {d: [] for d in deltas}
    viz_scene_name = "cameraman"
    recons_seq_viz: list[np.ndarray] = []
    recons_joint_viz: list[np.ndarray] = []
    viz_scene: np.ndarray | None = None

    t0 = time.time()
    for delta in deltas:
        M = max(4, int(round(delta * N)))
        for name, img in test_set:
            scene = extract_patch(img, side=side)
            raw_truth = scene * gradient[None, :]
            rng_local = np.random.default_rng(rng.integers(0, 2**31 - 1))
            A_pix = gaussian_sensing_matrix(M, N, rng_local)
            y_clean = A_pix @ raw_truth.reshape(-1)
            y_noisy, _ = add_measurement_noise(y_clean, snr_db=snr_db, rng=rng_local)

            seq = sequential_recover(A_pix, y_noisy, Psi_T, lam=0.02, n_iters=400, side=side)
            joint = joint_recover(A_pix, y_noisy, Psi_T, side=side,
                                  lam_c=0.02, n_outer=8, n_inner_fista=120)
            seq = gain_match(seq, scene)
            joint = gain_match(joint, scene)
            p_seq, p_joint = psnr(seq, scene), psnr(joint, scene)
            s_seq = ssim(scene, seq, data_range=1.0)
            s_joint = ssim(scene, joint, data_range=1.0)
            per_delta[delta].append({
                "scene": name,
                "seq_psnr": p_seq, "joint_psnr": p_joint,
                "seq_ssim": s_seq, "joint_ssim": s_joint,
            })
            if name == viz_scene_name:
                recons_seq_viz.append(seq)
                recons_joint_viz.append(joint)
                viz_scene = scene
        agg = per_delta[delta]
        seq_p = [r["seq_psnr"] for r in agg]
        jnt_p = [r["joint_psnr"] for r in agg]
        print(f"  delta={delta:.2f}  M={M:3d}  "
              f"seq {np.mean(seq_p):5.2f}±{np.std(seq_p):.2f}   "
              f"joint {np.mean(jnt_p):5.2f}±{np.std(jnt_p):.2f}   "
              f"Δ={np.mean(jnt_p)-np.mean(seq_p):+.2f}   "
              f"({time.time()-t0:.1f}s)")

    # Aggregate
    results = {
        "deltas": deltas,
        "snr_db": snr_db,
        "gradient_factor": float(gradient.max() / gradient.min()),
        "test_scenes": [name for name, _ in test_set],
        "seq_psnr_mean":   [float(np.mean([r["seq_psnr"]   for r in per_delta[d]])) for d in deltas],
        "seq_psnr_std":    [float(np.std ([r["seq_psnr"]   for r in per_delta[d]])) for d in deltas],
        "joint_psnr_mean": [float(np.mean([r["joint_psnr"] for r in per_delta[d]])) for d in deltas],
        "joint_psnr_std":  [float(np.std ([r["joint_psnr"] for r in per_delta[d]])) for d in deltas],
        "seq_ssim_mean":   [float(np.mean([r["seq_ssim"]   for r in per_delta[d]])) for d in deltas],
        "seq_ssim_std":    [float(np.std ([r["seq_ssim"]   for r in per_delta[d]])) for d in deltas],
        "joint_ssim_mean": [float(np.mean([r["joint_ssim"] for r in per_delta[d]])) for d in deltas],
        "joint_ssim_std":  [float(np.std ([r["joint_ssim"] for r in per_delta[d]])) for d in deltas],
        "per_scene": {f"{d:.2f}": per_delta[d] for d in deltas},
    }

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "joint_vs_sequential.json", "w") as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].errorbar(deltas, results["seq_psnr_mean"], yerr=results["seq_psnr_std"],
                     fmt="o-", capsize=3, label="sequential")
    axes[0].errorbar(deltas, results["joint_psnr_mean"], yerr=results["joint_psnr_std"],
                     fmt="s-", capsize=3, label="joint (ours)")
    axes[0].set_xlabel(r"measurement rate $\delta = M/N$")
    axes[0].set_ylabel("PSNR on scene (dB)")
    axes[0].set_title(f"Joint vs sequential, SNR={snr_db} dB, 8x gradient, {len(test_set)} scenes")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].errorbar(deltas, results["seq_ssim_mean"], yerr=results["seq_ssim_std"],
                     fmt="o-", capsize=3, label="sequential")
    axes[1].errorbar(deltas, results["joint_ssim_mean"], yerr=results["joint_ssim_std"],
                     fmt="s-", capsize=3, label="joint (ours)")
    axes[1].set_xlabel(r"measurement rate $\delta = M/N$")
    axes[1].set_ylabel("SSIM on scene")
    axes[1].set_title("Structural similarity (mean ± std across scenes)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.savefig(ROOT / "figures" / "joint_vs_sequential.png", dpi=140)

    # Qualitative grid using the cameraman scene
    n = len(deltas)
    fig2, ax2 = plt.subplots(3, n, figsize=(2.0 * n, 6.0))
    for j, d in enumerate(deltas):
        ax2[0, j].imshow(viz_scene, cmap="gray", vmin=0, vmax=1)
        ax2[0, j].set_title(f"δ={d:.2f}")
        ax2[1, j].imshow(recons_seq_viz[j], cmap="gray", vmin=0, vmax=1)
        ax2[2, j].imshow(recons_joint_viz[j], cmap="gray", vmin=0, vmax=1)
        for r in range(3):
            ax2[r, j].axis("off")
    ax2[0, 0].set_ylabel("scene", rotation=0, labelpad=24)
    ax2[1, 0].set_ylabel("seq.",  rotation=0, labelpad=24)
    ax2[2, 0].set_ylabel("joint", rotation=0, labelpad=24)
    fig2.suptitle(f"Joint vs sequential on '{viz_scene_name}' (scene 1 of {len(test_set)})")
    fig2.tight_layout()
    fig2.savefig(ROOT / "figures" / "joint_vs_sequential_qualitative.png", dpi=140)
    print(f"Saved figures.")


if __name__ == "__main__":
    main()
