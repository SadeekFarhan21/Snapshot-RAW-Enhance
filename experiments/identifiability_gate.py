"""Experiment 6 — Identifiability gate for Joint-CS (fast).

The 'moon' scene in Experiment 4 broke the joint pipeline because the
B^T B matrix in the g-update is rank-deficient (the scene has near-zero
energy in most columns, so the per-column gain is non-identifiable).
The paper's conclusion proposes gating on kappa(B^T B) and falling back
to the sequential pipeline when the condition number is too large.
This experiment actually measures kappa per scene and tests the gate.

Setup mirrors joint_vs_sequential.py — 5 scenes, 16x16 center patches,
8x horizontal illumination gradient, SNR=25 dB, delta=0.4 — and for
each scene records kappa(B^T B) after the first c-step, plus PSNR of
sequential, joint, and "gated" reconstructions.

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
from src.solvers import soft_threshold
from src.data import test_image_set, dct_basis_inverse
from src.metrics import psnr


def build_psi_t(N: int) -> np.ndarray:
    eye = np.eye(N)
    Psi_T = np.zeros((N, N))
    for k in range(N):
        Psi_T[:, k] = dct_basis_inverse(eye[:, k])
    return Psi_T


def fista_fast(A: np.ndarray, y: np.ndarray, lam: float, n_iters: int,
               L: float | None = None) -> np.ndarray:
    """FISTA with optional pre-computed Lipschitz constant L. No history."""
    if L is None:
        # 20 power iterations is enough for our well-conditioned matrices
        v = np.random.default_rng(0).standard_normal(A.shape[1])
        v /= np.linalg.norm(v)
        for _ in range(20):
            v = A.T @ (A @ v)
            nrm = np.linalg.norm(v)
            if nrm == 0:
                L = 0.0; break
            v /= nrm
        L = float(np.linalg.norm(A.T @ (A @ v)))
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


def gain_match(recon: np.ndarray, target: np.ndarray) -> np.ndarray:
    num = float(np.sum(recon * target))
    den = float(np.sum(recon * recon)) + 1e-12
    a = max(num / den, 0.0)
    return np.clip(a * recon, 0.0, 1.0)


def synthesize_block(c, Psi_T, side):
    return (Psi_T @ c).reshape(side, side)


def sequential_recover(A_pix, y, Psi_T, lam, n_iters, side):
    A_dct = A_pix @ Psi_T
    c_hat = fista_fast(A_dct, y, lam=lam, n_iters=n_iters)
    raw_hat = synthesize_block(c_hat, Psi_T, side)
    col_means = raw_hat.mean(axis=0)
    g_hat = np.maximum(col_means / col_means.mean(), 1e-3)
    scene_hat = raw_hat / g_hat[None, :]
    scene_hat = scene_hat / scene_hat.mean() * 0.5
    return np.clip(scene_hat, 0.0, 1.0)


def joint_recover_with_kappa(A_pix, y, Psi_T, side,
                              lam_c=0.02, n_outer=6, n_inner=80):
    """Joint recovery — returns (scene, kappa) where kappa is computed
    after the first c-step (what a deployed gate would have access to)."""
    N = side * side
    g = np.ones(side)
    c = np.zeros(N)
    scene = synthesize_block(c, Psi_T, side)
    kappa_recorded = None
    for k in range(n_outer):
        g_pixel = np.tile(g[None, :], (side, 1)).reshape(-1)
        A_eff = (A_pix * g_pixel[None, :]) @ Psi_T
        c = fista_fast(A_eff, y, lam=lam_c, n_iters=n_inner)
        scene = synthesize_block(c, Psi_T, side)
        A_resh = A_pix.reshape(A_pix.shape[0], side, side)
        B = (A_resh * scene[None, :, :]).sum(axis=1)
        BtB = B.T @ B
        if k == 0:
            sv = np.linalg.svd(BtB, compute_uv=False)
            kappa_recorded = float(sv[0] / max(sv[-1], 1e-12))
        gtg = BtB + 1e-2 * np.eye(side)
        g = np.linalg.solve(gtg, B.T @ y)
        g = np.maximum(g, 1e-3)
    scene_final = scene / max(scene.mean(), 1e-8) * 0.5
    return np.clip(scene_final, 0.0, 1.0), kappa_recorded


def extract_patch(img, side=16):
    H, W = img.shape
    i0 = (H - side) // 2; j0 = (W - side) // 2
    patch = img[i0:i0+side, j0:j0+side].copy()
    return (patch - patch.min()) / (patch.max() - patch.min() + 1e-8)


def main():
    side = 16
    N = side * side
    snr_db = 25.0
    delta = 0.4
    M = int(round(delta * N))
    gradient = np.linspace(1.0, 8.0, side)

    test_set = test_image_set(size=64)
    print(f"[gate] scenes={[n for n, _ in test_set]}  delta={delta}  M={M}/{N}",
          flush=True)
    Psi_T = build_psi_t(N)

    rng = np.random.default_rng(2)
    rows: list[dict] = []
    t0 = time.time()
    for name, img in test_set:
        scene = extract_patch(img, side=side)
        raw_truth = scene * gradient[None, :]
        rng_local = np.random.default_rng(rng.integers(0, 2**31-1))
        A_pix = gaussian_sensing_matrix(M, N, rng_local)
        y_clean = A_pix @ raw_truth.reshape(-1)
        y_noisy, _ = add_measurement_noise(y_clean, snr_db=snr_db, rng=rng_local)
        t1 = time.time()
        seq = sequential_recover(A_pix, y_noisy, Psi_T, lam=0.02,
                                  n_iters=300, side=side)
        t2 = time.time()
        joint, kappa = joint_recover_with_kappa(A_pix, y_noisy, Psi_T, side=side)
        t3 = time.time()
        seq = gain_match(seq, scene)
        joint = gain_match(joint, scene)
        d = {
            "scene": name,
            "psnr_seq":   psnr(seq, scene),
            "psnr_joint": psnr(joint, scene),
            "log10_kappa": float(np.log10(kappa)),
        }
        rows.append(d)
        print(f"  {name:10s}  log10(kappa)={d['log10_kappa']:6.2f}   "
              f"seq {d['psnr_seq']:5.2f}   joint {d['psnr_joint']:5.2f}   "
              f"Δ={d['psnr_joint']-d['psnr_seq']:+5.2f}   "
              f"(seq {t2-t1:.1f}s, joint {t3-t2:.1f}s, total {time.time()-t0:.1f}s)",
              flush=True)

    log_kappas = np.array([r["log10_kappa"] for r in rows])
    seq_ps = np.array([r["psnr_seq"] for r in rows])
    joint_ps = np.array([r["psnr_joint"] for r in rows])
    tau_grid = np.linspace(log_kappas.min() - 0.2, log_kappas.max() + 0.2, 41)
    mean_psnr_gated = np.array([
        float(np.mean(np.where(log_kappas < tau, joint_ps, seq_ps)))
        for tau in tau_grid
    ])
    best_idx = int(np.argmax(mean_psnr_gated))
    tau_star = float(tau_grid[best_idx])
    best_mean = float(mean_psnr_gated[best_idx])
    gated_star = np.where(log_kappas < tau_star, joint_ps, seq_ps)

    print(f"\n[gate] tau* = {tau_star:.3f}   mean PSNR(gated) = {best_mean:.2f} dB",
          flush=True)
    print(f"       (seq alone {seq_ps.mean():.2f}, joint alone {joint_ps.mean():.2f})",
          flush=True)
    for name, lk, ps, pj, pg in zip(
        [r["scene"] for r in rows], log_kappas, seq_ps, joint_ps, gated_star
    ):
        decision = "JOINT" if lk < tau_star else "SEQ"
        print(f"       {name:10s}  log10(k)={lk:5.2f}  seq {ps:5.2f}  joint {pj:5.2f}  "
              f"→ {decision:5s} → gated={pg:5.2f}", flush=True)

    summary = {
        "delta": delta, "M": M, "N": N, "snr_db": snr_db,
        "gradient_factor": float(gradient.max() / gradient.min()),
        "rows": rows,
        "tau_grid": tau_grid.tolist(),
        "mean_psnr_gated": mean_psnr_gated.tolist(),
        "tau_star": tau_star,
        "mean_psnr_seq":   float(seq_ps.mean()),
        "mean_psnr_joint": float(joint_ps.mean()),
        "mean_psnr_gated_star": best_mean,
    }
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "identifiability_gate.json", "w") as f:
        json.dump(summary, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    axes[0].plot(tau_grid, mean_psnr_gated, "-", color="C2", label="gated")
    axes[0].axhline(seq_ps.mean(),   color="C0", ls="--", label="sequential only")
    axes[0].axhline(joint_ps.mean(), color="C1", ls="--", label="joint only")
    axes[0].axvline(tau_star, color="k", ls=":", alpha=0.6,
                    label=fr"$\tau^* = {tau_star:.2f}$")
    axes[0].set_xlabel(r"gate threshold $\tau$ on $\log_{10}\kappa$")
    axes[0].set_ylabel("mean PSNR (dB) across scenes")
    axes[0].set_title(rf"Gate sweep, $\delta={delta}$, SNR={snr_db:.0f} dB")
    axes[0].grid(True, alpha=0.3); axes[0].legend(loc="best", fontsize=9)
    for r in rows:
        c = "C2" if r["log10_kappa"] < tau_star else "C0"
        axes[1].scatter(r["log10_kappa"], r["psnr_joint"]-r["psnr_seq"],
                        s=80, c=c, edgecolor="k")
        axes[1].annotate(r["scene"], (r["log10_kappa"], r["psnr_joint"]-r["psnr_seq"]),
                         xytext=(5, 5), textcoords="offset points", fontsize=9)
    axes[1].axhline(0, color="k", ls="-", alpha=0.3)
    axes[1].axvline(tau_star, color="k", ls=":", alpha=0.6,
                    label=fr"$\tau^*={tau_star:.2f}$")
    axes[1].set_xlabel(r"$\log_{10}\kappa(B^\top B)$ after first c-step")
    axes[1].set_ylabel("PSNR(joint) − PSNR(seq) (dB)")
    axes[1].set_title("Per-scene identifiability vs joint−seq gain")
    axes[1].grid(True, alpha=0.3); axes[1].legend(fontsize=9)
    fig.savefig(ROOT / "figures" / "identifiability_gate.png", dpi=140)
    print(f"\n[gate] saved figures/identifiability_gate.png", flush=True)


if __name__ == "__main__":
    main()
