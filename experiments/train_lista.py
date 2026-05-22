"""Train LISTA and compare against ISTA / FISTA at matched layer / iteration counts.

The whole point of deep unfolding is "fewer iterations for the same recovery
quality." So we fix K layers for LISTA and compare against K ISTA iterations
and K FISTA iterations on the held-out validation set.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import json

import numpy as np
import matplotlib.pyplot as plt
import torch

from src.measurement import gaussian_sensing_matrix, add_measurement_noise
from src.solvers import ista, fista
from src.data import sparse_signal
from src.lista import train_lista, LISTA
from src.metrics import nmse


def build_dataset(A: np.ndarray, n_examples: int, sparsity: int, snr_db: float,
                  rng: np.random.Generator):
    M, N = A.shape
    Y = np.zeros((n_examples, M))
    X = np.zeros((n_examples, N))
    for i in range(n_examples):
        x = sparse_signal(N, sparsity, rng)
        y_clean = A @ x
        y_noisy, _ = add_measurement_noise(y_clean, snr_db=snr_db, rng=rng)
        X[i] = x
        Y[i] = y_noisy
    return Y, X


def ista_fista_nmse_vs_iters(A, Y_va, X_va, lam, max_iters):
    """Compute val NMSE of ISTA / FISTA at every iteration from 1..max_iters.

    We do this for a sample of validation problems to keep runtime small.
    """
    sample = min(64, Y_va.shape[0])
    Y = Y_va[:sample]
    X = X_va[:sample]
    ista_curve = np.zeros(max_iters)
    fista_curve = np.zeros(max_iters)
    for i in range(sample):
        _, h_i = ista(A, Y[i], lam=lam, n_iters=max_iters, x_true=X[i])
        _, h_f = fista(A, Y[i], lam=lam, n_iters=max_iters, x_true=X[i])
        ista_curve += np.array(h_i.nmse)
        fista_curve += np.array(h_f.nmse)
    return ista_curve / sample, fista_curve / sample


def main():
    rng = np.random.default_rng(0)
    N, M, S = 200, 80, 10
    snr_db = 30.0
    n_layers = 10

    A_np = gaussian_sensing_matrix(M, N, rng)
    print(f"Building dataset: N={N}, M={M}, S={S}, SNR={snr_db} dB")
    Y_tr, X_tr = build_dataset(A_np, n_examples=5000, sparsity=S, snr_db=snr_db, rng=rng)
    Y_va, X_va = build_dataset(A_np, n_examples=500, sparsity=S, snr_db=snr_db, rng=rng)

    print(f"Training LISTA ({n_layers} unfolded layers)...")
    model, history = train_lista(
        A_np, (Y_tr, X_tr), (Y_va, X_va),
        n_layers=n_layers, n_epochs=80, batch_size=64, lr=5e-3, device="cpu",
    )

    # Final LISTA NMSE on val
    model.eval()
    with torch.no_grad():
        X_hat_lista = model(torch.tensor(Y_va, dtype=torch.float32)).numpy()
    lista_nmses = []
    for i in range(X_va.shape[0]):
        lista_nmses.append(nmse(X_hat_lista[i], X_va[i]))
    lista_val_nmse = float(np.mean(lista_nmses))
    print(f"LISTA val NMSE (K={n_layers} layers): {lista_val_nmse:.5f}")

    print(f"Evaluating ISTA / FISTA over 1..{n_layers} iterations...")
    ista_curve, fista_curve = ista_fista_nmse_vs_iters(
        A_np, Y_va, X_va, lam=0.05, max_iters=n_layers,
    )
    print(f"ISTA  val NMSE @ {n_layers} iters: {ista_curve[-1]:.5f}")
    print(f"FISTA val NMSE @ {n_layers} iters: {fista_curve[-1]:.5f}")

    # Run ISTA/FISTA to convergence for a "fully-converged baseline"
    converged_iters = 500
    ista_long, fista_long = ista_fista_nmse_vs_iters(
        A_np, Y_va, X_va, lam=0.05, max_iters=converged_iters,
    )

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    summary = {
        "N": N, "M": M, "S": S, "snr_db": snr_db, "n_layers": n_layers,
        "lista_val_nmse": lista_val_nmse,
        "ista_at_K": float(ista_curve[-1]),
        "fista_at_K": float(fista_curve[-1]),
        "ista_converged": float(ista_long[-1]),
        "fista_converged": float(fista_long[-1]),
        "speedup_vs_fista_iters_for_matched_nmse": None,
    }
    # If FISTA ever reaches LISTA's NMSE, find how many iters it needed.
    reached = np.where(fista_long <= lista_val_nmse)[0]
    if len(reached) > 0:
        summary["speedup_vs_fista_iters_for_matched_nmse"] = int(reached[0] + 1)
    with open(out_dir / "lista_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # Plot training and comparison curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].plot(history["train_loss"], label="train MSE")
    axes[0].plot(history["val_nmse"], label="val NMSE")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch")
    axes[0].set_title("LISTA training")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    iters_axis = np.arange(1, n_layers + 1)
    axes[1].plot(iters_axis, ista_curve, "o-", label="ISTA")
    axes[1].plot(iters_axis, fista_curve, "s-", label="FISTA")
    axes[1].axhline(lista_val_nmse, ls="--", color="C2", label=f"LISTA (K={n_layers})")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("iterations / layers")
    axes[1].set_ylabel("validation NMSE")
    axes[1].set_title("LISTA vs ISTA / FISTA at matched compute")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.savefig(ROOT / "figures" / "lista_comparison.png", dpi=140)
    print(f"Saved {ROOT/'figures'/'lista_comparison.png'}")


if __name__ == "__main__":
    main()
