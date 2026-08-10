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
import time

import numpy as np
import matplotlib.pyplot as plt
import torch

from src.measurement import gaussian_sensing_matrix, add_measurement_noise
from src.solvers import ista, fista
from src.lista import train_lista
from src.metrics import nmse


def sparse_signal(N: int, S: int, rng: np.random.Generator) -> np.ndarray:
    """Generate an S-sparse Gaussian vector without importing image tooling."""
    x = np.zeros(N)
    support = rng.choice(N, size=S, replace=False)
    x[support] = rng.standard_normal(S)
    return x


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


def plot_results(seed_rows, ista_long, fista_long, n_layers):
    """Plot cross-seed variability and the matched-quality crossing."""
    lista_values = np.array([row["lista_val_nmse"] for row in seed_rows])
    lista_mean = float(lista_values.mean())
    lista_std = float(lista_values.std(ddof=1))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    seeds = np.array([row["seed"] for row in seed_rows])
    axes[0].scatter(seeds, lista_values, s=62, color="C0", zorder=3,
                    label="optimizer seed")
    axes[0].axhspan(lista_mean - lista_std, lista_mean + lista_std,
                    color="C2", alpha=0.18, label="mean ± std")
    axes[0].axhline(lista_mean, ls="--", color="C2")
    axes[0].set_xticks(seeds)
    axes[0].set_xlabel("optimizer seed")
    axes[0].set_ylabel("validation NMSE")
    axes[0].set_title("LISTA variability across five seeds")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    max_plot_iters = 30
    iters_axis = np.arange(1, max_plot_iters + 1)
    axes[1].plot(iters_axis, ista_long[:max_plot_iters], "o-", markevery=3,
                 label="ISTA")
    axes[1].plot(iters_axis, fista_long[:max_plot_iters], "s-", markevery=3,
                 label="FISTA")
    axes[1].axhspan(lista_mean - lista_std, lista_mean + lista_std,
                    color="C2", alpha=0.18)
    axes[1].axhline(lista_mean, ls="--", color="C2",
                    label=f"LISTA mean±std (K={n_layers})")
    reached = np.where(fista_long <= lista_mean)[0]
    if len(reached):
        matched = int(reached[0] + 1)
        axes[1].axvline(matched, ls=":", color="0.3",
                        label=f"FISTA matches at {matched}")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("iterations")
    axes[1].set_ylabel("validation NMSE")
    axes[1].set_title("Classical convergence vs LISTA-10")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.savefig(ROOT / "figures" / "lista_comparison.png", dpi=220)
    print(f"Saved {ROOT/'figures'/'lista_comparison.png'}")


def main():
    dataset_seed = 0
    train_seeds = [0, 1, 2, 3, 4]
    rng = np.random.default_rng(dataset_seed)
    N, M, S = 200, 80, 10
    snr_db = 30.0
    n_layers = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"

    A_np = gaussian_sensing_matrix(M, N, rng)
    print(f"Building dataset: N={N}, M={M}, S={S}, SNR={snr_db} dB")
    Y_tr, X_tr = build_dataset(A_np, n_examples=5000, sparsity=S, snr_db=snr_db, rng=rng)
    Y_va, X_va = build_dataset(A_np, n_examples=500, sparsity=S, snr_db=snr_db, rng=rng)

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

    if "--plot-only" in sys.argv:
        with open(ROOT / "results" / "lista_results.json") as f:
            saved = json.load(f)
        plot_results(saved["per_seed"], ista_long, fista_long, n_layers)
        return

    seed_rows = []
    histories = []
    representative_model = None
    print(
        f"Training LISTA ({n_layers} unfolded layers) for seeds {train_seeds} "
        f"on {device}..."
    )
    for train_seed in train_seeds:
        # A, training examples, and validation examples stay fixed.  Only the
        # minibatch permutation changes, isolating optimizer/training variance.
        torch.manual_seed(train_seed)
        if device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        model, history = train_lista(
            A_np, (Y_tr, X_tr), (Y_va, X_va),
            n_layers=n_layers, n_epochs=80, batch_size=64, lr=5e-3,
            device=device, verbose=False,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        train_seconds = time.perf_counter() - started
        model.eval()
        with torch.no_grad():
            X_hat_lista = model(torch.tensor(
                Y_va, dtype=torch.float32, device=device
            )).cpu().numpy()
        lista_val_nmse = float(np.mean([
            nmse(X_hat_lista[i], X_va[i]) for i in range(X_va.shape[0])
        ]))
        reached = np.where(fista_long <= lista_val_nmse)[0]
        matched_iters = int(reached[0] + 1) if len(reached) > 0 else None
        row = {
            "seed": train_seed,
            "lista_val_nmse": lista_val_nmse,
            "best_epoch": int(history["best_epoch"]),
            "train_seconds": float(train_seconds),
            "fista_iters_for_matched_nmse": matched_iters,
        }
        seed_rows.append(row)
        histories.append(history)
        if train_seed == train_seeds[0]:
            representative_model = model
        print(
            f"  seed={train_seed}  NMSE={lista_val_nmse:.5f}  "
            f"best_epoch={history['best_epoch']:2d}  "
            f"FISTA_match={matched_iters}  train={train_seconds:.1f}s",
            flush=True,
        )

    lista_values = np.array([row["lista_val_nmse"] for row in seed_rows])
    matched_values = np.array([
        row["fista_iters_for_matched_nmse"] for row in seed_rows
        if row["fista_iters_for_matched_nmse"] is not None
    ])
    lista_mean = float(lista_values.mean())
    lista_std = float(lista_values.std(ddof=1))
    reached_mean = np.where(fista_long <= lista_mean)[0]
    matched_at_mean = int(reached_mean[0] + 1) if len(reached_mean) > 0 else None
    parameter_count = int(sum(p.numel() for p in representative_model.parameters()))

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    summary = {
        "dataset_seed": dataset_seed, "train_seeds": train_seeds,
        "training_device": device,
        "N": N, "M": M, "S": S, "snr_db": snr_db,
        "n_layers": n_layers,
        "lista_val_nmse": lista_mean,
        "lista_val_nmse_mean": lista_mean,
        "lista_val_nmse_std": lista_std,
        "lista_val_nmse_min": float(lista_values.min()),
        "lista_val_nmse_max": float(lista_values.max()),
        "per_seed": seed_rows,
        "parameter_count": parameter_count,
        "ista_at_K": float(ista_curve[-1]),
        "fista_at_K": float(fista_curve[-1]),
        "ista_converged": float(ista_long[-1]),
        "fista_converged": float(fista_long[-1]),
        "fista_iters_for_matched_mean_nmse": matched_at_mean,
        "fista_matched_iters_mean": float(matched_values.mean()),
        "fista_matched_iters_std": float(matched_values.std(ddof=1)),
        "iteration_reduction_at_mean": (
            float(matched_at_mean / n_layers) if matched_at_mean is not None else None
        ),
        "train_seconds_mean": float(np.mean([r["train_seconds"] for r in seed_rows])),
        "train_seconds_std": float(np.std(
            [r["train_seconds"] for r in seed_rows], ddof=1
        )),
        # Backward-compatible key used by the original report.
        "speedup_vs_fista_iters_for_matched_nmse": matched_at_mean,
    }
    with open(out_dir / "lista_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    plot_results(seed_rows, ista_long, fista_long, n_layers)


if __name__ == "__main__":
    main()
