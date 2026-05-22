"""Phase transition diagram for OMP / FISTA on Gaussian CS.

For a grid of (delta = M/N, rho = S/M) values we run T independent trials,
counting a trial as a recovery success if NMSE < 1e-3. The plotted heatmap
is the empirical success rate. The classical Donoho-Tanner curve separating
"always recoverable" from "always fails" should appear as a sharp diagonal
transition.
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

from src.measurement import gaussian_sensing_matrix
from src.solvers import omp, fista
from src.data import sparse_signal
from src.metrics import nmse


def run_phase_transition(
    N: int = 200,
    deltas: np.ndarray | None = None,
    rhos: np.ndarray | None = None,
    n_trials: int = 20,
    n_fista_iters: int = 400,
    fista_lam: float = 1e-3,
    success_threshold: float = 1e-3,
    seed: int = 0,
):
    if deltas is None:
        deltas = np.linspace(0.05, 0.95, 19)  # M/N
    if rhos is None:
        rhos = np.linspace(0.05, 0.95, 19)    # S/M

    success_omp = np.zeros((len(rhos), len(deltas)))
    success_fista = np.zeros((len(rhos), len(deltas)))

    base_rng = np.random.default_rng(seed)
    t0 = time.time()
    for j, delta in enumerate(deltas):
        M = max(1, int(round(delta * N)))
        for i, rho in enumerate(rhos):
            S = max(1, int(round(rho * M)))
            if S > M or S > N:
                continue
            wins_omp = 0
            wins_fista = 0
            for _ in range(n_trials):
                rng = np.random.default_rng(base_rng.integers(0, 2**31 - 1))
                A = gaussian_sensing_matrix(M, N, rng)
                x = sparse_signal(N, S, rng)
                y = A @ x  # noiseless — phase transition is a noiseless concept
                x_omp = omp(A, y, S)
                x_fista, _ = fista(A, y, lam=fista_lam, n_iters=n_fista_iters)
                if nmse(x_omp, x) < success_threshold:
                    wins_omp += 1
                if nmse(x_fista, x) < success_threshold:
                    wins_fista += 1
            success_omp[i, j] = wins_omp / n_trials
            success_fista[i, j] = wins_fista / n_trials
        print(f"  delta={delta:.2f}  elapsed={time.time()-t0:.1f}s")
    return deltas, rhos, success_omp, success_fista


def main():
    print("Running phase transition (N=200, 19x19 grid, 20 trials each)...")
    deltas, rhos, p_omp, p_fista = run_phase_transition()

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    np.savez(out_dir / "phase_transition.npz",
             deltas=deltas, rhos=rhos, p_omp=p_omp, p_fista=p_fista)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, data, title in zip(axes, [p_omp, p_fista], ["OMP", "FISTA (L1)"]):
        im = ax.imshow(
            data,
            origin="lower",
            extent=(deltas[0], deltas[-1], rhos[0], rhos[-1]),
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_xlabel(r"$\delta = M/N$  (undersampling rate)")
        ax.set_ylabel(r"$\rho = S/M$  (sparsity rate)")
        ax.set_title(f"{title}  —  P(NMSE < 1e-3)")
        fig.colorbar(im, ax=ax, label="success rate")
    fig.suptitle(f"Phase transition diagrams  (N={200}, 20 trials/cell)")

    fig_path = ROOT / "figures" / "phase_transition.png"
    fig_path.parent.mkdir(exist_ok=True)
    fig.savefig(fig_path, dpi=140)
    print(f"Saved {fig_path}")

    # console summary: success-rate at a couple of representative points
    summary = {
        "deltas": deltas.tolist(),
        "rhos": rhos.tolist(),
        "p_omp_50_50": float(p_omp[len(rhos)//2, len(deltas)//2]),
        "p_fista_50_50": float(p_fista[len(rhos)//2, len(deltas)//2]),
        "p_omp_max": float(p_omp.max()),
        "p_fista_max": float(p_fista.max()),
    }
    with open(out_dir / "phase_transition_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
