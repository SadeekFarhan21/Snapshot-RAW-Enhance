"""Measure inference cost and state scaling for LISTA and matched-quality FISTA.

Latency uses batch size one, float32, precomputed operator-specific step sizes,
one CPU thread, and median wall time after warmup.  Memory is the exact byte
count of persistent method state plus a conservative working-tensor count; it
does not include the shared Python runtime.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from src.lista import LISTA
from src.measurement import gaussian_sensing_matrix
from src.solvers import soft_threshold


def fista_precomputed(A: np.ndarray, y: np.ndarray, step: float, lam: float,
                      n_iters: int) -> np.ndarray:
    """FISTA inference with the fixed-operator step size precomputed."""
    x = np.zeros(A.shape[1], dtype=np.float32)
    z = x.copy()
    t = 1.0
    for _ in range(n_iters):
        grad = A.T @ (A @ z - y)
        x_new = soft_threshold(z - step * grad, step * lam).astype(np.float32)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = x_new + np.float32((t - 1.0) / t_new) * (x_new - x)
        x, t = x_new, t_new
    return x


def median_latency_ms(fn, warmup: int, repeats: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - started) / 1e6)
    return float(np.median(samples)), float(np.percentile(samples, 90))


def main() -> None:
    torch.set_num_threads(1)
    rng = np.random.default_rng(2026)
    sizes = [64, 200, 256, 1024]
    delta = 0.4
    n_layers = 10
    fista_iters = 23
    rows = []

    with threadpool_limits(limits=1):
        for N in sizes:
            M = int(round(delta * N))
            A = gaussian_sensing_matrix(M, N, rng).astype(np.float32)
            y = rng.standard_normal(M).astype(np.float32)
            # Setup cost is excluded for both solvers.  The step is computed
            # once per fixed operator, just as LISTA weights are trained once.
            spectral_sq = float(np.linalg.norm(A, ord=2) ** 2)
            step = 1.0 / max(spectral_sq, 1e-8)
            model = LISTA(torch.from_numpy(A), n_layers=n_layers, step=step)
            model.eval()
            y_t = torch.from_numpy(y[None, :])

            def run_lista():
                with torch.inference_mode():
                    model(y_t)

            def run_fista():
                fista_precomputed(A, y, step=step, lam=0.05,
                                  n_iters=fista_iters)

            repeats = 200 if N <= 256 else 40
            lista_ms, lista_p90 = median_latency_ms(run_lista, 10, repeats)
            fista_ms, fista_p90 = median_latency_ms(run_fista, 5, repeats)

            bytes_per = np.dtype(np.float32).itemsize
            lista_params = int(sum(p.numel() for p in model.parameters()))
            lista_state_bytes = int(
                sum(p.numel() * p.element_size() for p in model.parameters())
                + sum(b.numel() * b.element_size() for b in model.buffers())
            )
            # Peak live working tensors are bounded by state plus x, the two
            # matrix-vector outputs, their sum, and threshold output.
            lista_working_bytes = lista_state_bytes + 5 * N * bytes_per + M * bytes_per
            # FISTA stores A, y, x, z, x_new, gradient, residual, and a temp.
            fista_working_bytes = A.nbytes + (6 * N + 2 * M) * bytes_per
            row = {
                "N": N, "M": M, "delta": delta,
                "lista_layers": n_layers, "fista_iterations": fista_iters,
                "lista_parameters": lista_params,
                "lista_latency_ms_median": lista_ms,
                "lista_latency_ms_p90": lista_p90,
                "fista_latency_ms_median": fista_ms,
                "fista_latency_ms_p90": fista_p90,
                "measured_latency_ratio_fista_over_lista": fista_ms / lista_ms,
                "lista_state_mib": lista_state_bytes / 2**20,
                "lista_peak_working_mib": lista_working_bytes / 2**20,
                "fista_peak_working_mib": fista_working_bytes / 2**20,
                "lista_multiply_adds": n_layers * (N * N + N * M),
                "fista_multiply_adds": fista_iters * (2 * M * N),
            }
            rows.append(row)
            print(
                f"N={N:4d} M={M:4d}  LISTA={lista_ms:7.3f} ms  "
                f"FISTA={fista_ms:7.3f} ms  ratio={fista_ms/lista_ms:5.2f}x  "
                f"params={lista_params:,}", flush=True,
            )

    summary = {
        "device": "CPU",
        "dtype": "float32",
        "batch_size": 1,
        "threads": 1,
        "timing": "median wall time after warmup; setup excluded",
        "memory": "exact persistent state plus conservative live working tensors",
        "rows": rows,
    }
    out = ROOT / "results" / "compute_benchmark.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")

    Ns = np.array(sizes)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    axes[0].plot(Ns, [r["lista_latency_ms_median"] for r in rows], "o-", label="LISTA, 10 layers")
    axes[0].plot(Ns, [r["fista_latency_ms_median"] for r in rows], "s-", label="FISTA, 23 iterations")
    axes[0].set_xscale("log", base=2); axes[0].set_yscale("log")
    axes[0].set_xlabel("coefficient dimension N")
    axes[0].set_ylabel("batch-1 latency (ms)")
    axes[0].grid(True, which="both", alpha=0.3); axes[0].legend(fontsize=8)
    axes[1].plot(Ns, [r["lista_peak_working_mib"] for r in rows], "o-", label="LISTA")
    axes[1].plot(Ns, [r["fista_peak_working_mib"] for r in rows], "s-", label="FISTA")
    axes[1].set_xscale("log", base=2); axes[1].set_yscale("log")
    axes[1].set_xlabel("coefficient dimension N")
    axes[1].set_ylabel("method working state (MiB)")
    axes[1].grid(True, which="both", alpha=0.3); axes[1].legend(fontsize=8)
    fig.savefig(ROOT / "figures" / "compute_scaling.png", dpi=220)
    print(f"Saved {out} and figures/compute_scaling.png")


if __name__ == "__main__":
    main()
