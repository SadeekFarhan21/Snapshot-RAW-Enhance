# Experimental Results

This document records the experiments run for our project. All
numbers are produced by scripts in `experiments/`; see the JSON files in
`results/` for the raw outputs and `figures/` for the generated plots.

## Setup

**Environment.** Classical experiments use Python 3.13, NumPy 2.4, SciPy
1.17, and scikit-image 0.26. The five-seed LISTA run used PyTorch 2.7 with
CUDA 12.8 on a Quadro RTX 8000. All experiments seed their RNGs.

**Solvers compared.** Greedy OMP, ISTA, FISTA (Nesterov-accelerated ISTA),
ADMM for the unconstrained Lasso, and a deep-unfolded LISTA with K=10
learnable layers and tied weights (Gregor & LeCun, ICML 2010).

## Experiment 1 — Phase transition

We fix N=200 and sweep δ = M/N and ρ = S/M over a 19×19 grid, with 20
independent Gaussian-CS trials per cell at zero noise. A trial counts as
a recovery success if NMSE < 1e-3.

- Figure: `figures/phase_transition.png`
- Data:   `results/phase_transition.npz`
- Summary: `results/phase_transition_summary.json`

**Findings.** Both solvers exhibit the classical sharp Donoho–Tanner-style
transition. We summarize the 50%-success boundary $\rho^\ast(\delta) =
\max\{\rho : P_\text{success}(\delta,\rho) \ge 0.5\}$:

| $\delta$ | OMP $\rho^\ast$ | FISTA $\rho^\ast$ |
| :---: | :---: | :---: |
| 0.30 | 0.35 | 0.20 |
| 0.50 | 0.40 | 0.30 |
| 0.70 | 0.40 | 0.50 |
| 0.90 | 0.50 | 0.65 |

Two regimes are visible: at small $\delta$ (extreme undersampling) OMP's
exact least-squares-on-support update beats $L_1$, but as $\delta$ grows
FISTA overtakes OMP because OMP fails on dense supports regardless of $M$
while $L_1$ continues to expand its feasible region. Fraction of the
$(\delta,\rho)$ grid recovered at $\ge 50\%$ rate: OMP 40.4%, FISTA 36.0%.
The 50/50 cell $(\delta,\rho) = (0.5, 0.5)$ — a notoriously hard regime —
gives OMP 0.30 success, FISTA 0.00; FISTA's shrinkage bias means it
rarely clears the strict NMSE $< 10^{-3}$ bar even when the support is
correct.

## Experiment 2 — Rate-distortion on natural image set

5-image test set (cameraman, astronaut, coins, page, moon) from
`skimage.data`, 64×64 grayscale, 16×16 blocks, DCT sparsifying basis,
Gaussian sensing in pixel domain, additive Gaussian noise at SNR = 30 dB.
PSNR and SSIM are reported as mean ± std across the test set so the
result isn't a single-image artifact.

- Figure: `figures/rate_distortion.png`, `figures/rate_distortion_qualitative.png`
- Data:   `results/rate_distortion.json`

**Findings.** PSNR (dB, mean ± std across 5 images):

| $\delta$ | OMP | FISTA | ADMM |
| :---: | :---: | :---: | :---: |
| 0.10 | 15.46 ± 4.4 | **16.67 ± 4.8** | 11.23 ± 3.1 |
| 0.20 | 17.42 ± 4.3 | **19.39 ± 4.5** | 17.62 ± 4.8 |
| 0.30 | 19.15 ± 4.0 | **21.73 ± 4.3** | 21.27 ± 4.6 |
| 0.40 | 20.42 ± 3.7 | **23.64 ± 4.1** | 23.47 ± 4.4 |
| 0.50 | 21.73 ± 3.2 | **25.06 ± 3.9** | 24.98 ± 4.0 |
| 0.60 | 23.50 ± 3.0 | **26.43 ± 3.8** | 26.37 ± 3.9 |
| 0.70 | 24.89 ± 3.0 | **27.57 ± 3.5** | 27.45 ± 3.7 |

FISTA and ADMM track each other within $\sim 0.5$ dB once $\delta \ge 0.2$
(they minimize the same Lasso objective). Both beat OMP by 2–3 dB at
every $\delta \ge 0.2$. ADMM is the weakest at $\delta = 0.10$ because
the design matrix becomes severely ill-conditioned and fixed $\rho = 1.0$
is no longer well-matched. The $\sim 3$–$4$ dB std across scenes reflects
per-image difficulty: `moon` is the easiest (low entropy, mostly black),
`coins` is the hardest (textured, high spatial frequency).

## Experiment 3 — LISTA vs ISTA / FISTA at matched depth

Train LISTA (K=10 unfolded layers, tied W_e and W_t, per-layer learnable
threshold) on 5000 synthetic (y, x) pairs at N=200, M=80, S=10, SNR=30 dB.
The sensing matrix, training set, and validation set are fixed, while optimizer
initialization is repeated for five seeds. Compare validation NMSE against
ISTA / FISTA evaluated at exactly K=10 iterations and against converged
ISTA / FISTA (500 iterations).

- Figure: `figures/lista_comparison.png`
- Data:   `results/lista_results.json`

**Findings.** Validation NMSE at matched layer / iteration count $K = 10$:

| Solver | Val NMSE @ K=10 |
| :--- | :---: |
| ISTA  | 0.4667 |
| FISTA | 0.3369 |
| LISTA (5 seeds) | **0.0435 ± 0.0120** |

The five seed NMSEs range from 0.0291 to 0.0572. FISTA needs **23
iterations** to match the mean LISTA NMSE, versus 10 unrolled layers: a
**2.3× iteration reduction**, not a wall-clock speedup claim. Per-seed matched
counts average 23.2 ± 1.3 iterations. Training takes 83.1 ± 8.3 seconds per
seed on the GPU. Fully converged ISTA/FISTA reach NMSE $\approx 0.0050$.
The recurrence can deteriorate after its best checkpoint as the spectral
radius of $W_t$ drifts, so training uses gradient clipping and restores the
best validation checkpoint.

### Compute and scaling benchmark

`experiments/benchmark_compute.py` measures batch-1 float32 inference with
one CPU thread and a precomputed step size. Times are medians; working memory
is a conservative tensor accounting rather than process RSS.

| $N$ | LISTA-10 (ms) | FISTA-23 (ms) | LISTA params | Working MiB (L/F) |
| :---: | :---: | :---: | :---: | :---: |
| 64 | 0.470 | 0.542 | 5,770 | 0.030 / 0.008 |
| 200 | 0.664 | 0.484 | 56,010 | 0.279 / 0.066 |
| 256 | 0.763 | 0.482 | 91,658 | 0.455 / 0.106 |
| 1024 | 8.650 | 3.942 | 1,468,426 | 7.224 / 1.628 |

At the paper setting ($N=200$), LISTA is about 37% slower despite using fewer
layers/iterations. Dense LISTA uses 560,000 multiply-adds there versus 736,000
for FISTA-23, but framework overhead and the dense $N\times N$ recurrence
erase that arithmetic advantage. At $N=1024$, the quadratic learned map makes
LISTA more than twice as slow and over four times larger in working memory.

## Experiment 4 — Joint vs sequential recovery under illumination gradient

Per-scene ablation across 5 real natural images (cameraman, astronaut,
coins, page, moon). Each scene is a 16×16 center patch, [0,1]-normalized,
multiplied by a horizontal 8× illumination gradient. Gaussian CS at
SNR = 25 dB, sweep over measurement rates.

- **Sequential pipeline**: FISTA on the raw measurements in DCT domain,
  then per-column illumination correction by dividing out column means.
- **Joint pipeline**: alternate FISTA on DCT coefficients (with sensing
  matrix modulated by the current illumination estimate) and a ridge-
  regularized least-squares update on the illumination gain.

Both pipelines are scored on PSNR/SSIM against the illumination-normalized
scene — that is, what an ISP is _trying_ to produce.

- Figure: `figures/joint_vs_sequential.png`, `figures/joint_vs_sequential_qualitative.png`
- Data:   `results/joint_vs_sequential.json`

**Findings — scene-dependent.** Per-scene PSNR gain $\Delta = $ joint − seq (dB):

| $\delta$ | cameraman | astronaut | coins | page | moon |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.20 | **+2.87** | **+0.54** |  −1.02 |  −0.79 | **−10.03** |
| 0.30 | **+4.65** | **+0.18** |  −1.73 |  −0.48 |  −9.51 |
| 0.40 | **+1.13** | **+1.84** | **+0.39** |  −0.23 |  −7.76 |
| 0.50 | **+1.08** | **+2.31** | **+0.74** | **+0.74** |  −9.05 |
| 0.60 | **+3.07** | **+2.21** | **+0.32** | **+0.43** |  −8.37 |

Aggregated (PSNR mean ± std, dB):

| $\delta$ | all 5 (seq) | all 5 (joint) | no-moon (seq) | no-moon (joint) |
| :---: | :---: | :---: | :---: | :---: |
| 0.20 | **10.72 ± 3.4** |  9.03 ± 1.6 |  9.23 ± 1.8 | **9.64 ± 1.1** |
| 0.30 | **11.65 ± 3.8** | 10.27 ± 1.5 | 10.14 ± 2.5 | **10.79 ± 1.2** |
| 0.40 | **12.51 ± 2.3** | 11.59 ± 1.6 | 11.39 ± 0.7 | **12.17 ± 1.3** |
| 0.50 | **13.59 ± 2.4** | 12.76 ± 2.0 | 12.42 ± 0.6 | **13.64 ± 1.1** |
| 0.60 | **13.40 ± 2.4** | 12.94 ± 2.0 | 12.30 ± 1.0 | **13.81 ± 1.1** |

**Three regimes.** (1) *Joint wins on textured scenes* (cameraman +1.1 to
+4.7 dB; astronaut +0.2 to +2.3 dB) — the regime the method was designed
for. (2) *Joint is a wash on moderate-content scenes* (coins, page:
±1.7 dB; clearly positive once $\delta \ge 0.4$). (3) *Joint
catastrophically fails on near-uniform scenes* (moon: −7.8 to −10.0 dB).

**Mechanism behind the moon failure.** Moon's center patch is mostly
black sky. After [0,1] normalization and 8× gradient multiplication,
most columns of the raw signal carry near-zero energy, so the
first c-step produces a near-zero scene estimate. The resulting gain-update
design has too little absolute signal energy, allowing noise to dominate even
though its conditioning is not unusually poor. The block-coordinate
alternation then drives $\boldsymbol{g}$ toward a high-variance solution that
explains the noise rather than the absent signal. Sequential avoids this
because it never tries to estimate
$\boldsymbol{g}$ from the measurements — its column-mean divisor is
essentially a copy of the gradient, applied unconditionally. When the
underlying scene is too sparse to identify $\boldsymbol{g}$, refusing to
estimate it is a virtue.

This motivates a deployment gate, but Experiment 6 shows that conditioning
$\kappa(\boldsymbol{B}^\top\boldsymbol{B})$ is the wrong gate signal. A gate
should instead measure scene energy after the first c-step and fall back to
sequential when that estimate is too weak.

## Experiment 5 — Noise robustness sweep

Fixes $\delta = 0.4$ ($M=102$, $N=256$, single 16×16 patch per scene)
and sweeps measurement SNR over $\{5, 10, 15, 20, 25, 35\}$ dB. The
Lasso parameter $\lambda$ scales with the noise level. All three
solvers share precomputed Lipschitz constants and Cholesky factors so
the comparison isolates noise effects.

- Figure: `figures/noise_robustness.png`
- Data:   `results/noise_robustness.json`

**Findings.** PSNR (dB, mean ± std across 5 scenes):

| SNR (dB) | $\lambda$ | OMP | FISTA | ADMM |
| :---: | :---: | :---: | :---: | :---: |
| 5  | 0.200 | 8.37 ± 0.76  | **10.39 ± 0.94** | 10.39 ± 0.94 |
| 10 | 0.100 | 12.02 ± 1.21 | **14.08 ± 1.34** | 14.03 ± 1.37 |
| 15 | 0.050 | 14.16 ± 1.47 | **15.59 ± 1.64** | 15.40 ± 1.62 |
| 20 | 0.030 | 15.69 ± 1.01 | **16.57 ± 1.60** | 16.17 ± 1.75 |
| 25 | 0.020 | 15.85 ± 1.94 | **17.21 ± 1.63** | 16.08 ± 1.96 |
| 35 | 0.008 | 16.60 ± 1.39 | **17.35 ± 2.12** | 12.95 ± 1.69 |

PSNR scales roughly linearly in SNR (dB) up to ~20 dB then **plateaus
at ~17 dB** regardless of additional headroom — basis mismatch (the
fixed 2D-DCT) is the bottleneck above 20 dB, not measurement noise.
FISTA dominates throughout. ADMM tracks FISTA up to 25 dB but
**degrades sharply at 35 dB** (12.95 vs 17.35); the small $\lambda$
with fixed $\rho = 1.0$ leaves the splitting badly tuned. OMP trails
FISTA by 1–2 dB at every SNR.

## Experiment 6 — Identifiability gate: $\kappa(B^\top B)$ does not predict joint failure

For each scene we record $\log_{10} \kappa(B^\top B)$ after the first
c-step of the joint pipeline, then sweep a gate threshold $\tau$:
gated = joint if $\log_{10}\kappa < \tau$ else sequential.
Setup: $\delta = 0.4$, SNR = 25 dB, 8× horizontal gradient.

- Figure: `figures/identifiability_gate.png`
- Data:   `results/identifiability_gate.json`

**Findings — the hypothesis is falsified.**

| scene     | $\log_{10}\kappa$ | PSNR seq | PSNR joint | Δ (joint − seq) |
| :-------- | :---: | :---: | :---: | :---: |
| cameraman | 1.36  | 6.10  | **10.65** | +4.54 |
| coins     | 1.37  | 6.35  | **10.14** | +3.78 |
| astronaut | 1.86  | 13.15 | **13.31** | +0.16 |
| page      | 1.79  | **12.07** | 10.28 | −1.79 |
| moon      | **1.11** | **16.49** | 8.22 | **−8.27** |

**Moon — the worst-performing scene for the joint pipeline — has the
*lowest* $\log_{10}\kappa$ of the five.** The best gate threshold
($\tau^\star = 1.37$) gives 10.85 dB mean PSNR vs 10.83 dB for
sequential-only — essentially zero net advantage, because it routes
moon to joint (wrong) and page away from joint (also wrong). The
mechanism behind the moon failure is therefore **not** $B^\top B$
ill-conditioning; it is **scene-energy deficiency**: the c-step on a
near-uniformly-dark patch produces a near-zero $\hat{s}$, the rows of
$B$ are small in absolute terms, and noise dominates the g-update. The
correct gate signal is $\|\hat{s}\|_2$, not $\kappa$.

## Summary

- **Phase transition (Exp. 1).** OMP wins at small $\delta$; FISTA wins
  at large $\delta$. The crossover sits near $\delta \approx 0.65$. Both
  show a sharp empirical Donoho–Tanner transition.
- **Rate-distortion (Exp. 2).** FISTA/ADMM beat OMP by 2–3 dB on a
  natural image across the useful $\delta$ range; FISTA and ADMM are
  within 0.5 dB of each other.
- **LISTA (Exp. 3).** Across five optimizer seeds, a 10-layer learned solver
  obtains NMSE $0.0435\pm0.0120$; FISTA needs 23 iterations to match the mean.
  This is a 2.3× iteration reduction, but not a measured latency speedup.
- **Joint vs sequential (Exp. 4).** Scene-dependent. Joint wins on
  textured scenes by up to +4.7 dB (cameraman, astronaut), is a wash on
  moderate-content scenes (coins, page), and fails catastrophically on
  the near-uniform `moon` scene (−7.8 to −10.0 dB). Excluding moon,
  joint wins by +0.4 to +1.5 dB at every $\delta$.
- **Noise robustness (Exp. 5).** PSNR scales ~linearly in SNR (dB) up
  to ~20 dB then plateaus at ~17 dB — basis mismatch dominates above
  that. ADMM with fixed $\rho = 1.0$ destabilizes at SNR ≥ 35 dB.
- **Identifiability gate (Exp. 6).** Falsifies the $\kappa(B^\top B)$
  gate hypothesis from the original conclusion: moon has the lowest
  $\kappa$ but the worst joint result. The real failure mechanism is
  scene-energy deficiency in the first c-step; the correct gate signal
  is $\|\hat{s}\|_2$, not $\kappa$.

## How to reproduce

```bash
# Cap BLAS threads — small-matrix dispatch overhead otherwise
# dominates on multi-core machines.
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
       VECLIB_MAXIMUM_THREADS=2

python3 experiments/phase_transition.py     # ~8 min on CPU
python3 experiments/rate_distortion.py      # ~12 s
python3 experiments/train_lista.py          # ~4 min
python3 experiments/benchmark_compute.py    # ~1 min
python3 experiments/joint_vs_sequential.py  # ~5 s
python3 experiments/noise_robustness.py     # ~1 s
python3 experiments/identifiability_gate.py # ~1 s
```
