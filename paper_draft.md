# Simultaneous Illumination Normalization and Sparse Image Compression via Unfolded Recovery

_Draft — ENGS 109 final project. Sections III–IV are written; Sections I, II, V, VI are stubs filled in once the experiments complete._

---

## I. Abstract

_TBD after Section V is final._

---

## II. Introduction

_TBD — paragraph-level plan in `PROJECT_PLAN.md`._

---

## III. Mathematical Preliminaries

### A. The measurement model

The CS forward model treats image acquisition as a linear measurement:

$$
\boldsymbol{y} = \boldsymbol{A}\boldsymbol{x} + \boldsymbol{w}, \qquad
\boldsymbol{y} \in \mathbb{R}^M,\;
\boldsymbol{A} \in \mathbb{R}^{M \times N},\;
\boldsymbol{x} \in \mathbb{R}^N,\;
\|\boldsymbol{w}\|_2 \le \epsilon,
$$

with $M \ll N$. The sensing matrix $\boldsymbol{A}$ models the physical
encoder (e.g., a coded-aperture mask or a random projection circuit), and
$\boldsymbol{w}$ collects shot, read, and quantization noise.

### B. Sparsity prior

The unknown signal is assumed $S$-sparse in some basis $\boldsymbol{\Psi}$,
i.e. $\boldsymbol{x} = \boldsymbol{\Psi}\boldsymbol{c}$ with
$\boldsymbol{c} \in \Sigma_S \equiv \{\boldsymbol{c} \in \mathbb{R}^N \mid
\|\boldsymbol{c}\|_0 \le S\}$. In this work $\boldsymbol{\Psi}$ is the 2D
DCT — a strong sparsifying basis for natural images.

### C. Convex relaxation

Direct minimization of the $\ell_0$ count is combinatorial. Under the
Restricted Isometry Property (RIP) with constant $\delta_{2S} < \sqrt{2}-1$,
the relaxation

$$
\boldsymbol{c}_1^{\epsilon}
= \underset{\boldsymbol{c}}{\operatorname{arg\,min}}\;
  \|\boldsymbol{c}\|_1 \quad \text{s.t.} \quad
  \|\boldsymbol{A}\boldsymbol{\Psi}\boldsymbol{c} - \boldsymbol{y}\|_2 \le \epsilon
$$

recovers the true $\boldsymbol{c}$ to within $O(\epsilon)$ error (Candès,
Romberg, Tao, 2006). Equivalently, the Lasso form

$$
\boldsymbol{c}_\lambda
= \underset{\boldsymbol{c}}{\operatorname{arg\,min}}\;
  \tfrac{1}{2}\|\boldsymbol{A}\boldsymbol{\Psi}\boldsymbol{c} - \boldsymbol{y}\|_2^2
  + \lambda \|\boldsymbol{c}\|_1
$$

is solved efficiently with proximal-gradient methods.

### D. ISTA and FISTA

ISTA is the proximal gradient method for the Lasso:

$$
\boldsymbol{c}^{(k+1)} =
\mathcal{S}_{\lambda/L}\!\left(
  \boldsymbol{c}^{(k)} - \tfrac{1}{L} (\boldsymbol{A}\boldsymbol{\Psi})^\top
  \!\left( \boldsymbol{A}\boldsymbol{\Psi}\boldsymbol{c}^{(k)} - \boldsymbol{y} \right)
\right),
$$

where $L = \|(\boldsymbol{A}\boldsymbol{\Psi})^\top \boldsymbol{A}\boldsymbol{\Psi}\|_2$
and $\mathcal{S}_\tau(z) = \operatorname{sign}(z)\max(|z|-\tau, 0)$ is the
soft-threshold (the prox operator of $\tau\|\cdot\|_1$). FISTA augments this
with a Nesterov momentum term and improves the objective-gap rate from
$O(1/k)$ to $O(1/k^2)$.

### E. ADMM

For the same objective, ADMM splits $\boldsymbol{c} \to (\boldsymbol{c}, \boldsymbol{z})$
with constraint $\boldsymbol{c} = \boldsymbol{z}$ and alternates a
least-squares update on $\boldsymbol{c}$ (solved once via a cached Cholesky
factorization of $(\boldsymbol{A}\boldsymbol{\Psi})^\top
\boldsymbol{A}\boldsymbol{\Psi} + \rho \boldsymbol{I}$) with a
soft-threshold update on $\boldsymbol{z}$.

---

## IV. Proposed Method: Joint HDR-Compression Architecture

### A. Image-formation model with illumination

In a low-light or HDR scene, the RAW signal is the product of the
underlying reflectance image $\boldsymbol{s}$ and a spatial illumination
field $\boldsymbol{g}$:

$$
\boldsymbol{x} = \boldsymbol{s} \odot \boldsymbol{g}.
$$

The conventional pipeline first recovers $\boldsymbol{x}$ from
$\boldsymbol{y} = \boldsymbol{A}\boldsymbol{x} + \boldsymbol{w}$ and only
afterwards estimates and divides out $\boldsymbol{g}$. This sequential
ordering wastes the prior knowledge that $\boldsymbol{s}$ is sparse in
$\boldsymbol{\Psi}$ but $\boldsymbol{x}$ is *not* — multiplication by an
illumination field destroys DCT-domain sparsity and weakens the
$\ell_1$ recovery guarantees.

### B. Joint objective

We instead formulate a single objective in $(\boldsymbol{c}, \boldsymbol{g})$:

$$
\min_{\boldsymbol{c},\boldsymbol{g}}\;
\tfrac{1}{2} \big\| \boldsymbol{A}\,\mathrm{diag}(\boldsymbol{g})\,
  \boldsymbol{\Psi}\boldsymbol{c} - \boldsymbol{y} \big\|_2^2
+ \lambda_c \|\boldsymbol{c}\|_1
+ \lambda_g \|\nabla \boldsymbol{g}\|_2^2,
$$

where $\nabla$ is a discrete gradient operator that enforces smoothness on
the illumination. The model has a multiplicative bilinearity in
$(\boldsymbol{c}, \boldsymbol{g})$ — non-convex jointly, convex in each
block individually. We minimize by alternating:

- **c-step.** With $\boldsymbol{g}$ fixed, the effective sensing matrix is
  $\boldsymbol{A}_{\text{eff}} = \boldsymbol{A}\,\mathrm{diag}(\boldsymbol{g})\,\boldsymbol{\Psi}$
  and the c-update is a standard Lasso, solved with FISTA.
- **g-step.** With $\boldsymbol{c}$ fixed, the g-update is a ridge regression
  on the side-vector $\boldsymbol{g}$, with closed form
  $\boldsymbol{g}^\star = (\boldsymbol{B}^\top \boldsymbol{B} + \lambda_g \boldsymbol{L})^{-1}
   \boldsymbol{B}^\top \boldsymbol{y}$, where the rows of $\boldsymbol{B}$
  collect the partial sensing of the current scene estimate against each
  column of the image.

### C. Deep unfolding (LISTA)

A practical limitation of ISTA / FISTA is the iteration count — hundreds of
iterations at deployment. We replace the c-step with $K$ unrolled ISTA
layers in which the encoder, recurrent, and threshold parameters are
*learned*:

$$
\boldsymbol{c}^{(k+1)} =
\mathcal{S}_{\theta_k}\!\big(
  \boldsymbol{W}_t \boldsymbol{c}^{(k)} + \boldsymbol{W}_e \boldsymbol{y}
\big),\quad k=0,\dots,K-1.
$$

Initializing $\boldsymbol{W}_e = (1/L)(\boldsymbol{A}\boldsymbol{\Psi})^\top$,
$\boldsymbol{W}_t = \boldsymbol{I} - (1/L)(\boldsymbol{A}\boldsymbol{\Psi})^\top
\boldsymbol{A}\boldsymbol{\Psi}$, and $\theta_k = \lambda/L$ exactly
reproduces ISTA at training-step zero. End-to-end training on
$(\boldsymbol{y}, \boldsymbol{c})$ pairs minimizes the supervised MSE on
$\boldsymbol{c}$ and lets the thresholds adapt to local lighting
statistics — the property that motivates the "learned thresholds" claim in
the project plan.

---

## V. Experiments

_See `RESULTS.md` for the live numbers and figures. To be ported into this
draft once all experiment scripts have completed._

---

## VI. Conclusion & Future Work

_TBD._
