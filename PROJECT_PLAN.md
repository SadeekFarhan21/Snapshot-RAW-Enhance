# Joint-CS: Simultaneous Illumination Normalization and Sparse Image Compression via Unfolded Recovery

**Course:** ENGS 109 Final Project
**Format:** IEEE/CVPR-style computer vision & signal processing paper

---

## I. Abstract (1 Paragraph)

- **The Hook:** State the inefficiency of the traditional camera pipeline (Shannon-Nyquist sampling followed by heavy JPEG compression and sequential image signal processing).
- **The Problem:** Note that sequential processing discards valuable raw sensor data that could have been used to recover high-dynamic-range (HDR) lighting details.
- **The Solution:** Introduce the joint compressive sensing (CS) framework that treats RAW processing and compression as a single inverse problem.
- **The Results:** Briefly state the expected outcome (e.g., "Our method achieves higher Peak Signal-to-Noise Ratio (PSNR) at equivalent measurement rates compared to traditional sequential pipelines").

---

## II. Introduction (1–1.5 Pages)

- **Background:** Explain traditional data acquisition where we sample densely ($N$ samples) and compress later to retain only $S$ significant coefficients.
- **The CS Paradigm Shift:** Introduce the core philosophy of Compressed Sensing: if a signal is sparse, we can capture the compressed information directly at the sensor using a sensing matrix $A$.
- **The Gap in Research:** Identify that while CS is great for compression, it is rarely optimized jointly for lighting and noise enhancement in modern RAW processing.
- **Paper Contributions:** Clearly list three main contributions (e.g., formulating the joint optimization problem, introducing the deep-unfolded solver, and providing a rigorous ablation study).

---

## III. Mathematical Preliminaries & Background (1 Page)

- **The Measurement Model:** Define the noisy observation model as $y = Ax + w$, where $y$ represents the incomplete/noisy measurements, $A$ is the sensing operator (camera filter), $x$ is the true signal, and $w$ is the bounded noise.
- **Sparsity Prior:** Define the sparsity constraint mathematically, noting that the signal of interest $x$ belongs to the set of $S$-sparse vectors:
  $$x \in \Sigma_S \equiv \{x \in \mathbb{R}^N \mid \|x\|_0 \le S\}.$$
- **Convex Relaxation:** Explain that solving the exact $L_0$ minimization is a tough combinatorial search. Therefore, the problem must be relaxed to a convex $L_1$ optimization problem to guarantee a computationally feasible global solution.

---

## IV. Proposed Method: Joint HDR-Compression Architecture (2 Pages)

- **The Objective Function:** Define the joint problem. Minimize the $L_1$ norm subject to an error constraint to simultaneously find the sparsest (most compressed) and cleanest (denoised) image:
  $$x_1^{\epsilon} = \arg\min \|x\|_1 \quad \text{subject to} \quad \|Ax - y\|_2 \le \epsilon.$$
- **The Solver:** Use the Iterative Shrinkage-Thresholding Algorithm (ISTA) or the Alternating Direction Method of Multipliers (ADMM) to handle the separable cost functions.
- **Deep Unfolding (The Novelty):** Unfold the iterations of ISTA/ADMM into a learnable architecture so that parameters like the shrinkage threshold are learned dynamically based on local lighting conditions rather than being fixed.

---

## V. Experimental Setup & Results (2 Pages)

- **Dataset:** Specify the RAW image dataset used for training and testing (e.g., RawNIND or custom captured `.DNG` files).
- **Evaluation Metrics:** Evaluate success using visual fidelity (structural similarity) and PSNR versus the measurement rate.
- **Phase Transition Analysis:** Plot a Phase Transition Curve to map out the exact boundary where the algorithm succeeds versus fails, depending on the sparsity level $S$ and the number of measurements $M$.
- **Ablation Study:** Compare the joint approach against a classical sequential approach (e.g., recovering via standard Orthogonal Matching Pursuit (OMP) followed by post-processing).

---

## VI. Conclusion & Future Work (0.5 Page)

- Summarize how the joint framework successfully bypasses traditional ISP bottlenecks.
- Discuss limitations (e.g., computational time required for iterative solvers, matrix coherence issues).
- Propose future research directions, such as implementing the sensing matrix $A$ directly in physical optical hardware using metamaterials.

---

## VII. Project Execution Timeline

| Week       | Phase                   | Key Deliverables                                                                                       |
| ---------- | ----------------------- | ------------------------------------------------------------------------------------------------------ |
| **Week 1** | Data & Baselines        | Aggregate RAW dataset. Implement a basic OMP solver as a baseline.                                     |
| **Week 2** | Core Algorithm          | Build the forward measurement model ($y = Ax + w$). Implement the ISTA or ADMM optimizer.              |
| **Week 3** | Unfolding & Tuning      | Convert the classical solver into a deep-unfolded framework. Train the model to optimize thresholds.   |
| **Week 4** | Evaluation              | Generate Rate-Distortion graphs and Phase Transition curves. Compare against the baseline.             |
| **Week 5** | Drafting                | Write Introduction, Problem Formulation, and Results sections. Formatting and final polish.           |
