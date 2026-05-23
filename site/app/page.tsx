import {
  JointVsSequentialPlot,
  ListaPlot,
  PerScenePlot,
  RateDistortionPlot,
} from "@/components/plots";

export default function HomePage() {
  return (
    <main>
      <header>
        <h1>
          Joint-CS:
          <span className="small">
            Simultaneous Illumination Normalization and Sparse Image
            Compression via Unfolded Recovery
          </span>
        </h1>

        <div className="authors">
          <a href="mailto:farhan.sadeek.29@dartmouth.edu">Farhan Sadeek</a>
        </div>
        <p className="affiliation">
          Thayer School of Engineering, Dartmouth College
        </p>
        <p className="venue">
          ENGS 109 final project · NeurIPS-format manuscript, 2026
        </p>

        <div className="links">
          <a href="/paper/neurips_2026.pdf">Paper (PDF)</a>
          <a href="/poster/poster.pdf">Poster (PDF)</a>
          <a href="https://github.com/SadeekFarhan21/Snapshot-RAW-Enhance">
            Code
          </a>
          <a href="#bibtex">BibTeX</a>
        </div>
      </header>

      <section id="abstract">
        <h2>Abstract</h2>
        <p>
          The traditional camera pipeline samples at Shannon&ndash;Nyquist
          rates, compresses to a sparse representation, and only afterwards
          performs illumination correction on the decoded image. This
          sequential ordering discards information: sparse recovery applied
          to a signal whose dynamic range is dominated by an unknown
          multiplicative illumination field has weaker recovery guarantees
          than the same recovery applied to the underlying reflectance. We
          formulate <em>Joint-CS</em>, a single inverse problem in the joint
          variables of DCT-domain reflectance coefficients and a smooth
          per-column illumination field, and solve it by block-coordinate
          alternation between a learned-iterative-shrinkage step on the
          coefficients and a closed-form ridge step on the illumination.
          Across four experiments on a 5-image natural-scene set &mdash; an
          empirical Donoho&ndash;Tanner phase transition, a multi-image
          rate&ndash;distortion sweep, a deep-unfolded LISTA solver, and a
          per-scene ablation against a sequential
          CS-then-illumination-correct baseline &mdash; we find that the
          joint formulation outperforms the sequential one by up to{" "}
          <b>+4.7&nbsp;dB PSNR</b> on textured scenes under an 8&times;
          illumination gradient, and that a $K{"="}10$ unrolled LISTA matches
          what FISTA needs <b>22 iterations</b> to reach. We also report a
          clear failure mode: on near-uniform scenes (moon) the joint
          $g$-update lacks signal to estimate gain from and the method
          degrades by $\sim 8$&nbsp;dB &mdash; a scene-dependence that, to
          our knowledge, the prior CS+ISP literature does not report.
        </p>
      </section>

      <section id="overview">
        <h2>Method overview</h2>

        <p>
          The forward model is the standard compressive-sensing measurement{" "}
          $\boldsymbol{"{y}"} = \boldsymbol{"{A}"}\boldsymbol{"{x}"} +
          \boldsymbol{"{w}"}$ with $\boldsymbol{"{A}"} \in \mathbb{"{R}"}
          ^{"{M\\times N}"}$, $M \ll N$, and bounded noise $\boldsymbol
          {"{w}"}$. The reflectance $\boldsymbol{"{s}"}$ is assumed $S$
          -sparse in a 2D-DCT basis $\boldsymbol{"\\Psi"}$. In a low-light or
          HDR scene the RAW signal is the product of reflectance and a
          smooth spatial illumination field, $\boldsymbol{"{x}"} =
          \boldsymbol{"{s}"} \odot \boldsymbol{"{g}"}$.
        </p>

        <h3>Joint objective</h3>
        <p>
          Rather than recover $\boldsymbol{"{x}"}$ first and divide out{" "}
          $\boldsymbol{"{g}"}$ afterwards, we optimize a single objective in{" "}
          $(\boldsymbol{"{c}"}, \boldsymbol{"{g}"})$:
        </p>

        <p>
          {`$$
          \\min_{\\boldsymbol{c},\\boldsymbol{g}}\\;
          \\tfrac{1}{2} \\big\\| \\boldsymbol{A}\\,\\mathrm{diag}(\\boldsymbol{g})\\,
            \\boldsymbol{\\Psi}\\boldsymbol{c} - \\boldsymbol{y} \\big\\|_2^2
          + \\lambda_c \\|\\boldsymbol{c}\\|_1
          + \\lambda_g \\|\\nabla \\boldsymbol{g}\\|_2^2.
          $$`}
        </p>

        <p>
          The problem is multiplicatively bilinear in $(\boldsymbol{"{c}"},
          \boldsymbol{"{g}"})$ &mdash; non-convex jointly, convex in each
          block individually &mdash; so we alternate:
        </p>
        <ul>
          <li>
            <b>c-step.</b> With $\boldsymbol{"{g}"}$ fixed, the effective
            sensing matrix is $\boldsymbol{"{A}"}_{"{\\text{eff}}"} =
            \boldsymbol{"{A}"}\,\mathrm{"{diag}"}(\boldsymbol{"{g}"})\,
            \boldsymbol{"\\Psi"}$ and the $\boldsymbol{"{c}"}$-update is a
            Lasso, solved by FISTA &mdash; or by a $K{"="}10$-layer LISTA at
            deployment.
          </li>
          <li>
            <b>g-step.</b> With $\boldsymbol{"{c}"}$ fixed, the $\boldsymbol
            {"{g}"}$-update is a ridge regression with closed form{" "}
            $\boldsymbol{"{g}"}^\star = (\boldsymbol{"{B}"}^\top \boldsymbol
            {"{B}"} + \lambda_g \boldsymbol{"{L}"})^{"{-1}"} \boldsymbol
            {"{B}"}^\top \boldsymbol{"{y}"}$.
          </li>
        </ul>

        <h3>Deep-unfolded c-step (LISTA)</h3>
        <p>
          To amortize the c-step we replace the FISTA inner loop with $K$
          unrolled ISTA layers whose encoder, recurrent, and threshold
          parameters are <em>learned</em>:
        </p>
        <p>
          {`$$
          \\boldsymbol{c}^{(k+1)} =
          \\mathcal{S}_{\\theta_k}\\!\\big(
            \\boldsymbol{W}_t \\boldsymbol{c}^{(k)} + \\boldsymbol{W}_e \\boldsymbol{y}
          \\big),\\quad k=0,\\dots,K-1.
          $$`}
        </p>
        <p>
          Initializing $\boldsymbol{"{W}"}_e = (1/L)(\boldsymbol{"{A}"}
          \boldsymbol{"\\Psi"})^\top$, $\boldsymbol{"{W}"}_t = \boldsymbol
          {"{I}"} - (1/L)(\boldsymbol{"{A}"}\boldsymbol{"\\Psi"})^\top
          \boldsymbol{"{A}"}\boldsymbol{"\\Psi"}$, and $\theta_k =
          \lambda/L$ exactly reproduces ISTA at training-step zero, so the
          network is a strict generalization of the classical solver.
        </p>
      </section>

      <section id="results">
        <h2>Results</h2>

        <h3>
          Rate&ndash;distortion across natural images{" "}
          <span className="pill">Exp. 2</span>
        </h3>
        <RateDistortionPlot />
        <p className="plot-note">
          <b>
            FISTA and ADMM beat OMP by 2&ndash;3&nbsp;dB across the useful
            measurement-rate range.
          </b>{" "}
          5-image test set (cameraman, astronaut, coins, page, moon) at
          64&times;64 grayscale, 16&times;16 DCT blocks, Gaussian sensing,
          SNR&nbsp;=&nbsp;30&nbsp;dB. Hover for exact values; toggle solvers
          via the legend.
        </p>

        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>$\delta = M/N$</th>
                <th>OMP</th>
                <th>FISTA</th>
                <th>ADMM</th>
              </tr>
            </thead>
            <tbody>
              <tr><td>0.10</td><td>15.46 ± 4.4</td><td><b>16.67 ± 4.8</b></td><td>11.23 ± 3.1</td></tr>
              <tr><td>0.20</td><td>17.42 ± 4.3</td><td><b>19.39 ± 4.5</b></td><td>17.62 ± 4.8</td></tr>
              <tr><td>0.30</td><td>19.15 ± 4.0</td><td><b>21.73 ± 4.3</b></td><td>21.27 ± 4.6</td></tr>
              <tr><td>0.40</td><td>20.42 ± 3.7</td><td><b>23.64 ± 4.1</b></td><td>23.47 ± 4.4</td></tr>
              <tr><td>0.50</td><td>21.73 ± 3.2</td><td><b>25.06 ± 3.9</b></td><td>24.98 ± 4.0</td></tr>
              <tr><td>0.60</td><td>23.50 ± 3.0</td><td><b>26.43 ± 3.8</b></td><td>26.37 ± 3.9</td></tr>
              <tr><td>0.70</td><td>24.89 ± 3.0</td><td><b>27.57 ± 3.5</b></td><td>27.45 ± 3.7</td></tr>
            </tbody>
          </table>
        </div>
        <p style={{ fontSize: "13.5px", color: "var(--muted)", marginTop: 4 }}>
          PSNR in dB, mean ± std across the 5-image set.
        </p>

        <figure>
          <img
            src="/figures/rate_distortion_qualitative.png"
            alt="Qualitative reconstruction comparison."
          />
          <figcaption>
            Qualitative reconstructions on <em>cameraman</em> for OMP /
            FISTA / ADMM across the $\delta$ sweep.
          </figcaption>
        </figure>

        <h3>
          Donoho&ndash;Tanner phase transition{" "}
          <span className="pill">Exp. 1</span>
        </h3>
        <figure>
          <img
            src="/figures/phase_transition.png"
            alt="Empirical phase transition for OMP vs FISTA."
          />
          <figcaption>
            Both solvers exhibit a sharp empirical Donoho&ndash;Tanner
            transition. <b>OMP wins at small $\delta$; FISTA wins at large
            $\delta$</b>; the crossover sits near $\delta \approx 0.65$.
            Grid: 19&times;19, 20 trials per cell, $N{"="}200$, zero noise.
          </figcaption>
        </figure>

        <h3>
          LISTA at matched compute <span className="pill">Exp. 3</span>
        </h3>
        <ListaPlot />
        <p className="plot-note">
          A $K{"="}10$-layer learned solver matches what FISTA needs{" "}
          <b>22 iterations</b> to reach &mdash; a $\sim 2.2\times$ iteration
          speedup at equivalent recovery quality. Log-scale NMSE; lower is
          better.
        </p>

        <h3>
          Joint vs. sequential under an illumination gradient{" "}
          <span className="pill">Exp. 4 · headline</span>
        </h3>
        <JointVsSequentialPlot />
        <p className="plot-note">
          Aggregate PSNR vs measurement rate for the joint and sequential
          pipelines under an 8&times; horizontal illumination gradient. Both
          curves are scored against the illumination-normalized scene.
          Shaded bands show ±1&nbsp;std across the 5-scene set.
        </p>

        <PerScenePlot />
        <p className="plot-note">
          Per-scene PSNR gain $\Delta = \text{"{joint}"} - \text
          {"{sequential}"}$ (dB). Hover any bar for the exact value; the{" "}
          <em>moon</em> column reveals the failure mode where the joint
          $g$-update is rank-deficient.
        </p>

        <figure>
          <img
            src="/figures/joint_vs_sequential_qualitative.png"
            alt="Qualitative joint vs sequential reconstructions."
          />
          <figcaption>
            Qualitative reconstructions for the joint and sequential
            pipelines on each test scene.
          </figcaption>
        </figure>
      </section>

      <section id="summary">
        <h2>Summary</h2>
        <ul>
          <li>
            <b>Phase transition.</b> Sharp empirical Donoho&ndash;Tanner
            transition for both solvers; crossover near $\delta \approx
            0.65$.
          </li>
          <li>
            <b>Rate&ndash;distortion.</b> FISTA / ADMM beat OMP by
            2&ndash;3&nbsp;dB across the useful $\delta$ range.
          </li>
          <li>
            <b>LISTA.</b> A 10-layer learned solver matches FISTA at 22
            iterations &mdash; $\sim 2.2\times$ iteration speedup.
          </li>
          <li>
            <b>Joint vs sequential.</b> Up to <b>+4.7&nbsp;dB</b> on
            textured scenes; catastrophic failure on near-uniform scenes is
            the most informative limitation.
          </li>
        </ul>
      </section>

      <section id="paper">
        <h2>Read the paper</h2>
        <p>
          <a href="/paper/neurips_2026.pdf">Open in new tab</a> ·{" "}
          <a href="/paper/neurips_2026.pdf" download>Download PDF</a>
        </p>
        <iframe
          className="pdf-frame"
          src="/paper/neurips_2026.pdf#view=FitH"
          title="Joint-CS paper PDF"
        />
      </section>

      <section id="poster-section">
        <details>
          <summary>Poster</summary>
          <p style={{ marginTop: 10 }}>
            <a href="/poster/poster.pdf">Open in new tab</a> ·{" "}
            <a href="/poster/poster.pdf" download>Download PDF</a>
          </p>
          <iframe
            className="pdf-frame"
            src="/poster/poster.pdf#view=FitH"
            title="Joint-CS poster PDF"
          />
        </details>
      </section>

      <section id="reproduce">
        <h2>Reproducing the experiments</h2>
        <pre>
          <code>{`git clone https://github.com/SadeekFarhan21/Snapshot-RAW-Enhance
cd Snapshot-RAW-Enhance

python3 experiments/phase_transition.py      # ~8 min CPU
python3 experiments/rate_distortion.py       # ~12 s
python3 experiments/train_lista.py           # ~4 min CPU
python3 experiments/joint_vs_sequential.py   # ~5 s`}</code>
        </pre>
      </section>

      <section id="bibtex">
        <h2>BibTeX</h2>
        <pre>
          <code>{`@misc{sadeek2026jointcs,
  author       = {Farhan Sadeek},
  title        = {{Joint-CS}: Simultaneous Illumination Normalization
                  and Sparse Image Compression via Unfolded Recovery},
  year         = {2026},
  howpublished = {ENGS 109 final project,
                  Thayer School of Engineering, Dartmouth College},
  url          = {https://github.com/SadeekFarhan21/Snapshot-RAW-Enhance}
}`}</code>
        </pre>
      </section>

      <footer>
        <p>
          © 2026 Farhan Sadeek · Dartmouth College ·{" "}
          <a href="/paper/neurips_2026.pdf">PDF</a> ·{" "}
          <a href="https://github.com/SadeekFarhan21/Snapshot-RAW-Enhance">
            source
          </a>
        </p>
      </footer>
    </main>
  );
}
