# OPT 2026 submission checklist

## Required before upload

- [x] Use the official OPT 2026 anonymous class.
- [x] Keep the main text within the five-page soft limit.
- [x] Remove names, affiliations, acknowledgments, and identifying URLs.
- [x] State that LISTA is evaluated separately from the joint outer loop.
- [x] State that the benchmark is not real camera RAW data.
- [x] Include dense time/memory scaling and the low-energy failure regime.
- [x] Inspect the current PDF metadata and prose for author identity.
- [ ] Repeat the identity/metadata inspection immediately before upload.
- [ ] Create or verify every author's OpenReview profile; non-institutional profiles can require up to two weeks of moderation.
- [ ] Confirm no concurrent submission to another NeurIPS 2026 workshop.
- [ ] Enter the final title, abstract, author list, conflicts, and subject areas in OpenReview.

## High-priority scientific upgrades

- [x] Run LISTA with five optimizer seeds and report mean, standard deviation, range, and per-seed results.
- [x] Add measured wall-clock latency, working-memory estimates, parameter counts, and multiply-add counts at several block sizes.
- [x] Frame LISTA explicitly as a fixed-operator inner-solver diagnostic; it is not presented as part of the changing-operator joint loop.
- [ ] Validate the proposed scene-energy gate on held-out scenes; the current experiment only falsifies the condition-number gate.
- [ ] Add at least one modern learned inverse-problem baseline.
- [ ] For camera/RAW claims, evaluate linear sensor data from a real RAW dataset and use a physically plausible structured sensing operator.

## Current evidence boundary

- Joint optimization: five `scikit-image` scenes, one 16x16 center patch per scene, synthetic 8x horizontal gain, Gaussian sensing.
- Rate-distortion: five 64x64 grayscale images reconstructed in 16x16 blocks.
- LISTA: one fixed sensing matrix and dataset, synthetic sparse coefficients, five optimizer seeds ($0.0435\pm0.0120$ NMSE).
- Scaling: analytic complexity plus batch-1, one-thread CPU latency and working-memory estimates for $N\in\{64,200,256,1024\}$; no full-resolution runtime claim.
- Negative result: the measured condition number does not identify the low-energy failure scene.

## Suggested submission metadata

- Title: *Deep Unfolding and Failure Modes in Joint Sparse Recovery with Unknown Illumination*
- Venue: OPT 2026: Optimization for Machine Learning (NeurIPS 2026 Workshop)
- Keywords: deep unfolding; sparse optimization; block-coordinate optimization; inverse problems; compressed sensing
- Primary topic: algorithms for nonsmooth problems / optimization with sparsity constraints
- Secondary topics: nonconvex optimization; empirical evaluation of optimization algorithms; optimization software and scaling
