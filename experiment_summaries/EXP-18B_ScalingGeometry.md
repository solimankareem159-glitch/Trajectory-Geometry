# Experiment 18B: Scaling Geometry

**Status:** Completed (with partial invalidation)
**Date:** Feb 2026 (2026-02-13)
**Models:** Pythia-70m, Qwen2.5-0.5B, Qwen2.5-1.5B

## Motivation & Prior Assumptions

* **Goal:** Compute a unified suite of 57 geometric metrics across three transformer models of varying scales to investigate the emergence of "attractor dynamics" and "geometric signatures" in chain-of-thought reasoning.
* **Data Aggregation:** Reused hidden states from Experiment 14 (Qwen 0.5B), Experiment 16 (Pythia 70m), and Experiment 16B (Qwen 1.5B).

## Key Results & Analysis

* **Scaling Laws:** Analyzed how 57 distinct metrics (across 12 families) scale across intra-model and cross-model conditions.
* **Attractor Dynamics:** Investigated the distance to success centroids, local expansion rates, and trajectory divergence.

## Methodological Issues / Limitations

* **Data Integrity Failure:** The "Pythia 70M" identity crisis. The Pythia 70M hidden states data was corrupted (dimension was 1536 instead of 512), completely invalidating the Pythia 70M results for this experiment.
* **Broadcasting Ambiguity:** `attractor_metrics` crashed when subtracting a trajectory centroid due to unpredictable numpy broadcasting when `T_ref == D`.
* **Multiprocessing Overhead:** On Windows, the heavy `torch` and `transformers` imports in global scope during `spawn` caused massive memory pressure and UI freezing.
* **Lack of Persistence:** The system lacked incremental saving, meaning any crash wiped out all progress, prompting a move to atomic operations and incremental CSV appending.

## Interpretation (Meanings & Implications)

* **Hard Constraints Discovered:** This experiment functioned as a stress test for the pipeline. It resulted in hard constraints for future experiments: mandatory pre-flight validation of tensor shapes, explicit dimension handling for centroids, lazy imports for multiprocessing, and process isolation to prevent memory leakage.
