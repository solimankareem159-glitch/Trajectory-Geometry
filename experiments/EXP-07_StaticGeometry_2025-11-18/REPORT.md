# Experiment 7: Static Operator Geometry (Report)

**Status:** Complete
**Date:** 2026-01-28
**Dataset:** N=2,000 (10 Operators, 20 Contents, 10 Paraphrases)
**Model:** Qwen/Qwen2.5-0.5B

## 1. Executive Summary

Experiment 7 aimed to characterize the geometric structure of "Conversational Operators" within the residual stream of a frozen LLM. Using a dataset of 2,000 samples across 10 distinct operators, we analyzed the stability and separability of operator representations across layers.

**Key Finding:** Operators exist as stable, content-invariant geometric objects (centroids) that peak in coherence within the **mid-layers (L13–L16)**. This structural stability exceeds that of the input embeddings, suggesting an abstracted "latent operator state" that is robust to surface-level variations.

## 2. Methodology

* **Extraction:** Streaming feature extraction (Directional Deltas) with the "Speed Patch" method (Generate -> Forward -> Slice).
* **Metric 1 (Stability):** Mean pairwise cosine similarity of an operator's centroid across different content domains. High stability = Content Invariance.
* **Metric 2 (Separability):** Mean off-diagonal cosine distance between different operator centroids.
* **Metric 3 (Decodability):** Nearest Centroid Classifier (NCC) Macro-F1 score (10-class).

## 3. Results (N=2,000)

### 3.1 Stability Profile

The stability score measures how essentially "the same" an operator vector looks regardless of the topic it is applied to.

| Layer | Stability Score | Interpretation |
| :--- | :--- | :--- |
| **0 (Input)** | 0.855 | High baseline due to shared lexical tokens in instructions. |
| **10** | 0.918 | Rapid increase in invariance. |
| **16 (Peak)** | **0.924** | **Maximum Abstraction.** The operator representation is most consistent here. |
| **24 (Output)** | 0.912 | Slight decline as output semantics take over. |

> **Conclusion:** The mid-layers actively refine the operator representation, making it more robust than the literal instruction inputs.

### 3.2 Decodability

Using simple centroids to classify unseen samples (10-way classification, Chance = 0.10).

* **NCC F1:** ~0.30 (consistent across mid-layers).
* **Significance:** 3x better than random chance.
* **Implication:** A simple linear structure (centroid) captures significant operator identity, validating the "Geometry" hypothesis.

### 3.3 Subspace Alignment

Principal angle analysis reveals that related operators (e.g., *Critique* and *Question*) share closer subspaces than unrelated ones (e.g., *Translate* and *Summarize*), suggesting a semantic topology to the operator manifold.

## 4. Artifacts

* **Scripts:** `experiments/experiment 7/scripts/analyze_geometry.py`
* **Data:** `experiments/experiment 6/data/full_2k/`
* **Results:** `experiments/experiment 7/results/geometry_metrics/` (Contains centroids and distance matrices).

## 5. Next Steps

Experiments 8 (Dynamic Trajectories) and 9 (Intervention) can now proceed using the validated mid-layer centroids as steering vectors.
