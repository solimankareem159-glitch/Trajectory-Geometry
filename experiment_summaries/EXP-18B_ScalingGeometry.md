# Experiment 18B: Scaling Geometry

**Phase:** 4 — Consolidation & Robustness
**Date:** February 2026 (2026-02-13)
**Models:** Pythia-70m, Qwen2.5-0.5B, Qwen2.5-1.5B
**Status:** Completed (with partial invalidation)

## Connection to Prior Work

EXP-18 formalized the 54-metric suite. EXP-18B was its first real-world stress test: deploying the full pipeline across three architectures simultaneously to investigate cross-model scaling laws for geometric signatures. This was the bridge between the engineering consolidation (EXP-18) and the final validation study (EXP-19).

## Research Question

**How do geometric signatures of reasoning scale across model architectures and parameter counts?** Specifically: do attractor dynamics, trajectory complexity, and convergence signatures follow predictable scaling laws from 70m to 1.5B parameters?

## Experimental Design

*   **Models:** Pythia-70m, Qwen2.5-0.5B, Qwen2.5-1.5B.
*   **Metric Suite:** 57 metrics (54 base + 3 derived) across 12 conceptual families.
*   **Data Reuse:** Hidden states from Experiment 14 (Qwen 0.5B), Experiment 16 (Pythia 70m), and Experiment 16B (Qwen 1.5B).
*   **Analysis:** Cross-model metric comparison, attractor dynamics investigation, trajectory divergence analysis.

## Key Findings

### 1. Data Integrity Failure (Pythia)
The Pythia-70m hidden states data was **corrupted** — tensor dimensions were 1536 instead of the expected 512. This completely invalidated the Pythia results and revealed the absence of preflight shape validation in the pipeline.

### 2. Infrastructure Bugs Discovered
*   **Broadcasting Ambiguity:** `attractor_metrics` crashed when subtracting trajectory centroids due to numpy broadcasting failures when the number of timesteps equaled the embedding dimension ($T_{ref} = D$).
*   **Multiprocessing Overhead:** On Windows, heavy `torch` and `transformers` imports in global scope during process `spawn` caused massive memory pressure and UI freezing.
*   **Persistence Failure:** No incremental saving meant any crash wiped out all progress.

### 3. Hard Constraints Codified
The failures yielded mandatory engineering constraints for all future experiments:
*   Mandatory preflight validation of tensor shapes
*   Explicit dimension handling for centroids (no implicit broadcasting)
*   Lazy imports for multiprocessing workers
*   Process isolation to prevent memory leakage
*   Atomic operations and incremental CSV appending for crash recovery

## Limitations

*   **Pythia Results Invalid:** The corrupted data meant no cross-architecture scaling conclusions could be drawn for the smallest model.
*   **Pipeline Fragility:** The infrastructure failures revealed that the pipeline was not production-ready, despite EXP-18's consolidation.

## Conclusions & Implications

**Verdict: PARTIAL FAILURE / CRITICAL STRESS TEST.** While the empirical goals were only partially achieved (Qwen models succeeded, Pythia failed), EXP-18B served a vital function as a stress test that hardened the pipeline for EXP-19. The engineering constraints discovered here were essential prerequisites for the final validation study.

## Influence on Next Experiment

The hard constraints and infrastructure lessons from EXP-18B directly shaped the design of **EXP-19: Robustness Replication**:
*   Pythia was upgraded from 70m to **410m** (matching the 24-layer depth of Qwen 0.5B for layer-to-layer comparison)
*   Mandatory preflight tensor validation was built into the pipeline
*   The anti-contamination pipeline was redesigned with strict truncation and boundary detection
*   Incremental persistence was implemented to survive crashes
