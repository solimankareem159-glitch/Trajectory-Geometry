# Experiment 17: Baseline Replication & Multi-Mode Prompting

**Status:** Completed
**Date:** Feb 2026 (2026-02-11)
**Models:** Qwen2.5-3B-Instruct

## Motivation & Prior Assumptions

* **Goal:** Replicate Experiment 09 findings using a full 33-metric pipeline and test multi-mode computational signatures.
* **Scale Ladder:** Extends the Qwen family scale ladder to 3B parameters (following 0.5B in Exp 14 and 1.5B in Exp 16B).

## The Experiment

* **Phase 17A:** Direct replication of EXP-09 using the same multi-step arithmetic dataset (300 problems).
* **Phase 17B:** 8-mode multi-mode prompting, testing the same content across different computational modes.

## Key Results & Analysis

* **Regime-Relative Geometry:** Continued testing of how geometric signatures operate relative to the reasoning regime.
* **Metric Expansion:** Successfully applied an expanded 33-metric pipeline to a larger 3B parameter model.

## Methodological Issues / Limitations

* Hardware constraints (AMD RX 5700 XT with 8GB via DirectML) began to pose challenges at the 3B scale.

## Interpretation (Meanings & Implications)

* **Scaling Consistency:** Provides another data point to confirm if geometric signatures of reasoning hold consistent as model capacity increases up to 3 billion parameters.
