# Experiment 19: Robustness Replication

**Status:** Planning
**Date:** Feb 2026 (2026-02-14)
**Models:** Qwen2.5-0.5B, Qwen2.5-1.5B, Pythia-410m

## Motivation & Prior Assumptions

* **Goal:** Definitive validation study for Trajectory Geometry across three architectures with the full 57-metric suite and strict anti-contamination measures.
* **Design Shift:** Upgraded Pythia from 70m to 410m to match the 24-layer depth of Qwen 0.5B, allowing direct layer-to-layer cross-architecture comparison.

## The Experiment

* **Dataset:** 400 problems balanced across 4 difficulty bins (Small, Medium, Large, Negative).
* **Anti-Contamination Pipeline:** Implementation of strict multi-stage guardrails (prompt engineering, generation stop sequences, post-generation text truncation, and boundary detection) to solve the "runaway hallucination" issue seen in Exps 16/16B.

## Planned Analysis

* **Difficulty Stratification:** Will test if geometric signatures survive when difficulty is controlled (e.g., separating true reasoning from mere problem complexity).
* **Clustering:** Unsupervised subtyping of failure and success trajectories.
* **Predictive Modeling:** Isolating geometric predictive power from the "length confound" (trajectory duration).

## Methodological Improvements

* **Strict Truncation:** Ensures trajectories are evaluated strictly on the reasoning tokens, ignoring post-answer hallucinations.
* **Layer Strategy:** Selectively runs core metrics on all layers while confining expensive metrics to diagnostic layers (Early, Middle, Late) to minimize multiple independent comparisons.
