# Experiment 7B: Comprehensive Rigor Audit Report

## Executive Summary

This report details the findings of the rigor audit performed on Experiment 7B. The objective was to validate the existence of "Latent Operator States" in the transformer representation space by measuring geometric invariance, stability, and robustness to controls.

**Status**: Analysis Complete
**Outcome**: Negative Result
**Recommendation**: **STOP**. Do not proceed to Experiment 8.

## Methodology

- **Data**: Full 2K dataset (2000 samples, 20 contents, 10 paraphrases, 10 operators).
- **Metric**: Nearest-Centroid Classification (NCC) Macro-F1.
- **Controls**:
  - **Baseline**: Standard stratified split.
  - **LOPO (Leave-One-Paraphrase-Out)**: Tests robustness to specific wording.
  - **Joint Holdout**: Tests hard generalization (hold out content+paraphrase).
  - **Instruction Masking**: *Skipped (Data Unavailable)*.
  - **Permutation**: Chance baseline.

## Detailed Results

### 1. Layer-wise Geometric Signature

| Layer | Baseline F1 | LOPO F1 | Joint F1 | Stability | Permutation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0** (Embed) | 0.288 ± 0.017 | 0.203 ± 0.075 | 0.226 | 0.855 | 0.080 |
| **10** | 0.306 ± 0.015 | 0.202 ± 0.072 | 0.239 | 0.918 | 0.085 |
| **11** | 0.307 ± 0.015 | 0.206 ± 0.076 | 0.240 | 0.918 | 0.069 |
| **12** | 0.303 ± 0.019 | 0.205 ± 0.073 | 0.244 | 0.918 | 0.077 |
| **13** | 0.305 ± 0.017 | 0.205 ± 0.071 | 0.236 | 0.920 | 0.067 |
| **14** | 0.308 ± 0.022 | 0.203 ± 0.069 | 0.241 | 0.919 | 0.068 |
| **15** (Peak) | **0.316** ± 0.016 | **0.215** ± 0.066 | **0.247** | **0.922** | 0.079 |
| **16** | 0.303 ± 0.015 | 0.210 ± 0.069 | 0.239 | 0.924 | 0.069 |
| **24** | 0.268 ± 0.012 | 0.185 ± 0.050 | 0.211 | 0.912 | 0.086 |

### 2. Validation Checks

#### A. Layer 0 vs. Mid-Layers (The "Emergence" Check)

- **Layer 0 F1**: 0.288
- **Peak Mid-Layer F1**: 0.316 (Layer 15)
- **Delta**: +0.028 (+9.7%)
- **Interpretation**: The improvement is marginal. The vast majority of the geometric structure used to distinguish operators is already present in the embeddings (Layer 0). This suggests the model is relying on "Instruction Lexical Templates" (specific words in the prompt) rather than building an abstract internal state.

#### B. Robustness (LOPO & Joint Holdout)

- **Drop from Baseline to LOPO**: ~0.31 -> ~0.21 (-32%)
- **Interpretation**: The geometry is highly sensitive to the specific paraphrase wording. If the model had learned an abstract "Operator" concept, it should generalize better to unseen paraphrases. The significant drop confirms that the "Baseline" performance is inflated by memorizing specific paraphrase patterns.

#### C. Stability

- **Observation**: Stability is high across the board (>0.85), even in Layer 0. This confirms that the embeddings themselves are stable across contents, likely due to consistent instruction wording.

## Conclusions

1. **Lexical Dominance**: The results are consistent with a model that relies primarily on surface-level Instruction Lexical Templates. There is no strong evidence of a "Latent Operator State" emerging in the middle layers that is distinct from the input instruction.
2. **Weak Generalization**: The poor LOPO and Joint Holdout performance indicates the geometric "invariance" is brittle and likely not a robust causal mechanism for operator transitions.
3. **Experiment 8 Viability**: Analysis of dynamic trajectories requires a stable, high-quality static target (the "Operator Attractor"). Since the static geometry is weak and indistinguishable from embeddings, trajectory analysis would likely just track the "decay" of the instruction signal rather than true computation.

## Recommendation

**DO NOT PROCEED**.
The criteria for Experiment 8 (Robust Geometric Structure) have **not** been met.
The project should pivot to investigating why the abstract state is not forming (e.g., model size, task complexity, or increasing dataset diversity to force generalization).
