# Experiment 6 Checklist: Evaluation & Controls

**Status:** PENDING  
**Date:** 2026-01-27

## Data Generation & Preprocessing

- [ ] **Operator Set**: Defined 10 distinct operators.
- [ ] **Content Set**: Sourced $\ge 20$ diverse passages.
- [ ] **Paraphrase Set**: Created $\ge 10$ unique phrasings per operator.
- [ ] **Generation**: 2,000 samples generated ($T=0$).
- [ ] **Quality Check**: Manual review of a random sub-sample (N=20) to ensure model actually followed instructions.

## Representation Extraction

- [ ] **Feature Calculation**: Normalized Directional Deltas computed.
- [ ] **Neutral Baseline**: "Continue" condition generated and subtracted (if using subtraction method).
- [ ] **Tensors Saved**: shape `[N_samples, N_layers, N_tokens, D_model]`.

## Modeling & decoding

- [ ] **Split Construction**:
  - [ ] Random Split (Sanity Check)
  - [ ] Cross-Topic Split (Context Hold-out)
  - [ ] Cross-Paraphrase Split (Instruction Hold-out)
  - [ ] Joint Hold-out (Rigorous Test)
- [ ] **Probe Training**: Logistic Regression trained on each split.
- [ ] **Sweep**: Trained independent probes for each Layer $L$.

## Critical Controls (The "Falsification Filters")

- [ ] **Control 1: Permutation Test**: Labels shuffled, probe re-trained. Result $\approx$ Chance?
- [ ] **Control 2: Content-Only**: Probe trained on content embeddings only. Result $\approx$ Chance?
- [ ] **Control 3: Instruction Masking**: Probe tested on inputs where specific instruction words are masked/neutralized (if feasible).
- [ ] **Control 4: Lexical Baseline**: Simple classifier (TF-IDF/Bag-of-Words) on instruction text. Result < Internal State Probe?

## Reporting & Visualization

- [ ] **Layer-wise Accuracy Curve**: Plot Accuracy vs. Layer Index for all splits.
- [ ] **Confidence Intervals**: 95% Bootstrap intervals shaded.
- [ ] **Confusion Matrix**: Heatmap of predicted vs. true operator.
- [ ] **Claims**: Do explicitly stat limitations if generalization fails in Joint Hold-out.
