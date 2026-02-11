# Experiment Specification: Geometric Signatures of Conversational Operators

## 1. System Configuration

- **Model Architecture**: `Qwen/Qwen2.5-0.5B` (0.5 Billion parameters, Transformer decoder-only).
- **Source**: Hugging Face Hub (`Qwen/Qwen2.5-0.5B`).
- **Precision**: FP32 (Full Precision) for extraction and analysis.
- **Hardware details**: Local CPU execution.
- **Key Libraries**:
  - `torch` (v2.x)
  - `transformers` (v4.x)
  - `ruptures` (Change-point detection)
  - `scikit-learn` (PCA, NMF, Clustering)

## 2. Dataset Construction

### Operator Families

We defined 10 distinct conversational operators applied to a fixed base content ("Artificial intelligence is transforming science, medicine, and creativity."):

1. Summarize
2. Criticize
3. Reframe Positively
4. Reframe Negatively
5. Ask Question
6. Explain Simply
7. Explain Technically
8. Opposite Stance
9. Poetic
10. Ethical Risks

### Composite Operators

We constructed 5 composite prompts to test superposition:

1. Summarize + Criticize
2. Reframe Negatively + Summarize
3. Poetic + Criticize
4. Explain Technically + Ethical Risks
5. Opposite Stance + Ask Question

**Total Samples**: 10 Single + 5 Composite = 15 Controlled Prompts.

## 3. Preprocessing & Extraction

- **Input Processing**: Tokenization using Qwen tokenizer.
- **Internal Tensors**:
  - Hidden States ($h_l^t$): Extracted for all layers $l \in [0, 24]$ and all tokens $t$.
- **Geometric Features**:
  - **Warp Signal**: Euclidean norm of the backward difference vector: $\|\Delta_t\| = \|h_t - h_{t-1}\|_2$.
  - Computed per-layer and averaged across layers ("Mean Warp Trace").

## 4. Analysis Pipeline

### A. Segmentation

- **Algorithm**: PELT (Pruned Exact Linear Time).
- **Metric**: Radial Basis Function (RBF).
- **Implementation**: `uptures.Pelt`.

### B. Signature Extraction

- **Definition**: The mean warp trace of the first 15 tokens of the generated response.
- **Normalization**: None (raw magnitude preserved).

### C. Latent Decomposition

- **Method**: Non-negative Matrix Factorization (NMF).
- **Components ($k$)**: Tested $k \in [2, 8]$. Selected $k=4$ for primary analysis.
- **Initialization**: Random.

## 5. Metadata

- **Date**: 2026-01-27
- **Codebase**: `antigravity` scratch directory.
