# Methods Report: Geometric Signatures of Conversational Operators

## 1. Introduction

This study investigates whether "conversational operators"—high-level cognitive moves like *summarizing*, *reframing*, or *critiquing*—manifest as distinct, recoverable geometric signatures within the latent space of a transformer language model. We hypothesize that these operators are not merely semantic labels but represent distinct modes of processing with consistent geometric properties.

## 2. Geometric Feature Extraction

### 2.1. Warp Signal Definition

We define the "warp signal" $W(t)$ at token position $t$ as the Euclidean norm of the difference between the hidden state vector $h_t$ and its predecessor $h_{t-1}$:

$$ W(t) = \|h_t - h_{t-1}\|_2 $$

This metric captures the magnitude of state transition in the embedding space. A high warp value indicates a significant shift in the model's internal representation, potentially corresponding to the initiation of a new cognitive operation. We compute this signal for each layer $l$ and average across layers to obtain a robust global signal.

### 2.2. Operator Span Detection

To automatically segment the warp signal into distinct operational phases, we employ the **Pruned Exact Linear Time (PELT)** change-point detection algorithm. We model the signal using a Radial Basis Function (RBF) cost function, identifying time points $t^*$ where the statistical properties of the warp signal change abruptly.

## 3. Latent Factor Decomposition

### 3.1. Signature Construction

For each detected operator span, we extract a **geometric signature** vector $S$, defined as the sequence of warp values over the first $N=15$ tokens of the span. This fixed-length vector captures the initial "velocity profile" of the operator.

### 3.2. Non-negative Matrix Factorization (NMF)

To test the compositionality of these operators, we apply Non-negative Matrix Factorization (NMF) to the set of single-operator signatures. We decompose the signature matrix $X$ into two lower-rank matrices $W$ and $H$:

$$ X \approx WH $$

Where:

- $X$ is the $n \times m$ matrix of observed signatures ($n$ operators, $m$ time steps).
- $W$ is the $n \times k$ matrix of activation weights.
- $H$ is the $k \times m$ matrix of latent geometric basis vectors (factors).

We hypothesize that composite operators (e.g., "Summarize + Criticize") can be reconstructed as linear combinations of the basis vectors learned from single operators, with activation weights reflecting the superposition of the underlying cognitive moves.

## 4. Evaluation Metrics

We evaluate the quality of the decomposition using:

- **Reconstruction Error (MSE)**: Mean Squared Error between the original and reconstructed signatures.
- **Factor Sparsity**: The degree to which operators activate specific latent factors.
- **Visual Inspection**: Qualitative assessment of the correspondence between warp curves and operator semantics.
