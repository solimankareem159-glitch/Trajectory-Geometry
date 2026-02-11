# Quantitative Results: Geometric Signatures

## Table 1: Latent Factor Model Selection (Ablation on $k$)

Reconstruction Mean Squared Error (MSE) for composite operator signatures using NMF with varying numbers of latent factors.

| Number of Factors ($k$) | Reconstruction MSE | Relative Improvement |
| :--- | :--- | :--- |
| **2** | 17.05 | - |
| **3** | 15.93 | 6.6% |
| **4 (Selected)** | 14.85 | 6.7% |
| **5** | 12.48 | 16.0% |
| **6** | 12.38 | 0.8% |
| **7** | 10.23 | 17.3% |
| **8** | 8.29 | 19.0% |

*Interpretation: Increasing $k$ improves reconstruction, but we selected $k=4$ as a parsimonious balance between accuracy and interpretability.*

## Table 2: Layer-wise Sensitivity Analysis

Reconstruction MSE for operator signatures extracted from specific transformer layers ($k=4$).

| Layer Index | Reconstruction MSE | Network Depth | Interpretation |
| :--- | :--- | :--- | :--- |
| **0** (Embeddings) | 0.002 | Input | Identity/near-identity structure. Trivial reconstruction. |
| **5** (Early) | 2.77 | 20% | Emerging geometric differentiation. |
| **12** (Middle) | 4.40 | 50% | Stable representation space. |
| **18** (Late) | 19.97 | 75% | High-divergence processing. |
| **24** (Final) | 1809.72 | Output | Logit preparation; geometry breaks down for next-token prediction. |

*Interpretation: Geometric signatures are most stable and "clean" in the early-to-middle layers. The final layer shows a massive explosion in variance, likely due to the projection onto the vocabulary space.*

## Table 3: Cross-Prompt Stability (Qualitative)

Based on visual inspection of warp traces across operator families.

| Operator Family | Onset Type | Duration | Stability |
| :--- | :--- | :--- | :--- |
| **Summarize** | Sharp Spike | Short (<5 tokens) | High |
| **Criticize** | Gradual Ramp | Medium | Moderate |
| **Poetic** | Oscillating | Long | Low (Content dependent) |
| **Reframe** | Multi-modal | Medium | Moderate |
