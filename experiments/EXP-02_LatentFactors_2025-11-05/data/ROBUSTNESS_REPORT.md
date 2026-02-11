# Robustness Report: Sensitivity & Stability Analysis

## 1. Dimensionality of Operator Space

We performed an ablation study on the number of latent factors $k$ used in the Non-negative Matrix Factorization (NMF) of operator signatures.

- **Finding**: There is no sharp "elbow" in the Reconstruction MSE curve. The error decreases monotonically as $k$ increases from 2 to 8.
- **Implication**: The "true" dimensionality of the operator space may be higher than 4. However, $k=4$ captures the dominant modes (likely corresponding to broad categories like "reduction", "expansion", "negation", "creation").
- **Robustness**: The qualitative shape of the top 2 factors remains consistent across $k=3$ and $k=4$.

## 2. Layer-wise Localization

We tested the hypothesis that geometric signatures are a property of the "middle" conceptual reasoning layers.

- **Hypothesis**: Supported.
- **Observations**:
  - **Layer 0 (Input)**: Almost zero error. The "warp" here involves embedding lookups which are static. The low error suggests operator prompts have very similar initial token embedding dynamics (likely due to common phrasing structure).
  - **Layers 5-12 (Middle)**: Moderate error (2.7 - 4.4). This is the "sweet spot" where meaningful, distinctive processing occurs, and signatures are stable enough to be modeled.
  - **Layer 24 (Output)**: Catastrophic error (>1800). The geometry at the final layer is dominated by the need to predict specific next tokens, causing high-variance shifts that obscure the high-level operator signal.

## 3. Limits of Linearity

The assumption that composite operators = $\sum$ single operators is a linearization of a non-linear system.

- **Supported**: Factor activation heatmaps show that composite prompts do trigger the relevant factors from their components.
- **Challenged**: The reconstruction is imperfect (MSE ~14.8 at $k=4$), indicating that non-linear interaction terms (interference or synergy between operators) play a significant role.
