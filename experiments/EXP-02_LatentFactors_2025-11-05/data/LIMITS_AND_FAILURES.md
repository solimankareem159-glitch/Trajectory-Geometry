# Limits and Failures Log

## 1. Sensitivity to Output Layer Dynamics

- **Observation**: Geometric signatures completely break down at the final transformer layer (Layer 24).
- **Metric**: Reconstruction MSE explodes from ~4.4 (Layer 12) to ~1809 (Layer 24).
- **Cause**: The final layer's geometry is likely dominated by the "unembedding" projection to the vocabulary size (vocab size > hidden dim), creating high-variance, sparse activations needed for next-token prediction.
- **Implication**: Any "cognitive geometry" analysis must be restricted to the middle layers (6-18) where abstract reasoning occurs.

## 2. Limits of Linear Compositionality

- **Observation**: While composite operators activate the correct latent factors, the linear reconstruction is imperfect (MSE ~14.8).
- **Interpretation**: Constructive interference or non-linear contextual modulation occurs when operators are combined. The whole is not exactly the sum of its parts. A simple linear superposition model captures the *dominant* trend but misses interaction terms.

## 3. Sample Size Constraints

- **Constraint**: This study used a controlled dataset of 15 prompts (10 single + 5 composite) with a single base context.
- **Risk**: The observed signatures might partially overfit to the specific phrasing of the prompts or the base content "Artificial intelligence...".
- **Mitigation**: Future work must test cross-topic generalization (as done in Paper I) combined with this token-level analysis.

## 4. Segmentation Ambiguity

- **Issue**: The PELT algorithm requires a penalty parameter (`pen`) which significantly affects the number of detected change points.
- **Tuning**: We used `pen=2` based on visual inspection of the "Summarize" operator. This may not be optimal for all operator types, leading to potential over-segmentation or under-segmentation in subtler operators like "Poetic".
