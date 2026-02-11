# Paper II Executive Summary: The Geometry of Conversational Operators

## Abstract

This study provides empirical evidence that high-level conversational operators (e.g., *summarization*, *critique*, *reframing*) correspond to distinct, reproducible geometric trajectories within the latent space of a transformer language model (Qwen2.5-0.5B). By analyzing the "warp signal"—the token-to-token displacement in embedding space—we identify characteristic signatures for known operators. Furthermore, using Non-negative Matrix Factorization (NMF), we demonstrate that composite operators can be modeled as linear superpositions of latent geometric factors, suggesting a compositional structure to the model's internal cognitive processing.

## Key Empirical Findings

1. **Distinct Geometric Profiles**: Conversational operators are geometrically distinguishable. "Summarize" induces a sharp, high-magnitude initial warp followed by stability, whereas "Poetic" and "Reframe" induce continuous, oscillating warp traces.
2. **Layer Localization**: These geometric signatures are most coherent in the middle layers (5-12) of the transformer. The geometry degrades significantly at the final output layer, where variance explodes.
3. **Compositionality**: Composite prompts (e.g., "Summarize + Criticize") activate the latent factors associated with their constituent single operators. The model's internal state for a mixed task is approximately the sum of the states for the individual tasks.

## Hypothesis Evaluation

- **H1 (Operator Signatures)**: **Supported**. Change-point detection consistently isolates operator activity, and PCA projections show clear separation between operator classes.
- **H2 (Geometric Compositionality)**: **Supported with Caveats**. Linear reconstruction explains the majority of the variance, though interaction effects exist.

## Implications for Cognitive Geometry

The results suggest that "thought" in a transformer is not a monolithic vector but a trajectory through a structured manifold. Different cognitive moves correspond to different "velocities" and "directions" in this space. The ability to decompose these trajectories implies that we could potentially intervene: steering a model's "thought process" by injecting the geometric signature of a specific operator (e.g., adding the "Criticize" vector to a "Summarize" process).

## Open Problems

- **Cross-Context Invariance**: Does the "Summarize" signature look the same regardless of the text being summarized?
- **Causal Intervention**: Can we induce the *behavior* of an operator solely by injecting its geometric signature?
- **Non-Linear Interaction**: Modeling the interaction terms that simpler linear superposition misses.
