# Experiment 2: Latent Factor Decomposition (Token-Level)

This experiment uses a local transformer (`Qwen/Qwen2.5-0.5B`) to extract internal hidden states at every token step. It aims to decompose the "trace" of an operator into latent geometric components.

## Directory Structure

- **`data/`**: Input prompt configurations.
  - `composite_prompts.json`: Mixed operators for decomposition testing.
- **`scripts/`**: Analysis logic.
  - `extract_internals.py`: Basic extraction of warp traces.
  - `decompose_operator_factors.py`: Applies NMF to find latent factors.
  - `run_paper2_ablations.py`: Robustness testing (k-factors, layers).
- **`results/`**: The full research report package.
  - **`PAPER2_SUMMARY.md`**: Executive summary.
  - **`figures_factors/`**: Visualizations of the latent factors and reconstructions.
  - **`figures_paper2/`**: Ablation plots.

## Key Findings

- "Operators" are not just static states but trajectories (curves) in latent space.
- Composite operators (e.g., "Summarize + Criticize") can be linearly decomposed into the factors of their components (Linear Superposition).
- This geometry is most stable in the middle layers of the transformer.
