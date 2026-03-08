# Contributing

Thank you for your interest in Trajectory Geometry.

## Replication

The most valuable contribution is **independent replication**. If you have access to models or architectures not yet tested (Gemma, Llama-3, Mamba, MoE), running the geometric analysis pipeline on your own data would significantly strengthen (or challenge) the findings.

Each experiment under `experiments/` is self-contained. See the [main README](README.md) for reproduction instructions.

## Reporting Issues

If you find errors in the analysis, statistical methodology, or code:

1. Open an issue describing the specific problem
2. Reference the experiment number and file path
3. If possible, include a minimal reproduction

## Extending the Work

Priority areas for extension:

- **New architectures** — Do geometric signatures hold for Mamba, MoE, or state-space models?
- **New tasks** — Do the findings generalize beyond arithmetic to logical reasoning, code generation, or natural language inference?
- **Causal intervention** — Can forcing geometric expansion (e.g., via activation steering) improve reasoning quality?
- **Scale** — Do signatures persist at 7B+ parameters?

## Code Style

- Python scripts should be self-contained within their experiment directory
- Use `requirements.txt` for dependencies
- Include clear docstrings for any new metric definitions

## Attribution

This research was conducted by a psychology researcher working with AI tools. If you build on this work, please cite using the [CITATION.cff](CITATION.cff) file.
