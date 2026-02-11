# Experiment 1: Geometric Signatures (Turn-Level)

This experiment uses the Gemini API to analyze how the "warp" (distance between embedding states) evolves over the course of a conversation.

## Directory Structure

- **`data/`**: JSONL datasets of prompts and turns, plus Parquet files of computed states and warps.
  - `operator_prompts.jsonl`: The input prompts.
  - `warps.parquet`: The computed warp traces.
- **`scripts/`**: Python scripts for data generation and analysis.
  - `geometric_signatures.ipynb`: The main notebook prototyping the concept.
  - `cluster_warps.py`: Performs clustering on the warp traces.
- **`results/`**: Figures and Markdown reports.
  - `CLUSTER_REPORT.md`: Analysis of how well operators cluster.
  - `ROBUSTNESS_REPORT.md`: Robustness of the geometric signals.

## Key Findings

- Conversational operators produced distinguishable "warp signatures" in the embedding space of `text-embedding-004`.
- These signatures were robust enough to predict the next operator in a sequence.
