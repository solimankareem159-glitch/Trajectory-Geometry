
# Experiment 18B: Scaling Geometry Audit Package

## Overview

This repository contains the complete replication materials for "Scaling Geometry: Emergent Attractor Dynamics in Large Language Models".
We provide a unified dataset of 57 geometric metrics computed across three models (Pythia 70m, Qwen 0.5B, Qwen 1.5B) on a standardized arithmetic reasoning task.

## Repository Structure

### Data & Results (`results/`)

- `qwen_0_5b_metrics_57.csv`: Full metric suite for Qwen 2.5 0.5B (24 layers).
- `qwen_1_5b_metrics_57.csv`: Full metric suite for Qwen 2.5 1.5B (28 layers).
- `pythia_70m_metrics_57.csv`: Full metric suite for Pythia 70m (6 layers).
- `reports/*.md`: Statistical analysis reports for each model.
- `figures/*.png`: Visualizations of metric scaling and distributions.

### Code (`scripts/`)

- `metric_suite.py`: The core library implementing the 57 metrics (Families 1-12).
- `run_18b_metrics.py`: The driver script used to generate the datasets.
- `analyze_18b_comparisons.py`: Script for reproducing intra-model statistical tests.
- `analyze_18b_cross_model.py`: Script for reproducing scaling laws figures.

### Documentation

- `METHODS_18B.md`: Detailed mathematical definitions of all 57 metrics.
- `manifest.json`: Links to raw hidden states and prompt files (see below).

## Provenance

This dataset aggregates results from three prior experiments:

1. **Experiment 14 (Qwen 0.5B)**: `experiments/EXP-14_UniversalSignature_2025-12-03`
2. **Experiment 16 (Pythia 70m)**: `experiments/EXP-16_Pythia70m_2025-12-07`
3. **Experiment 16B (Qwen 1.5B)**: `experiments/EXP-16B_Qwen15B_2025-12-08`

All hidden states were generated using the `exp9_dataset.jsonl` (300 arithmetic problems) found in `experiments/EXP-09_GeometryCapability_2025-11-22`.

## Reproduction

To reproduce the metrics from raw hidden states:

```bash
python experiments/EXP-18B_ScalingGeometry_2026-02-13/scripts/run_18b_metrics.py
```

To reproduce the analysis reports:

```bash
python experiments/EXP-18B_ScalingGeometry_2026-02-13/scripts/analyze_18b_comparisons.py
python experiments/EXP-18B_ScalingGeometry_2026-02-13/scripts/analyze_18b_cross_model.py
```

## Citation

[Placeholder for Arxiv Citation]
