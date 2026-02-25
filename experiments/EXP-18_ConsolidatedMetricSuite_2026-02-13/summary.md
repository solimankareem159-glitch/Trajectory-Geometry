# Experiment 18: Consolidated Metric Suite

## Objective

To compute the definitive metric suite (54 metrics across 12 families) on the consolidated data from Experiment 9 (dataset) and Experiment 14 (hidden states).

## Data Sources

- **Dataset**: `experiments/EXP-09_GeometryCapability_2025-11-22/data/exp9_dataset.jsonl`
- **Hidden States**: `experiments/EXP-14_UniversalSignature_2025-12-03/data/hidden_states_full/`
- **Metadata**: `experiments/EXP-14_UniversalSignature_2025-12-03/data/metadata_full.csv` (Renamed to `metadata.csv`)

## Metrics

The metric suite includes:

1. **Kinematic**: Speed, Turn Angle, Tortuosity, etc.
2. **Volumetric**: Radius of Gyration, Effective Dimension, etc.
3. **Convergence**: Cosine Slope, Time to Commit, etc.
4. **Diffusion**: MSD Exponent.
5. **Spectral**: Spectral Entropy, PSD Slope.
6. **RQA**: Recurrence Rate, Determinism, etc.
7. **Cross-Layer**: Interlayer Alignment, Depth Acceleration.
8. **Landmark**: Logit Lens, Answer Logit Trajectory, etc.
9. **Attractor**: Distance to Success Centroid, etc.
10. **Embedding Stability**: Logit Consistency, etc.
11. **Information**: Step Surprisal, Entropy Rate.
12. **Inference**: Confidence Slope, Anomaly Score.

## methodology

The `metric_suite.py` script implements the comprehensive `TrajectoryMetrics` class. The `compute_metrics.py` script loads the data and applies the suite.
