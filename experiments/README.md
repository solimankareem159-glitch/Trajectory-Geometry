# Experiments

This directory contains all 17 experiments in the Trajectory Geometry research project, organized chronologically with standardized naming.

**Format:** `EXP-<NN>_<ShortTitle>_<YYYY-MM-DD>`

Each experiment folder follows a standardized structure:

```
EXP-<NN>_<Title>_<Date>/
├── data/           # Raw and processed experimental outputs
├── analysis/       # Scripts, notebooks, and results
├── figures/        # Visualizations and plots
├── README.md       # Experiment overview
├── metadata.md     # Full experimental metadata
└── summary.md      # Concise summary
```

See the main [README](../README.md) for the full experiment index and project context.

See [`/experiment_summaries/`](../experiment_summaries/) for quick-reference summaries.

## Note on Data Files

Large data files (hidden state arrays as `.npy`/`.npz`) are excluded from version control via `.gitignore` due to their size (often 100MB+ per experiment). The analysis scripts, metrics CSVs, and documentation are fully tracked. To reproduce experiments, run the data generation scripts in each experiment's `analysis/` directory.
