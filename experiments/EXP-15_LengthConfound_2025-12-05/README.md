# Experiment 15: Deep Dive Analyses & New Geometric Signals

This experiment package implements the "Deep Dive" roadmap, extending Experiment 14 with stratified analyses, failure subtyping, token dynamics, response length controls, and additional geometric feature extraction.

## Quick Start

Run the full pipeline:
```powershell
.\run_all.ps1
```
Or with Python:
```bash
python run_all.py
```

## Folder Structure

- `data/`: Intermediate CSVs and unified datasets.
- `figures/`: Generated plots (Heatmaps, PCA clusters, etc.).
- `reports/`: Final markdown reports and run metadata.
- `scripts/`: Sequential analysis scripts.

## Scripts Overview

| Script | Purpose | Output |
| :--- | :--- | :--- |
| `00_env_check.py` | verifies DirectML/GPU/CPU context | `reports/run_metadata.json` |
| `01_ingest_and_join.py` | Joins Exp 9 (text) + Exp 14 (metrics) | `data/unified_metrics.csv` |
| `02_analysis_A...` | Difficulty x Geometry stratification | `analysis_A_difficulty_results.csv`, Heatmaps |
| `03_analysis_B...` | Failure Subtyping (Clustering G3) | `analysis_B_clusters_pca.png`, Centroids CSV |
| `04_analysis_C...` | Token-level sliding window dynamics | `analysis_C_window_metrics.csv`, Dynamics Plot |
| `05_analysis_D...` | Response Length controls/prediction | `analysis_D_prediction.csv`, AUC Plot |
| `06_analysis_E...` | Direct-Only Successes Casebook | `reports/direct_only_successes.md` |
| `07_new_signals...` | Extracts Curvature, Correlations, etc. | `data/exp15_extra_metrics.csv` |
| `08_report_compile.py` | Compiles all findings into final report | `reports/experiment15_report.md` |

## Inputs
- **Experiment 14 Metrics**: `../Experiment 14/data/exp14_metrics.csv`
- **Experiment 9 Dataset**: `../Experiment 9/data/exp9_dataset.jsonl`
- **Hidden States**: `../Experiment 14/data/hidden_states/*.npy` (for scripts 04 and 07)

## Outputs
The main deliverable is `reports/experiment15_report.md`, which summarizes all findings with embedded figures.
transient data is saved in `data/`.

## Configuration
All scripts use `scripts/exp15_utils.py` for shared configuration (paths, device selection).
Edit `exp15_utils.py` to change root directories if needed.
