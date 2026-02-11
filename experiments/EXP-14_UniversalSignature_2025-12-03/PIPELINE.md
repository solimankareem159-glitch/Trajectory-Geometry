# Experiment 14: Unified Pipeline Plan

## 1. Extraction Phase (`extract_hidden_states.py`)
**Goal**: Run model forward passes *once* and save all hidden states to disk.
- **Input**: `Experiment 9/data/exp9_dataset.jsonl`
- **Output**:
    - `experiments/Experiment 14/data/hidden_states/{id}_{cond}.npy`: Shape `[25, T, 896]` (Float16)
    - `experiments/Experiment 14/data/metadata.csv`: problem_id, condition, group, correct, filename
- **Volume**: ~600 files, ~840MB total.

## 2. Analysis Phase (`compute_metrics.py`)
**Goal**: Compute geometric metrics from saved states.
- **Input**: `.npy` files + `metadata.csv`
- **Metrics**: All layer-wise metrics + cross-layer metrics.
- **Optimization**: Use SVD for Anisotropy.
- **Output**: `experiments/Experiment 14/data/exp14_metrics.csv`

## 3. Reporting Phase (`generate_report.py`)
**Goal**: Generate tables, stats, and figures.
- **Input**: `exp14_metrics.csv`
- **Output**:
    - `experiments/Experiment 14/results/exp14_report.md`
    - `experiments/Experiment 14/figures/*.png`

## Execution Order
1. `python experiments/Experiment 14/scripts/extract_hidden_states.py`
2. `python experiments/Experiment 14/scripts/compute_metrics.py`
3. `python experiments/Experiment 14/scripts/generate_report.py`
