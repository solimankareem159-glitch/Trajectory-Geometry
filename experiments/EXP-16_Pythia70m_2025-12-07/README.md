# Experiment 16: Cross-Architecture Replication (Qwen 1.5B & Pythia 70m)

## Overview
Replicates Experiment 14 findings using EleutherAI/pythia-70m to test the architecture-independence of geometric reasoning signatures identified in previous experiments.

## Quick Start

### 1. Installation
```bash
pip install -r requirements.txt

# Optional: DirectML for AMD GPU acceleration
pip install torch-directml
```

### 2. Run Full Pipeline
```bash
# Python (cross-platform)
python run_all.py

# PowerShell (Windows)
.\run_all.ps1
```

### 3. Run Individual Scripts
```bash
python scripts/00_verify_inputs.py
python scripts/01_env_and_device.py
python scripts/02_preflight_20q.py
python scripts/03_run_inference_dump_hidden.py  # --save_hidden_states=1 to save
python scripts/04_compute_metrics.py
python scripts/05_stats_and_tests.py
python scripts/06_make_figures.py
python scripts/07_write_report.py
```

## Folder Structure
```
Experiment 16/
├── data/
│   ├── metadata.csv                    # Full inference results
│   ├── exp16_metrics.csv               # Computed geometric metrics
│   ├── exp16_comparisons.csv           # Statistical test results
│   ├── checkpoints/                    # Resumable checkpoints (every 50 problems)
│   └── hidden_states/                  # Optional: raw hidden states
├── figures/
│   ├── heatmap_g4_vs_g3.png
│   ├── heatmap_g2_vs_g1.png
│   └── accuracy_by_condition.png
├── scripts/
│   ├── 00_verify_inputs.py
│   ├── 01_env_and_device.py
│   ├── 02_preflight_20q.py
│   ├── 03_run_inference_dump_hidden.py
│   ├── 04_compute_metrics.py
│   ├── 05_stats_and_tests.py
│   ├── 06_make_figures.py
│   └── 07_write_report.py
├── run_all.py
├── run_all.ps1
├── requirements.txt
├── README.md
└── Experiment_16_report.md            # Final report
```

## Pipeline Stages

### Phase 0: Verification
- **00_verify_inputs.py**: Validates exp9_dataset.jsonl schema
- **01_env_and_device.py**: DirectML/CUDA/CPU device setup

### Phase 1: Inference
- **02_preflight_20q.py**: Capability test on 20 problems
- **03_run_inference_dump_hidden.py**: Full 300-problem inference
  - Checkpoints every 50 problems
  - Resumable with `--resume`
  - Optional hidden state saving with `--save_hidden_states=1`

### Phase 2: Analysis
- **04_compute_metrics.py**: DirectML-accelerated metric computation
- **05_stats_and_tests.py**: 6 key comparisons with permutation tests

### Phase 3: Reporting
- **06_make_figures.py**: Heatmaps and visualizations
- **07_write_report.py**: Comprehensive markdown report

## Key Outputs

### Metadata (`metadata.csv`)
Per-problem inference results with parsed answers and correctness.

### Metrics (`exp16_metrics.csv`)
Per-layer geometric metrics:
- speed, dir_consistency, stabilization
- effective_dim, radius_of_gyration
- msd_exponent, cos_to_late_window
- interlayer_alignment (cross-layer)

### Comparisons (`exp16_comparisons.csv`)
Statistical tests for 6 group comparisons:
1. G4 vs G3 (CoT Success vs CoT Failure)
2. G4 vs G2 (CoT Success vs Direct Success)
3. G4 vs G1 (CoT Success vs Direct Failure)
4. G2 vs G1 (Direct Success vs Direct Failure)
5. G3 vs G1 (CoT Failure vs Direct Failure)
6. G3 vs G2 (CoT Failure vs Direct Success)

## Configuration
- **Model**: EleutherAI/pythia-70m
- **Dataset**: 300 arithmetic problems from Experiment 9
- **Seed**: 42 (deterministic)
- **Device**: DirectML > CUDA > CPU (automatic selection)
- **Precision**: float16 (GPU) or float32 (CPU)

## Resume from Checkpoint
If inference is interrupted:
```bash
python scripts/03_run_inference_dump_hidden.py --resume
```

## Troubleshooting

### Low Accuracy
Pythia-70m has limited arithmetic capability. Low accuracy (<10%) is expected but does not invalidate geometric analysis.

### DirectML Issues
If DirectML causes errors, it will automatically fall back to CPU. Check `data/env_info.json` for device status.

### Missing Hidden States
Hidden states are only saved if `--save_hidden_states=1` is specified AND sufficient disk space is available.

## Citation
Part of the Dynamic Semantic Geometry research project investigating geometric structure in transformer reasoning trajectories.
