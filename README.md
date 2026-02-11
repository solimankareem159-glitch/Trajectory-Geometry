# Trajectory Geometry: Measuring the Shape of Reasoning in Transformers

When a transformer model reasons correctly, its hidden states trace geometrically distinct trajectories compared to when it fails. The effect sizes are massive (Cohen's d > 4.5 for dimensional collapse, > 17.0 for difficulty scaling), the patterns replicate across model scales (70M to 1.5B parameters), and geometry predicts correctness better than response length alone.

This repository is a complete audit trail of 17 experiments that discovered, validated, and stress-tested these findings.

## Key Findings

| Finding | Effect Size | Experiments |
|---------|------------|-------------|
| **Dimensional Collapse in Failure** — Failed reasoning collapses into a low-dimensional subspace (D_eff ~ 3 vs ~ 13) | d > 4.5 | 11, 12, 14 |
| **Regime-Relative Success** — "Good geometry" is opposite for CoT vs Direct answering; 10 of 14 metrics flip sign | d > 0.7 | 14 |
| **Difficulty-Driven Expansion** — Harder problems induce proportionally more geometric expansion | d > 17.0 | 15 |
| **Failure Subtypes** — Failures cluster into "Collapsed" (gave up) and "Wandering" (got lost) | Clear separation | 13 |
| **Commitment Timing** — Measurable phase transition from exploration to execution | Direct: ~5 tokens, CoT: ~11 tokens | 12, 14 |
| **Geometry Predicts Correctness** — Trajectory metrics outperform response length | AUC 0.898 (Direct) | 13, 15 |

All reported effects are statistically significant (p < 0.001, permutation testing, 10,000 shuffles, N = 300).

## The Research Story

This project began by searching for static "thought vectors" in language model representations. That search failed — seven experiments demonstrated that static operator signatures are too weak for practical use. The pivot to *dynamic* analysis — measuring how hidden states move rather than where they are — produced the findings above.

For the full narrative: **[`research_history/Journey/PROJECT_NARRATIVE.md`](research_history/Journey/PROJECT_NARRATIVE.md)**

## Repository Structure

### Core Research

```
experiments/                  # All 17 experiments with data, analysis, and figures
  EXP-01_GeometricSignatures/ through EXP-17_BaselineReplication/

experiment_summaries/         # Quick-reference summaries for each experiment

research_history/
  Journey/
    PROJECT_NARRATIVE.md      # Full research narrative
    Findings_Catalogue.md     # Empirical results reference
    Metrics_Appendix.md       # All 36 metric definitions
    Perplexity Deep Research.md  # Literature review with 185 citations
    TG medium_article_rewrite.md # Medium article draft
  Experimental_logs/          # Detailed per-experiment methodology logs
  Roadmap/                    # Research roadmap and dissemination strategy
  figures/                    # Publication-ready figures
```

### Tools and Scripts

```
visualisation_tool/           # Interactive 3D trajectory visualization (Three.js)
scripts/                      # Data processing and verification utilities
```

### Archive

```
archive/                      # Historical design documents and deep research notes
```

## Getting Started

### Prerequisites

- Python 3.10+
- A CUDA-capable GPU is recommended but not required (experiments were originally run on consumer hardware with DirectML)

### Installation

```bash
git clone https://github.com/<your-username>/dsg-experiments.git
cd dsg-experiments
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Reproducing an Experiment

Each experiment is self-contained under `experiments/EXP-<NN>_<Title>/`. The typical workflow is:

1. **Generate data** — Run the extraction script (e.g., `extract_hidden_states.py`) to collect hidden-state trajectories from the model. This downloads the model from Hugging Face on first run.
2. **Compute metrics** — Run the analysis script (e.g., `compute_metrics.py`) to calculate geometric and dynamical measures from the extracted trajectories.
3. **Generate figures** — Run any visualization scripts to reproduce the plots.

For example, to reproduce EXP-14 (Universal Signature):

```bash
cd experiments/EXP-14_UniversalSignature_2025-12-03/analysis
python extract_hidden_states.py
python compute_metrics.py
python generate_report.py
```

### Visualisation Tool

An interactive 3D trajectory viewer is included:

```bash
cd visualisation_tool
npm install
npm run dev
```

Then open `http://localhost:5173` in your browser.

## Data Availability

Large data files (hidden-state arrays stored as `.npy`/`.npz`, often 100MB+ per experiment) are excluded from this repository via `.gitignore`. All analysis scripts, computed metrics (CSVs), documentation, and figures are fully tracked. To regenerate the raw trajectory data, run the extraction scripts in each experiment's `analysis/` directory — this will download the relevant model from Hugging Face and re-extract hidden states locally.

The pre-computed visualization data (`visualisation_tool/data/trajectory_data.json`, ~36MB) is included in the repository for convenience. If you prefer not to clone this large file, use a shallow clone or sparse checkout.

## Experiment Index

| # | Title | Model | Verdict | Phase |
|---|-------|-------|---------|-------|
| 01 | Geometric Signatures (API) | Gemini | Invalid | Static |
| 02 | Latent Factors (NMF) | Qwen-0.5B | Invalid / Trivial | Static |
| 03 | Regime Invariants | Qwen-0.5B | Failed (confounded) | Static |
| 04 | Operator-Gated Multi-Pass | Qwen-0.5B | Success (+23%) | Intervention |
| 05 | Safety OG-MPT | Qwen-0.5B | Failed (capacity) | Intervention |
| 06 | Pilot Metric Validation | Qwen-0.5B | Inconclusive | Static |
| 07 | Static Operator Geometry | Qwen-0.5B | Valid but Insufficient | Static |
| 08 | Trajectory Geometry | Qwen-0.5B | Foundational Success | Dynamic |
| 09 | Geometry-Capability Correlation | Qwen-0.5B | **Breakthrough** | Dynamic |
| 09B | Cross-Model (TinyLlama) | TinyLlama-1.1B | Failed (capability floor) | Replication |
| 10 | Self-Report Consistency | Qwen-0.5B | Failed (no correlation) | Dynamic |
| 11 | Extended Geometric Suite | Qwen-0.5B | Success (D_eff discovery) | Dynamic |
| 12 | Advanced Diagnostics | Qwen-0.5B | Success (fractal dim) | Dynamic |
| 13 | Failure Subtyping | Qwen-0.5B | Success (AUC 0.898) | Dynamic |
| 14 | Universal Signature | Qwen-0.5B | **Breakthrough** (regime-relative) | Dynamic |
| 15 | Length Confound | Qwen-0.5B | Success (geometry > length) | Validation |
| 16/16B | Scale & Architecture | Qwen-1.5B, Pythia-70m | Success (replicated) | Validation |

## Models Tested

- **Qwen2.5-0.5B** — Primary model (all experiments)
- **Qwen2.5-1.5B** — Scale replication (EXP-16B)
- **Pythia-70m** — Architecture independence (EXP-16 salvage)
- **TinyLlama-1.1B** — Capability floor demonstration (EXP-09B)

All experiments run locally on personal hardware using open-weights models.

## Methodology

- **Task:** Multi-step arithmetic (e.g., `(47 * 3) + 12`) — chosen for unambiguous binary outcomes
- **Conditions:** Direct prompting vs Chain-of-Thought
- **Measurement:** Residual stream hidden states extracted across all transformer layers
- **Metrics:** 36 geometric/dynamical indicators (speed, curvature, effective dimension, radius of gyration, commitment timing, fractal dimension, recurrence quantification, and more)
- **Statistics:** Permutation testing (10,000 shuffles), p < 0.001, Cohen's d for effect sizes
- **Controls:** First-32-token window (EXP-01 to 13); full trajectory with controlled truncation (EXP-14+)

## Note on Collaboration

This research was conducted by a psychology researcher working with AI tools. The theoretical framework, experimental design, hypothesis generation, result interpretation, and course corrections were human contributions. Mathematical formalization, code implementation, and statistical computation were performed in collaboration with AI systems (Claude, Google Antigravity, ChatGPT). The findings stand on their statistical merits and are presented here for independent verification, replication, and critique.

## Status and Next Steps

The immediate priorities are cross-architecture replication (Gemma, Llama) and expansion beyond arithmetic to other reasoning domains. See [`research_history/Roadmap/`](research_history/Roadmap/) for the full roadmap.

Collaboration is welcome. If you have access to larger models or different architectures and want to test whether these geometric signatures replicate, please reach out.
