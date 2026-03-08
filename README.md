# Trajectory Geometry: Measuring the Shape of Reasoning in Transformers

When a transformer model reasons correctly, its hidden states trace geometrically distinct trajectories compared to when it fails. This repository is a complete audit trail of 19 experiments that discovered, validated, and stress-tested these findings across multiple architectures and scales. The effect sizes are massive, the patterns replicate, and geometry predicts correctness better than response length alone.

## Key Findings

| Finding | Effect Size | Experiments |
|---------|------------|-------------|
| **Dimensional Collapse in Failure** — Failed reasoning collapses into a low-dimensional subspace (D_eff ~ 3 vs ~ 13) | d > 4.5 | 11, 12, 14 |
| **Regime-Relative Success** — "Good geometry" is opposite for CoT vs Direct answering; 10 of 14 metrics flip sign | d > 0.7 | 14 |
| **Difficulty-Driven Expansion** — Harder problems induce proportionally more geometric expansion | d > 17.0 | 15 |
| **Failure Subtypes** — Failures cluster into "Collapsed" (gave up) and "Wandering" (got lost) | Clear separation | 13 |
| **Commitment Timing** — Measurable phase transition from exploration to execution | Direct: ~5 tokens, CoT: ~11 tokens | 12, 14 |
| **Geometry Predicts Correctness** — Trajectory metrics outperform response length | AUC 0.898 (Direct) | 13, 15 |
| **Architecture Invariance** — 19 geometric signatures hold across Qwen and Pythia families | Confirmed | 19 |
| **Within-Regime Prediction** — Geometry predicts success even controlling for CoT vs Direct | AUC 0.78 | 19B |

All reported effects are statistically significant (p < 0.001, permutation testing, 10,000 shuffles).

## Quick Start

### Prerequisites

- Python 3.10+
- A CUDA-capable GPU is recommended but not required (experiments were run on consumer hardware with AMD DirectML)

### Installation

```bash
git clone https://github.com/solimankareem159-glitch/Trajectory-Geometry.git
cd Trajectory-Geometry
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Reproducing an Experiment

Each experiment is self-contained under `experiments/EXP-<NN>_<Title>/`. The typical workflow:

1. **Generate data** — Run the extraction script to collect hidden-state trajectories from the model (downloads from Hugging Face on first run)
2. **Compute metrics** — Run the analysis script to calculate geometric measures
3. **Generate figures** — Run visualization scripts to reproduce plots

Example (EXP-14, the regime-relative geometry breakthrough):

```bash
cd experiments/EXP-14_UniversalSignature_2025-12-03/analysis
python extract_hidden_states.py
python compute_metrics.py
python generate_report.py
```

## Repository Structure

```
README.md                          # This file
CITATION.cff                       # Citation metadata
CONTRIBUTING.md                    # Replication and contribution guide
LICENSE                            # Apache 2.0
requirements.txt                   # Python dependencies

docs/                              # Research documentation
  research_narrative.md            # Full 19-experiment research story
  findings_catalogue.md            # Empirical results reference
  metric_definitions.md            # Definitive 54-metric suite (12 families)
  metrics_appendix.md              # Supplementary metric definitions
  literature_review.md             # Literature review (185 citations)

experiments/                       # All 19 experiments with data, scripts, and figures
  INDEX.md                         # Master experiment index
  EXP-01_GeometricSignatures/      # through EXP-19_Robustness/

experiment_summaries/              # One-page summary per experiment
  EXP-01_GeometricSignatures.md    # through EXP-19_Robustness.md

figures/                           # Publication-ready figures

scripts/                           # Cross-experiment analysis utilities

visualisation_tool/                # Interactive 3D trajectory viewer (React/Three.js)

archive/                           # Historical design documents and protocols
```

## Experiment Index

| # | Title | Model | Verdict | Phase |
|---|-------|-------|---------|-------|
| 01 | Geometric Signatures (API) | Gemini | Invalid | I — Shapes |
| 02 | Latent Factors (NMF) | Qwen-0.5B | Invalid / Trivial | I — Shapes |
| 03 | Regime Invariants | Qwen-0.5B | Failed (confounded) | II — Invariants |
| 04 | Operator-Gated Multi-Pass | Qwen-0.5B | Success (+23%) | II — Detour |
| 05 | Safety OG-MPT | Qwen-0.5B | Failed (capacity) | II — Detour |
| 06 | Pilot Metric Validation | Qwen-0.5B | Inconclusive | II — Wilderness |
| 07 | Static Operator Geometry | Qwen-0.5B | Valid but Insufficient | II — Ceiling |
| 08 | Trajectory Geometry | Qwen-0.5B | Foundational Success | III — Dynamics |
| 09 | Geometry-Capability Correlation | Qwen-0.5B | **Breakthrough** | III — Dynamics |
| 09B | Cross-Model (TinyLlama) | TinyLlama-1.1B | Failed (capability floor) | III — Dynamics |
| 10 | Self-Report Consistency | Qwen-0.5B | Failed (no correlation) | III — Dynamics |
| 11 | Extended Geometric Suite | Qwen-0.5B | Success (D_eff discovery) | IV — Failure Analysis |
| 12 | Advanced Diagnostics | Qwen-0.5B | Success (fractal dim) | IV — Failure Analysis |
| 13 | Failure Subtyping | Qwen-0.5B | Success (AUC 0.898) | IV — Failure Analysis |
| 14 | Universal Signature | Qwen-0.5B | **Breakthrough** (regime-relative) | V — Paradigm |
| 15 | Length Confound | Qwen-0.5B | Success (geometry > length) | V — Validation |
| 16/16B | Scale & Architecture | Qwen-1.5B, Pythia-70m | Success (replicated) | V — Scale |
| 17 | Baseline Replication (3B) | Qwen-3B | Success | VI — Robustness |
| 18 | Consolidated Metric Suite | Qwen-0.5B | Success (54 metrics) | VI — Robustness |
| 18B | Scaling Geometry | Multi-model | Partial (Pythia data corrupted) | VI — Robustness |
| 19 | Robustness Replication | Qwen-0.5B/1.5B, Pythia-410m | **Replication Confirmed** | VI — Robustness |

For detailed per-experiment methodology and results, see [`experiment_summaries/`](experiment_summaries/).

## The Research Story

This project began by searching for static "thought vectors" in language model representations. That search failed — seven experiments demonstrated that static operator signatures are too weak for practical use. The pivot to *dynamic* analysis — measuring how hidden states move rather than where they are — produced all the findings above.

For the full narrative: [`docs/research_narrative.md`](docs/research_narrative.md)

## Models Tested

| Model | Parameters | Architecture | Experiments |
|-------|-----------|--------------|-------------|
| Gemini API (`text-embedding-004`) | — | Closed | 01 |
| Qwen2.5-0.5B | 500M | LLaMA-style | 02–19 (primary) |
| Qwen2.5-1.5B | 1.5B | LLaMA-style | 16B, 19 |
| Qwen2.5-3B-Instruct | 3B | LLaMA-style | 17 |
| TinyLlama-1.1B-Chat | 1.1B | LLaMA | 09B |
| Pythia-70m | 70M | GPT-style | 16 |
| Pythia-410m | 410M | GPT-style | 19 |

All experiments run locally on consumer hardware (AMD RX 5700 XT, 8GB VRAM, DirectML) using open-weights models.

## Methodology

- **Task:** Multi-step arithmetic (e.g., `(47 * 3) + 12`) — chosen for unambiguous binary correctness
- **Conditions:** Direct prompting vs Chain-of-Thought
- **Measurement:** Residual stream hidden states extracted across all transformer layers
- **Metrics:** 54 geometric/dynamical indicators across 12 families (kinematic, volumetric, convergence, diffusion, spectral, RQA, cross-layer, landmark, attractor, embedding stability, information, inference)
- **Statistics:** Permutation testing (10,000 shuffles), p < 0.001, Cohen's d for effect sizes
- **Controls:** First-32-token window (EXP-01 to 13); full trajectory with controlled truncation (EXP-14+)

For complete metric definitions: [`docs/metric_definitions.md`](docs/metric_definitions.md)

## Data Availability

Large data files (hidden-state arrays stored as `.npy`/`.npz`, often 100MB+ per experiment) are excluded from this repository. All analysis scripts, computed metrics (CSVs), documentation, and figures are fully tracked.

To regenerate raw trajectory data, run the extraction scripts in each experiment's `analysis/` directory — this downloads the relevant model from Hugging Face and re-extracts hidden states locally.

## Visualisation Tool

An interactive 3D trajectory viewer is included:

```bash
cd visualisation_tool
npm install
npm run dev
```

Then open `http://localhost:5173`. Trajectories are color-coded by diffusion regime (sub-diffusive, Brownian, super-diffusive, hyper-diffusive).

## Documentation

| Document | Description |
|----------|-------------|
| [`docs/research_narrative.md`](docs/research_narrative.md) | Full 19-experiment research story with phase transitions |
| [`docs/findings_catalogue.md`](docs/findings_catalogue.md) | Comprehensive empirical results reference |
| [`docs/metric_definitions.md`](docs/metric_definitions.md) | Definitive 54-metric suite across 12 families |
| [`docs/metrics_appendix.md`](docs/metrics_appendix.md) | Supplementary metric definitions and formulas |
| [`docs/literature_review.md`](docs/literature_review.md) | Literature review with 185 citations |

## Note on Collaboration

This research was conducted by a psychology researcher working with AI tools. The theoretical framework, experimental design, hypothesis generation, result interpretation, and course corrections were human contributions. Mathematical formalization, code implementation, and statistical computation were performed in collaboration with AI systems (Claude, ChatGPT). The findings stand on their statistical merits and are presented here for independent verification, replication, and critique.

## Citation

```bibtex
@software{soliman2026trajectory,
  author = {Soliman, Kareem},
  title = {Trajectory Geometry: Measuring the Shape of Reasoning in Transformers},
  year = {2026},
  url = {https://github.com/solimankareem159-glitch/Trajectory-Geometry}
}
```

## License

Apache 2.0. See [LICENSE](LICENSE).
