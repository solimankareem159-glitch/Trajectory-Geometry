"""
Script 08: Report Compilation
=============================
Compiles all Analysis A-E results and figures into a single Markdown report.
"""

import os
import sys
import json
from datetime import datetime

# Import utils
sys.path.append(os.path.dirname(__file__))
from exp15_utils import EXP15_REPORTS_DIR, EXP15_DATA_DIR, EXP15_FIGURES_DIR

def main():
    print("="*60)
    print("08_report_compile.py: Generating Final Report")
    print("="*60)
    
    report_path = os.path.join(EXP15_REPORTS_DIR, "experiment15_report.md")
    
    # Load Run Metadata
    meta_path = os.path.join(EXP15_REPORTS_DIR, "run_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
    else:
        meta = {"error": "Metadata missing"}

    report = [
        "# Experiment 15: Deep Dive Analyses & New Geometric Signals",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Executive Summary",
        "This experiment implements the 'Deep Dive' roadmap, extending Experiment 14 with stratified analyses, failure subtyping, token dynamics, and response length controls.",
        "",
        "### Run Configuration",
        f"- **Torch**: {meta.get('torch_version', '?')}",
        f"- **DirectML**: {meta.get('directml_available', '?')} ({meta.get('directml_device', '?')})",
        f"- **Seed**: {meta.get('seed', '?')}",
        "",
        "## Analysis A: Difficulty x Geometry",
        "Investigating whether geometric signatures of reasoning vary by problem difficulty (Magnitude magnitude).",
        "",
        "### Key Visualizations",
        "![Heatmap Diff Rg](../figures/heatmap_difficulty_radius_of_gyration.png)",
        "![Heatmap Diff Dim](../figures/heatmap_difficulty_effective_dim.png)",
        "",
        "## Analysis B: Failure Subtyping",
        "Clustering CoT failures (G3) to identify distinct failure modes (e.g., 'Lost Wandering' vs 'Collapsed').",
        "",
        "### Cluster Visualizations",
        "![Cluster PCA](../figures/analysis_B_clusters_pca.png)",
        "",
        "## Analysis C: Token Dynamics",
        "Sliding-window analysis of geometric properties to detect phase transitions within trajectories.",
        "",
        "### Aggregate Dynamics",
        "![Dynamics](../figures/analysis_C_dynamics_aggregate.png)",
        "",
        "## Analysis D: Response Length Controls",
        "Evaluating if geometric metrics are merely proxies for response length.",
        "",
        "### Predictive Power (AUC)",
        "![Prediction](../figures/analysis_D_prediction_auc.png)",
        "",
        "## Analysis E: Direct-Only Successes",
        "Qualitative deep-dive into cases where Direct Answering succeeded but Chain-of-Thought failed.",
        f"See separate casebook: [Direct Only Successes](direct_only_successes.md)",
        "",
        "## New Geometric Signals",
        "Additional signals extracted.",
        "See `data/exp15_extra_metrics.csv` for raw data on curvature and cross-layer correlations.",
        "",
        "---",
        "**End of Report**"
    ]
    
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
        
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()
