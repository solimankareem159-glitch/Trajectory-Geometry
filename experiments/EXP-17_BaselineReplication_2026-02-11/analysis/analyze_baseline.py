"""
EXP-17A: Comprehensive Baseline Analysis
=========================================
Compares trajectory geometry across groups for Qwen2.5-3B-Instruct.
Produces:
  - exp17a_comparisons.csv  (per-layer + cross-layer stats)
  - exp17a_baseline_report.md (narrative report)
  - Figures: layer evolution, heatmaps, regime classification
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

print(f"PID: {os.getpid()}", flush=True)

# ── Configuration ────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, '..', 'data')
FIGURES_DIR = os.path.join(BASE, '..', 'figures')
METRICS_FILE = os.path.join(DATA_DIR, 'exp17a_metrics.csv')
COMPARISONS_FILE = os.path.join(DATA_DIR, 'exp17a_comparisons.csv')
REPORT_FILE = os.path.join(DATA_DIR, 'exp17a_baseline_report.md')

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
N_LAYERS = 37  # layers 0–36

PER_LAYER_METRICS = [
    'speed', 'dir_consistency', 'stabilization', 'turning_angle', 'dir_autocorr',
    'tortuosity', 'effective_dim', 'cos_slope', 'dist_slope', 'early_late_ratio',
    'radius_of_gyration', 'gyration_anisotropy', 'drift_to_spread',
    'dir_autocorr_lag2', 'dir_autocorr_lag4', 'dir_autocorr_lag8',
    'vel_autocorr_lag1', 'vel_autocorr_lag2', 'vel_autocorr_lag4',
    'msd_exponent', 'cos_to_running_mean', 'cos_to_late_window', 'time_to_commit',
    'recurrence_rate', 'determinism', 'laminarity', 'trapping_time',
    'diagonal_entropy', 'psd_slope', 'spectral_entropy',
]
CROSS_LAYER_METRICS = ['interlayer_alignment_mean', 'depth_accel_speed', 'depth_accel_tortuosity']

KEY_METRICS_FOR_PLOTS = [
    'speed', 'effective_dim', 'tortuosity', 'dir_consistency',
    'msd_exponent', 'radius_of_gyration', 'drift_to_spread',
    'recurrence_rate', 'laminarity', 'psd_slope',
]

GROUP_COLORS = {'G1': '#e74c3c', 'G2': '#3498db', 'G3': '#f1c40f', 'G4': '#2ecc71'}
GROUP_LABELS = {
    'G1': 'Direct Fail', 'G2': 'Direct Success',
    'G3': 'CoT Fail', 'G4': 'CoT Success',
}

# Regime metrics to compare direction of effect against EXP-14
REGIME_METRICS = {
    'radius_of_gyration': 'Spatial extent of computation',
    'effective_dim': 'Dimensionality of trajectory',
    'msd_exponent': 'Diffusion regime',
    'time_to_commit': 'Explore → Commit transition',
    'dir_consistency': 'Directional stability',
    'tortuosity': 'Path complexity',
    'speed': 'Step magnitude per layer',
}

# EXP-14 reference signs (G4 minus G1): positive means G4 > G1
# These are from the Qwen 0.5B findings
EXP14_EXPECTED_SIGNS = {
    'radius_of_gyration': +1,   # G4 explores more
    'effective_dim': +1,        # G4 higher dimensionality
    'msd_exponent': +1,         # G4 more super-diffusive
    'time_to_commit': +1,       # G4 commits later
    'dir_consistency': -1,      # G4 less consistent (exploring)
    'tortuosity': +1,           # G4 more tortuous
    'speed': +1,                # G4 faster steps
}


# ── Statistics ───────────────────────────────────────────────

def permutation_test(a, b, k=1000):
    """Two-sided permutation test for difference in means."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 5 or len(b) < 5:
        return np.nan
    observed = np.mean(a) - np.mean(b)
    combined = np.concatenate([a, b])
    n_a = len(a)
    count = 0
    rng = np.random.default_rng(42)
    for _ in range(k):
        rng.shuffle(combined)
        perm_diff = np.mean(combined[:n_a]) - np.mean(combined[n_a:])
        if np.abs(perm_diff) >= np.abs(observed):
            count += 1
    return count / k


def cohens_d(a, b):
    """Pooled Cohen's d (a minus b)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
    s_pooled = np.sqrt(((len(a)-1)*var_a + (len(b)-1)*var_b) / (len(a)+len(b)-2))
    return (np.mean(a) - np.mean(b)) / s_pooled if s_pooled > 0 else 0.0


# ── Main ─────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("EXP-17A: Comprehensive Baseline Analysis")
    print(f"Model: {MODEL_NAME}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # ── Load Data ────────────────────────────────────────────
    if not os.path.exists(METRICS_FILE):
        print(f"ERROR: {METRICS_FILE} not found.")
        sys.exit(1)

    df = pd.read_csv(METRICS_FILE)
    print(f"Loaded {len(df)} rows from metrics CSV")

    df_layers = df[df['layer'] >= 0].copy()
    df_cross = df[df['layer'] == -1].copy()

    # ── Group Summary ────────────────────────────────────────
    groups_present = sorted(df['group'].unique())
    print(f"\nGroups present: {groups_present}")
    group_counts = df.groupby('group')['problem_id'].nunique()
    print(f"Samples per group:")
    for g, n in group_counts.items():
        label = GROUP_LABELS.get(g, g)
        print(f"  {g} ({label}): n={n}")

    # ── Build Group Pairs ────────────────────────────────────
    # Only compare groups that actually exist
    all_pairs = [
        ('G4', 'G1', 'CoT Success vs Direct Failure'),
        ('G4', 'G3', 'CoT Success vs CoT Failure'),
        ('G2', 'G1', 'Direct Success vs Direct Failure'),
        ('G4', 'G2', 'CoT Success vs Direct Success'),
        ('G3', 'G1', 'CoT Failure vs Direct Failure'),
    ]
    group_pairs = [(a, b, desc) for a, b, desc in all_pairs
                   if a in groups_present and b in groups_present]
    print(f"\nActive comparisons: {len(group_pairs)}")
    for a, b, desc in group_pairs:
        print(f"  {a} vs {b}: {desc}")

    # ── Per-Layer Comparisons ────────────────────────────────
    print("\nRunning per-layer comparisons (permutation tests, k=1000)...")
    layers = sorted(df_layers['layer'].unique().astype(int))
    available_metrics = [m for m in PER_LAYER_METRICS if m in df.columns]

    results = []
    total = len(group_pairs) * len(layers) * len(available_metrics)
    done = 0

    for ga, gb, desc in group_pairs:
        for layer in layers:
            da = df_layers[(df_layers['group'] == ga) & (df_layers['layer'] == layer)]
            db = df_layers[(df_layers['group'] == gb) & (df_layers['layer'] == layer)]

            for metric in available_metrics:
                va = da[metric].dropna().values
                vb = db[metric].dropna().values

                if len(va) < 5 or len(vb) < 5:
                    done += 1
                    continue

                d = cohens_d(va, vb)
                p = permutation_test(va, vb, k=1000)

                results.append({
                    'comparison': desc,
                    'g1': ga, 'g2': gb,
                    'layer': int(layer),
                    'metric': metric,
                    'g1_mean': np.mean(va),
                    'g2_mean': np.mean(vb),
                    'cohens_d': d,
                    'p_value': p,
                    'significant': p < 0.05 and abs(d) > 0.5,
                })
                done += 1
                if done % 500 == 0:
                    print(f"  Progress: {done}/{total} ({100*done/total:.0f}%)", flush=True)

    # ── Cross-Layer Comparisons ──────────────────────────────
    print("Running cross-layer comparisons...")
    available_cross = [m for m in CROSS_LAYER_METRICS if m in df.columns]

    for ga, gb, desc in group_pairs:
        da = df_cross[df_cross['group'] == ga]
        db = df_cross[df_cross['group'] == gb]

        for metric in available_cross:
            va = da[metric].dropna().values
            vb = db[metric].dropna().values

            if len(va) < 5 or len(vb) < 5:
                continue

            d = cohens_d(va, vb)
            p = permutation_test(va, vb, k=1000)

            results.append({
                'comparison': desc,
                'g1': ga, 'g2': gb,
                'layer': 'cross',
                'metric': metric,
                'g1_mean': np.mean(va),
                'g2_mean': np.mean(vb),
                'cohens_d': d,
                'p_value': p,
                'significant': p < 0.05 and abs(d) > 0.5,
            })

    comp_df = pd.DataFrame(results)
    comp_df.to_csv(COMPARISONS_FILE, index=False)
    print(f"\nSaved {len(comp_df)} comparison rows to {COMPARISONS_FILE}")

    # ── Figures ──────────────────────────────────────────────
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("\nGenerating figures...")

    plot_metrics = [m for m in KEY_METRICS_FOR_PLOTS if m in df.columns]

    # Fig 1: Layer Evolution
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    axes = axes.flatten()
    for idx, metric in enumerate(plot_metrics[:10]):
        ax = axes[idx]
        for grp in groups_present:
            means, sems = [], []
            for layer in layers:
                vals = df_layers[(df_layers['group'] == grp) & (df_layers['layer'] == layer)][metric].dropna()
                means.append(vals.mean() if len(vals) > 0 else np.nan)
                sems.append(vals.sem() if len(vals) > 1 else 0)
            means, sems = np.array(means), np.array(sems)
            ax.plot(layers, means, label=GROUP_LABELS.get(grp, grp),
                    color=GROUP_COLORS.get(grp, '#888'), linewidth=2)
            ax.fill_between(layers, means-sems, means+sems, alpha=0.15,
                            color=GROUP_COLORS.get(grp, '#888'))
        ax.set_title(metric.replace('_', ' ').title(), fontsize=10)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7)
    plt.suptitle(f'EXP-17A: Layer-Wise Metric Evolution ({MODEL_NAME})', fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'exp17a_layer_evolution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved exp17a_layer_evolution.png")

    # Fig 2: Cohen's d Heatmaps
    heatmap_files = []
    for ga, gb, desc in group_pairs:
        sub = comp_df[(comp_df['g1'] == ga) & (comp_df['g2'] == gb) &
                      (comp_df['layer'] != 'cross')]
        if sub.empty:
            continue
        pivot = sub.pivot(index='metric', columns='layer', values='cohens_d')
        # Sort by mean absolute effect
        order = pivot.abs().mean(axis=1).sort_values(ascending=False).index
        pivot = pivot.reindex(order)

        fig, ax = plt.subplots(figsize=(18, 12))
        sns.heatmap(pivot, cmap='RdBu_r', center=0, ax=ax,
                    cbar_kws={'label': "Cohen's d"}, vmin=-3, vmax=3)
        ax.set_title(f"Cohen's d: {desc} ({ga} vs {gb})")
        ax.set_xlabel('Layer')
        ax.set_ylabel('Metric')

        fname = f'exp17a_heatmap_{ga.lower()}_{gb.lower()}.png'
        plt.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=150, bbox_inches='tight')
        plt.close()
        heatmap_files.append((desc, fname))
        print(f"  Saved {fname}")

    # ── Regime Classification ────────────────────────────────
    print("\nClassifying regime...")
    regime_results = {}
    primary = comp_df[(comp_df['g1'] == 'G4') & (comp_df['g2'] == 'G1')]

    if len(primary) > 0:
        for metric, description in REGIME_METRICS.items():
            metric_data = primary[(primary['metric'] == metric) &
                                  (primary['layer'] != 'cross')]
            if metric_data.empty:
                regime_results[metric] = {'observed_sign': 0, 'mean_d': 0, 'n_sig': 0}
                continue

            mean_d = metric_data['cohens_d'].mean()
            n_sig = (metric_data['significant'] == True).sum()
            observed_sign = 1 if mean_d > 0 else (-1 if mean_d < 0 else 0)

            expected = EXP14_EXPECTED_SIGNS.get(metric, 0)
            matches = observed_sign == expected

            regime_results[metric] = {
                'description': description,
                'mean_d': mean_d,
                'n_sig': n_sig,
                'n_layers': len(metric_data),
                'observed_sign': observed_sign,
                'expected_sign': expected,
                'matches_exp14': matches,
            }

        # Overall regime classification
        match_count = sum(1 for v in regime_results.values() if v.get('matches_exp14'))
        total_regime = len(regime_results)
        if match_count >= total_regime * 0.7:
            regime_label = "EXPANSION (consistent with EXP-14)"
        elif match_count <= total_regime * 0.3:
            regime_label = "COMPRESSION (inverted from EXP-14)"
        else:
            regime_label = "MIXED / NOVEL"
    else:
        regime_label = "INSUFFICIENT DATA"

    print(f"  Regime: {regime_label}")

    # ── Generate Report ──────────────────────────────────────
    print("\nWriting report...")
    report = generate_report(
        comp_df, group_counts, groups_present, regime_results, regime_label,
        heatmap_files
    )
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Report saved to {REPORT_FILE}")
    print("\nDone.")


def generate_report(comp_df, group_counts, groups_present, regime_results,
                    regime_label, heatmap_files):
    lines = []
    lines.append(f"# EXP-17A: Baseline Replication Report")
    lines.append(f"")
    lines.append(f"**Model:** {MODEL_NAME}")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Layers:** {N_LAYERS} (0–36)")
    lines.append(f"")

    # Group summary
    lines.append("## Group Distribution")
    lines.append("")
    lines.append("| Group | Label | N (samples) |")
    lines.append("|---|---|---|")
    for g in ['G1', 'G2', 'G3', 'G4']:
        n = group_counts.get(g, 0)
        label = GROUP_LABELS.get(g, '?')
        marker = " ⚠️ **MISSING**" if n == 0 else ""
        lines.append(f"| {g} | {label} | {n}{marker} |")
    lines.append("")

    if 'G2' not in groups_present:
        lines.append("> [!WARNING]")
        lines.append("> **No Direct Success (G2) group exists.** Qwen2.5-3B-Instruct fails ALL 300 direct-answer problems.")
        lines.append("> This suggests the model requires Chain-of-Thought prompting for arithmetic, or the parsing heuristic is too strict.")
        lines.append("")

    # Regime Classification
    lines.append("## Regime Classification")
    lines.append("")
    lines.append(f"**Observed Regime: {regime_label}**")
    lines.append("")
    lines.append("| Metric | Description | Mean Cohen's d | Sig. Layers | G4 Direction | EXP-14 Expected | Match? |")
    lines.append("|---|---|---|---|---|---|---|")
    for metric, info in regime_results.items():
        if isinstance(info, dict) and 'description' in info:
            d_val = info['mean_d']
            direction = "↑ Higher" if info['observed_sign'] > 0 else ("↓ Lower" if info['observed_sign'] < 0 else "→ No diff")
            expected_dir = "↑" if info['expected_sign'] > 0 else "↓"
            match = "✅" if info.get('matches_exp14') else "❌"
            lines.append(f"| `{metric}` | {info['description']} | {d_val:+.3f} | {info['n_sig']}/{info['n_layers']} | {direction} | {expected_dir} | {match} |")
    lines.append("")

    # Primary comparison: G4 vs G1
    lines.append("## Primary Comparison: G4 (CoT Success) vs G1 (Direct Fail)")
    lines.append("")

    primary = comp_df[(comp_df['g1'] == 'G4') & (comp_df['g2'] == 'G1')]
    if len(primary) > 0:
        sig = primary[primary['significant'] == True].sort_values('cohens_d', key=abs, ascending=False)

        lines.append(f"Total comparisons: {len(primary)}")
        lines.append(f"Significant (p<0.05, |d|>0.5): {len(sig)}")
        lines.append("")

        lines.append("### Top 20 Discriminators")
        lines.append("")
        lines.append("| Layer | Metric | Cohen's d | G4 Mean | G1 Mean | p-value |")
        lines.append("|---|---|---|---|---|---|")
        for _, row in sig.head(20).iterrows():
            lines.append(f"| {row['layer']} | `{row['metric']}` | {row['cohens_d']:+.3f} | {row['g1_mean']:.4f} | {row['g2_mean']:.4f} | {row['p_value']:.3f} |")
    else:
        lines.append("No data for G4 vs G1 comparison.")
    lines.append("")

    # G4 vs G3
    g4g3 = comp_df[(comp_df['g1'] == 'G4') & (comp_df['g2'] == 'G3')]
    if len(g4g3) > 0:
        lines.append("## G4 (CoT Success) vs G3 (CoT Fail)")
        lines.append("")
        sig = g4g3[g4g3['significant'] == True].sort_values('cohens_d', key=abs, ascending=False)
        lines.append(f"Significant: {len(sig)}")
        lines.append("")
        if len(sig) > 0:
            lines.append("### Top Discriminators")
            lines.append("")
            lines.append("| Layer | Metric | Cohen's d | G4 Mean | G3 Mean | p-value |")
            lines.append("|---|---|---|---|---|---|")
            for _, row in sig.head(15).iterrows():
                lines.append(f"| {row['layer']} | `{row['metric']}` | {row['cohens_d']:+.3f} | {row['g1_mean']:.4f} | {row['g2_mean']:.4f} | {row['p_value']:.3f} |")
        lines.append("")

    # G3 vs G1
    g3g1 = comp_df[(comp_df['g1'] == 'G3') & (comp_df['g2'] == 'G1')]
    if len(g3g1) > 0:
        lines.append("## G3 (CoT Fail) vs G1 (Direct Fail)")
        lines.append("")
        sig = g3g1[g3g1['significant'] == True].sort_values('cohens_d', key=abs, ascending=False)
        lines.append(f"Significant: {len(sig)}")
        lines.append("")
        if len(sig) > 0:
            lines.append("### Top Discriminators")
            lines.append("")
            lines.append("| Layer | Metric | Cohen's d | G3 Mean | G1 Mean | p-value |")
            lines.append("|---|---|---|---|---|---|")
            for _, row in sig.head(15).iterrows():
                lines.append(f"| {row['layer']} | `{row['metric']}` | {row['cohens_d']:+.3f} | {row['g1_mean']:.4f} | {row['g2_mean']:.4f} | {row['p_value']:.3f} |")
        lines.append("")

    # Figures
    lines.append("## Visualizations")
    lines.append("")
    lines.append("### Layer Evolution")
    lines.append("![Layer Evolution](../figures/exp17a_layer_evolution.png)")
    lines.append("")
    lines.append("### Cohen's d Heatmaps")
    lines.append("")
    for desc, fname in heatmap_files:
        lines.append(f"**{desc}**")
        lines.append(f"![{desc}](../figures/{fname})")
        lines.append("")

    # Interpretation
    lines.append("## Interpretation Notes")
    lines.append("")
    lines.append("- G4 (n=55) is small; effect sizes may be unstable. Interpret cautiously.")
    lines.append("- The absence of G2 suggests Qwen 3B cannot perform multi-step arithmetic via direct retrieval.")
    lines.append("- Compare regime classification against EXP-14 (Qwen 0.5B) and EXP-16B (Qwen 1.5B) for scale stability.")
    lines.append("")

    return '\n'.join(lines)


if __name__ == "__main__":
    main()
