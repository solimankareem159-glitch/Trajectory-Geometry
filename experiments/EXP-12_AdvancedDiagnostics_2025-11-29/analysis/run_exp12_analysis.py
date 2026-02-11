"""
Experiment 12: Advanced Trajectory Diagnostics
New metrics from updated roadmap P1.1-P1.4:
- P1.1: Fractal Dimension (Higuchi method on step norms)
- P1.2: Intrinsic Dimensionality (MLE-based estimator)
- P1.3: Convergence-to-Final-State (cosine/distance to h_T)
- P1.4: Recurrence/Loopiness (recurrence rate, determinism)

All metrics computed from existing Experiment 9 hidden states.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATA_DIR = r"experiments/Experiment 9/data"
DATA_FILENAME = "exp9_dataset.jsonl"
OUTPUT_DIR = r"experiments/Experiment 12/results"
REPORT_FILENAME = "exp12_advanced_diagnostics_report.md"

LAYERS = [0, 10, 13, 16, 24]
WINDOW_SIZE = 32

def load_data():
    path = os.path.join(DATA_DIR, DATA_FILENAME)
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

# --- P1.1: Fractal Dimension ---

def higuchi_fd(series, kmax=8):
    """Higuchi fractal dimension on 1D time series."""
    N = len(series)
    if N < kmax + 2:
        return 1.0  # Default for very short series
    
    L = []
    for k in range(1, min(kmax + 1, N // 2)):
        Lk = 0
        for m in range(1, k + 1):
            n_intervals = int((N - m) / k)
            if n_intervals < 1:
                continue
            Lmk = sum(abs(series[m + i*k - 1] - series[m + (i-1)*k - 1])
                      for i in range(1, n_intervals + 1))
            Lmk = Lmk * (N - 1) / (n_intervals * k * k)
            Lk += Lmk
        if k > 0:
            L.append(Lk / k)
    
    if len(L) < 2:
        return 1.0
    
    # Linear fit in log-log space
    x = np.log(1.0 / np.arange(1, len(L) + 1))
    y = np.log(np.array(L) + 1e-10)
    
    try:
        slope, _ = np.polyfit(x, y, 1)
        return float(np.clip(slope, 1.0, 2.0))
    except:
        return 1.0

def compute_fractal_dim(hidden_states):
    """Fractal dimension from step-norm time series."""
    h = hidden_states
    if len(h) < 5:
        return 1.0
    
    delta = h[1:] - h[:-1]
    norms = np.linalg.norm(delta, axis=1)
    
    return higuchi_fd(norms)

# --- P1.2: Intrinsic Dimensionality ---

def mle_intrinsic_dim(points, k=5):
    """MLE-based intrinsic dimensionality estimator (Levina & Bickel)."""
    n_points = len(points)
    if n_points < k + 2:
        return 0.0
    
    # Compute pairwise distances
    D = squareform(pdist(points))
    
    # For each point, get k nearest neighbor distances
    id_estimates = []
    for i in range(n_points):
        dists = np.sort(D[i])[1:k+1]  # Exclude self
        if dists[-1] < 1e-10:
            continue
        
        # MLE estimate
        log_ratios = np.log(dists[-1] / (dists[:-1] + 1e-10))
        if np.sum(log_ratios) > 0:
            id_est = (k - 1) / np.sum(log_ratios)
            if 0 < id_est < 1000:  # Sanity bounds
                id_estimates.append(id_est)
    
    if len(id_estimates) < 3:
        return 0.0
    
    return float(np.median(id_estimates))

def compute_intrinsic_dim(hidden_states, k=5):
    """Intrinsic dimensionality of the trajectory point cloud."""
    h = hidden_states
    if len(h) < k + 2:
        return 0.0
    
    return mle_intrinsic_dim(h, k=min(k, len(h) - 2))

# --- P1.3: Convergence-to-Final-State ---

def compute_convergence_metrics(hidden_states):
    """Convergence to final state: cosine and distance metrics."""
    h = hidden_states
    if len(h) < 4:
        return {"cos_slope": 0.0, "dist_slope": 0.0, "early_late_ratio": 1.0}
    
    h_final = h[-1]
    
    # Cosine similarity to final state
    cos_to_final = []
    for t in range(len(h) - 1):
        norm_t = np.linalg.norm(h[t])
        norm_f = np.linalg.norm(h_final)
        if norm_t > 1e-10 and norm_f > 1e-10:
            cos = np.dot(h[t], h_final) / (norm_t * norm_f)
            cos_to_final.append(cos)
        else:
            cos_to_final.append(0.0)
    
    # Distance to final state
    dist_to_final = [np.linalg.norm(h[t] - h_final) for t in range(len(h) - 1)]
    
    # Slopes
    t_vals = np.arange(len(cos_to_final))
    
    if len(t_vals) < 2:
        return {"cos_slope": 0.0, "dist_slope": 0.0, "early_late_ratio": 1.0}
    
    cos_slope, _, _, _, _ = stats.linregress(t_vals, cos_to_final)
    dist_slope, _, _, _, _ = stats.linregress(t_vals, dist_to_final)
    
    # Early/Late ratio
    # NOTE: late_mean can be extremely close to 0 if the trajectory rapidly converges,
    # which can create numerically extreme ratios. Use a scale-aware epsilon and clip.
    mid = len(dist_to_final) // 2
    early_mean = np.mean(dist_to_final[:mid]) if mid > 0 else 0.0
    late_mean = np.mean(dist_to_final[mid:]) if mid < len(dist_to_final) else 0.0
    eps = 1e-6 * (early_mean + late_mean + 1.0)
    early_late_ratio = (early_mean + eps) / (late_mean + eps)
    early_late_ratio = float(np.clip(early_late_ratio, 1e-3, 1e3))

    return {
        "cos_slope": float(cos_slope),
        "dist_slope": float(dist_slope),
        "early_late_ratio": early_late_ratio
    }

# --- P1.4: Recurrence/Loopiness ---

def compute_recurrence_metrics(hidden_states, epsilon_percentile=10):
    """Compute recurrence rate and determinism."""
    h = hidden_states
    if len(h) < 5:
        return {"recurrence_rate": 0.0, "determinism": 0.0}
    
    # Compute pairwise distance matrix
    D = squareform(pdist(h))
    
    # IMPORTANT: choosing epsilon as a per-example percentile makes recurrence_rate
    # nearly constant by construction (e.g., ~10%). Instead, use a scale-based epsilon.
    nonzero_dists = D[D > 0]
    if len(nonzero_dists) < 10:
        return {"recurrence_rate": 0.0, "determinism": 0.0}

    scale = np.median(nonzero_dists)
    epsilon = max(1e-12, 0.1 * scale)
    
    # Recurrence matrix
    R = (D < epsilon).astype(int)
    np.fill_diagonal(R, 0)  # Exclude self-recurrence
    
    # Recurrence Rate
    n = len(h)
    RR = np.sum(R) / (n * (n - 1))
    
    # Determinism: proportion of recurrent points in diagonal lines (length >= 2)
    # Count diagonal lines
    diagonal_count = 0
    total_recurrent = np.sum(R)
    
    if total_recurrent < 2:
        return {"recurrence_rate": float(RR), "determinism": 0.0}
    
    # Check diagonals (simplified: count consecutive 1s along diagonals)
    for offset in range(-n + 2, n - 1):
        diag = np.diag(R, k=offset)
        if len(diag) < 2:
            continue
        
        # Count points in runs of length >= 2
        in_run = False
        run_len = 0
        for val in diag:
            if val == 1:
                run_len += 1
            else:
                if run_len >= 2:
                    diagonal_count += run_len
                run_len = 0
        if run_len >= 2:
            diagonal_count += run_len
    
    DET = diagonal_count / (total_recurrent + 1e-10)
    
    return {
        "recurrence_rate": float(RR),
        "determinism": float(np.clip(DET, 0, 1))
    }

# --- Old Metrics for Correlation ---

def compute_speed(hidden_states):
    h = hidden_states
    if len(h) < 2:
        return 0.0
    delta = h[1:] - h[:-1]
    return float(np.mean(np.linalg.norm(delta, axis=1)))

def compute_dir_consistency(hidden_states):
    h = hidden_states
    if len(h) < 3:
        return 0.0
    delta = h[1:] - h[:-1]
    norms = np.linalg.norm(delta, axis=1, keepdims=True) + 1e-9
    unit_delta = delta / norms
    mean_dir = np.mean(unit_delta, axis=0)
    return float(np.linalg.norm(mean_dir))

def compute_effective_dim(hidden_states):
    h = hidden_states
    if len(h) < 4:
        return 0.0
    delta = h[1:] - h[:-1]
    delta_centered = delta - np.mean(delta, axis=0, keepdims=True)
    try:
        _, s, _ = np.linalg.svd(delta_centered, full_matrices=False)
        eigenvalues = (s ** 2) / len(delta)
        sum_sq = np.sum(eigenvalues) ** 2
        sq_sum = np.sum(eigenvalues ** 2)
        if sq_sum < 1e-12:
            return 0.0
        return float(sum_sq / sq_sum)
    except:
        return 0.0

def compute_all_metrics(hidden_states):
    """Compute all new metrics for hidden states."""
    conv = compute_convergence_metrics(hidden_states)
    rec = compute_recurrence_metrics(hidden_states)
    
    return {
        # New P1.1-P1.4 metrics
        "fractal_dim": compute_fractal_dim(hidden_states),
        "intrinsic_dim": compute_intrinsic_dim(hidden_states),
        "cos_slope": conv["cos_slope"],
        "dist_slope": conv["dist_slope"],
        "early_late_ratio": conv["early_late_ratio"],
        "recurrence_rate": rec["recurrence_rate"],
        "determinism": rec["determinism"],
        # Old metrics for correlation
        "speed": compute_speed(hidden_states),
        "dir_consistency": compute_dir_consistency(hidden_states),
        "effective_dim": compute_effective_dim(hidden_states),
    }

def permutation_test(a, b, k=1000):
    observed_diff = np.mean(a) - np.mean(b)
    combined = np.array(list(a) + list(b))
    n_a = len(a)
    diffs = []
    for _ in range(k):
        np.random.shuffle(combined)
        diffs.append(np.mean(combined[:n_a]) - np.mean(combined[n_a:]))
    p_val = np.mean(np.abs(diffs) >= np.abs(observed_diff))
    return p_val, observed_diff

def cohens_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return 0.0
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    s_pooled = np.sqrt(((len(a)-1)*var_a + (len(b)-1)*var_b) / (len(a)+len(b)-2))
    return (np.mean(a) - np.mean(b)) / s_pooled if s_pooled > 0 else 0

def run_comparison(g1_vals, g2_vals):
    if len(g1_vals) < 5 or len(g2_vals) < 5:
        return None
    p_val, diff = permutation_test(g1_vals, g2_vals)
    d = cohens_d(g1_vals, g2_vals)
    return {
        "n1": len(g1_vals), "n2": len(g2_vals),
        "mean1": np.mean(g1_vals), "mean2": np.mean(g2_vals),
        "diff": diff, "p": p_val, "d": d,
        "sig": p_val < 0.05 and abs(d) > 0.5
    }

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    print(f"PID: {os.getpid()}", flush=True)
    print("Loading Model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    
    data = load_data()
    print(f"Loaded {len(data)} samples.")
    
    results = {
        "G1": {l: [] for l in LAYERS},
        "G2": {l: [] for l in LAYERS},
        "G3": {l: [] for l in LAYERS},
        "G4": {l: [] for l in LAYERS}
    }
    
    # For convergence dynamics plot
    dist_dynamics = {"G1": [], "G4": []}
    
    for i, rec in enumerate(data):
        print(f"Analyzing {i+1}/{len(data)}...", end='\r')
        
        # Direct condition
        group_d = "G2" if rec["direct"]["correct"] else "G1"
        prompt_d = rec["direct"]["prompt"]
        resp_d = rec["direct"]["response"]
        full_d = prompt_d + resp_d
        
        inputs = tokenizer(full_d, return_tensors="pt").to(model.device)
        len_prompt = len(tokenizer.encode(prompt_d, add_special_tokens=False))
        len_full = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            out = model(inputs.input_ids, output_hidden_states=True)
        
        start = len_prompt
        end = min(len_full, start + WINDOW_SIZE)
        
        if end > start + 4:
            for l in LAYERS:
                h = out.hidden_states[l][0, start:end, :].float().cpu().numpy()
                metrics = compute_all_metrics(h)
                results[group_d][l].append(metrics)
                
                # Track convergence dynamics for layer 13
                if l == 13 and group_d == "G1":
                    h_final = h[-1]
                    dists = [np.linalg.norm(h[t] - h_final) for t in range(len(h) - 1)]
                    if len(dists) > 0:
                        dist_dynamics["G1"].append(dists)
        
        # CoT condition
        group_c = "G4" if rec["cot"]["correct"] else "G3"
        prompt_c = rec["cot"]["prompt"]
        resp_c = rec["cot"]["response"]
        full_c = prompt_c + resp_c
        
        inputs = tokenizer(full_c, return_tensors="pt").to(model.device)
        len_prompt = len(tokenizer.encode(prompt_c, add_special_tokens=False))
        len_full = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            out = model(inputs.input_ids, output_hidden_states=True)
        
        start = len_prompt
        end = min(len_full, start + WINDOW_SIZE)
        
        if end > start + 4:
            for l in LAYERS:
                h = out.hidden_states[l][0, start:end, :].float().cpu().numpy()
                metrics = compute_all_metrics(h)
                results[group_c][l].append(metrics)
                
                if l == 13 and group_c == "G4":
                    h_final = h[-1]
                    dists = [np.linalg.norm(h[t] - h_final) for t in range(len(h) - 1)]
                    if len(dists) > 0:
                        dist_dynamics["G4"].append(dists)
    
    print("\n\nGenerating Report and Figures...")
    
    # --- Convergence Dynamics Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for group, label, color in [("G4", "CoT Success", "#2ecc71"), ("G1", "Direct Failure", "#e74c3c")]:
        dynamics = dist_dynamics[group]
        if len(dynamics) > 0:
            max_len = max(len(d) for d in dynamics)
            padded = [d + [np.nan]*(max_len - len(d)) for d in dynamics]
            arr = np.array(padded)
            mean_dyn = np.nanmean(arr, axis=0)
            std_dyn = np.nanstd(arr, axis=0)
            x = np.arange(len(mean_dyn))
            ax.plot(x, mean_dyn, label=label, color=color, linewidth=2)
            ax.fill_between(x, mean_dyn - std_dyn, mean_dyn + std_dyn, alpha=0.2, color=color)
    
    ax.set_xlabel("Token Position", fontsize=12)
    ax.set_ylabel("Distance to Final State", fontsize=12)
    ax.set_title("Convergence to Final State (Layer 13)", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(OUTPUT_DIR, "exp12_convergence_dynamics.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # --- Layer Profiles Plot ---
    new_metrics = ["fractal_dim", "intrinsic_dim", "cos_slope", "recurrence_rate"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for idx, metric in enumerate(new_metrics):
        ax = axes[idx // 2, idx % 2]
        for group, label, color in [("G4", "CoT Success", "#2ecc71"), ("G1", "Direct Failure", "#e74c3c")]:
            means = []
            stds = []
            for l in LAYERS:
                vals = [x[metric] for x in results[group][l]]
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            ax.errorbar(LAYERS, means, yerr=stds, label=label, color=color, marker='o', capsize=3)
        ax.set_xlabel("Layer")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs Layer")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "exp12_layer_profiles.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # --- Correlation Matrix ---
    all_metrics = ["speed", "dir_consistency", "effective_dim", "fractal_dim", "intrinsic_dim", 
                   "cos_slope", "dist_slope", "recurrence_rate", "determinism"]
    g4_data = results["G4"][13]
    
    if len(g4_data) > 10:
        corr_matrix = np.zeros((len(all_metrics), len(all_metrics)))
        for i, m1 in enumerate(all_metrics):
            for j, m2 in enumerate(all_metrics):
                vals1 = np.array([x[m1] for x in g4_data], dtype=float)
                vals2 = np.array([x[m2] for x in g4_data], dtype=float)

                # pearsonr returns nan if either vector is constant; guard explicitly.
                if np.nanstd(vals1) < 1e-12 or np.nanstd(vals2) < 1e-12:
                    r = np.nan
                else:
                    r, _ = stats.pearsonr(vals1, vals2)
                corr_matrix[i, j] = r
        
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(np.arange(len(all_metrics)))
        ax.set_yticks(np.arange(len(all_metrics)))
        ax.set_xticklabels(all_metrics, rotation=45, ha='right')
        ax.set_yticklabels(all_metrics)
        for i in range(len(all_metrics)):
            for j in range(len(all_metrics)):
                ax.text(j, i, f"{corr_matrix[i, j]:.2f}", ha="center", va="center",
                       color="white" if abs(corr_matrix[i, j]) > 0.5 else "black", fontsize=8)
        plt.colorbar(im)
        ax.set_title("Full Metric Correlation Matrix (G4, Layer 13)")
        fig.savefig(os.path.join(OUTPUT_DIR, "exp12_correlation_extended.png"), dpi=150, bbox_inches='tight')
        plt.close()
    
    # --- Report ---
    report = [
        "# Experiment 12: Advanced Trajectory Diagnostics Report",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Model**: {MODEL_NAME}",
        f"**N Problems**: {len(data)}",
        "",
        "---",
        "",
        "## 1. Group Sizes",
        "",
        "| Group | Description | N |",
        "|---|---|---|",
        f"| G1 | Direct Failure | {len(results['G1'][LAYERS[0]])} |",
        f"| G2 | Direct Success | {len(results['G2'][LAYERS[0]])} |",
        f"| G3 | CoT Failure | {len(results['G3'][LAYERS[0]])} |",
        f"| G4 | CoT Success | {len(results['G4'][LAYERS[0]])} |",
        "",
        "---",
        "",
        "## 2. New Metrics: Primary Comparison (G4 vs G1)",
        "",
        "| Layer | Metric | G4 Mean | G1 Mean | Cohen's d | p-value | Sig? |",
        "|---|---|---|---|---|---|---|",
    ]
    
    new_metric_names = ["fractal_dim", "intrinsic_dim", "cos_slope", "dist_slope", 
                        "early_late_ratio", "recurrence_rate", "determinism"]
    
    for l in LAYERS:
        for m in new_metric_names:
            g4_vals = [x[m] for x in results["G4"][l]]
            g1_vals = [x[m] for x in results["G1"][l]]
            result = run_comparison(g4_vals, g1_vals)
            if result:
                sig = "✓" if result["sig"] else ""
                report.append(f"| {l} | {m} | {result['mean1']:.4f} | {result['mean2']:.4f} | {result['d']:.2f} | {result['p']:.4f} | {sig} |")
    
    # Add factorial sections
    comparisons = [
        ("G4", "G3", "3.1 Success Effect Within CoT (G4 vs G3)"),
        ("G2", "G1", "3.2 Success Effect Within Direct (G2 vs G1)"),
        ("G3", "G1", "3.3 Prompting Effect - Failures (G3 vs G1)"),
        ("G4", "G2", "3.4 Prompting Effect - Successes (G4 vs G2)"),
    ]
    
    report.extend(["", "---", "", "## 3. Factorial Decomposition (New Metrics)", ""])
    
    for ga, gb, title in comparisons:
        report.extend([f"### {title}", "", f"| Layer | Metric | {ga} Mean | {gb} Mean | Cohen's d | p | Sig? |", "|---|---|---|---|---|---|---|"])
        for l in LAYERS:
            for m in new_metric_names:
                ga_vals = [x[m] for x in results[ga][l]]
                gb_vals = [x[m] for x in results[gb][l]]
                result = run_comparison(ga_vals, gb_vals)
                if result:
                    sig = "✓" if result["sig"] else ""
                    report.append(f"| {l} | {m} | {result['mean1']:.4f} | {result['mean2']:.4f} | {result['d']:.2f} | {result['p']:.4f} | {sig} |")
        report.append("")
    
    # Correlation summary
    report.extend([
        "---",
        "",
        "## 4. Metric Independence",
        "",
        "Correlation of new metrics with existing ones (Layer 13, G4):",
        "",
        "| New Metric | r(speed) | r(DC) | r(eff_dim) |",
        "|---|---|---|---|",
    ])
    
    if len(g4_data) > 10:
        old_metrics = ["speed", "dir_consistency", "effective_dim"]
        for m in new_metric_names:
            m_vals = [x[m] for x in g4_data]
            correlations = []
            for old_m in old_metrics:
                old_vals = [x[old_m] for x in g4_data]
                r, _ = stats.pearsonr(m_vals, old_vals)
                correlations.append(f"{r:.2f}")
            report.append(f"| {m} | {' | '.join(correlations)} |")
    
    report.extend([
        "",
        "![Correlation Matrix](exp12_correlation_extended.png)",
        "",
        "---",
        "",
        "## 5. Figures",
        "",
        "### Convergence Dynamics",
        "![Convergence](exp12_convergence_dynamics.png)",
        "",
        "### Layer Profiles",
        "![Layer Profiles](exp12_layer_profiles.png)",
        "",
        "---",
        "",
        "## 6. Summary",
        "",
    ])
    
    # Count significant effects
    sig_counts = {comp[0] + "_" + comp[1]: 0 for comp in comparisons}
    sig_counts["g4_g1"] = 0
    
    for l in LAYERS:
        for m in new_metric_names:
            g4 = [x[m] for x in results["G4"][l]]
            g1 = [x[m] for x in results["G1"][l]]
            r = run_comparison(g4, g1)
            if r and r["sig"]: sig_counts["g4_g1"] += 1
            
            for ga, gb, _ in comparisons:
                ga_v = [x[m] for x in results[ga][l]]
                gb_v = [x[m] for x in results[gb][l]]
                r = run_comparison(ga_v, gb_v)
                if r and r["sig"]: sig_counts[ga + "_" + gb] += 1
    
    total = len(LAYERS) * len(new_metric_names)
    
    report.extend([
        f"**New metrics with significant effects (p<0.05, |d|>0.5)**:",
        "",
        f"- G4 vs G1 (primary): {sig_counts['g4_g1']}/{total}",
        f"- G4 vs G3 (success within CoT): {sig_counts['G4_G3']}/{total}",
        f"- G2 vs G1 (success within Direct): {sig_counts['G2_G1']}/{total}",
        f"- G3 vs G1 (prompting effect): {sig_counts['G3_G1']}/{total}",
        f"- G4 vs G2 (prompting, successes): {sig_counts['G4_G2']}/{total}",
        "",
        "---",
        "",
        "*Report generated by run_exp12_analysis.py*",
    ])
    
    report_path = os.path.join(OUTPUT_DIR, REPORT_FILENAME)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"\nAnalysis Complete!")
    print(f"Report: {report_path}")

if __name__ == "__main__":
    main()
