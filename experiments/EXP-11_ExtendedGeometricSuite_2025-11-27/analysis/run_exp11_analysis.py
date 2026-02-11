"""
Experiment 11: Extended Metric Suite Analysis
Adds: turning angle, directional autocorrelation, tortuosity, effective dimension

Addresses Phase 1 priorities P1.1-P1.3 from roadmap.
Reuses Experiment 9 data (300 arithmetic problems).
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
import numpy as np
from scipy import stats
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATA_DIR = r"Experiment 9/data"
DATA_FILENAME = "exp9_dataset.jsonl"
OUTPUT_DIR = r"Experiment 11/results"
REPORT_FILENAME = "exp11_extended_metrics_report.md"

LAYERS = [0, 10, 13, 16, 24]
WINDOW_SIZE = 32

def load_data():
    path = os.path.join(DATA_DIR, DATA_FILENAME)
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

# --- New Metrics ---

def compute_turning_angle(hidden_states):
    """Mean turning angle between consecutive steps (in radians)."""
    h = hidden_states
    if len(h) < 3:
        return 0.0
    
    delta = h[1:] - h[:-1]
    norms = np.linalg.norm(delta, axis=1, keepdims=True) + 1e-9
    unit_delta = delta / norms
    
    # Cosine similarity between consecutive unit deltas
    cos_sims = np.sum(unit_delta[:-1] * unit_delta[1:], axis=1)
    cos_sims = np.clip(cos_sims, -1.0, 1.0)  # Numerical stability
    
    angles = np.arccos(cos_sims)
    return float(np.mean(angles))

def compute_directional_autocorr(hidden_states):
    """Directional autocorrelation (signed persistence)."""
    h = hidden_states
    if len(h) < 3:
        return 0.0
    
    delta = h[1:] - h[:-1]
    norms = np.linalg.norm(delta, axis=1, keepdims=True) + 1e-9
    unit_delta = delta / norms
    
    # Mean dot product of consecutive unit deltas
    cos_sims = np.sum(unit_delta[:-1] * unit_delta[1:], axis=1)
    cos_sims = np.clip(cos_sims, -1.0, 1.0)  # Numerical stability
    return float(np.mean(cos_sims))

def compute_tortuosity(hidden_states):
    """Path efficiency: net displacement / arc length. Range [0, 1]."""
    h = hidden_states
    if len(h) < 2:
        return 0.0
    
    delta = h[1:] - h[:-1]
    
    # Net displacement
    net_displacement = np.linalg.norm(np.sum(delta, axis=0))
    
    # Arc length (sum of step magnitudes)
    arc_length = np.sum(np.linalg.norm(delta, axis=1))
    
    if arc_length < 1e-9:
        return 0.0
    
    return float(net_displacement / arc_length)

def compute_effective_dim(hidden_states):
    """Participation ratio of PCA eigenvalues on step vectors."""
    h = hidden_states
    if len(h) < 4:
        return 0.0
    
    delta = h[1:] - h[:-1]
    
    # Center the deltas
    delta_centered = delta - np.mean(delta, axis=0, keepdims=True)
    
    # SVD to get eigenvalues (squared singular values / n)
    try:
        _, s, _ = np.linalg.svd(delta_centered, full_matrices=False)
        eigenvalues = (s ** 2) / len(delta)
        
        # Participation ratio
        sum_sq = np.sum(eigenvalues) ** 2
        sq_sum = np.sum(eigenvalues ** 2)
        
        if sq_sum < 1e-12:
            return 0.0
        
        return float(sum_sq / sq_sum)
    except:
        return 0.0

def compute_effective_dim_over_time(hidden_states, window=8):
    """Compute effective dimension at each position using a sliding window."""
    h = hidden_states
    if len(h) < window + 2:
        return []
    
    delta = h[1:] - h[:-1]
    dims = []
    
    for i in range(len(delta) - window + 1):
        chunk = delta[i:i+window]
        chunk_centered = chunk - np.mean(chunk, axis=0, keepdims=True)
        
        try:
            _, s, _ = np.linalg.svd(chunk_centered, full_matrices=False)
            eigenvalues = (s ** 2) / len(chunk)
            sum_sq = np.sum(eigenvalues) ** 2
            sq_sum = np.sum(eigenvalues ** 2)
            
            if sq_sum < 1e-12:
                dims.append(0.0)
            else:
                dims.append(sum_sq / sq_sum)
        except:
            dims.append(0.0)
    
    return dims

# --- Original Metrics (for correlation analysis) ---

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

def compute_stabilization(hidden_states):
    h = hidden_states
    if len(h) < 3:
        return 0.0
    delta = h[1:] - h[:-1]
    norms = np.linalg.norm(delta, axis=1)
    slope, _, _, _, _ = stats.linregress(np.arange(len(norms)), norms)
    return float(slope)

def compute_all_metrics(hidden_states):
    """Compute all metrics (original + new) for hidden states."""
    return {
        # Original
        "speed": compute_speed(hidden_states),
        "dir_consistency": compute_dir_consistency(hidden_states),
        "stabilization": compute_stabilization(hidden_states),
        # New (P1.1-P1.3)
        "turning_angle": compute_turning_angle(hidden_states),
        "dir_autocorr": compute_directional_autocorr(hidden_states),
        "tortuosity": compute_tortuosity(hidden_states),
        "effective_dim": compute_effective_dim(hidden_states),
    }

def permutation_test(a, b, k=1000):
    """Two-tailed permutation test for mean difference."""
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
        "n1": len(g1_vals),
        "n2": len(g2_vals),
        "mean1": np.mean(g1_vals),
        "mean2": np.mean(g2_vals),
        "diff": diff,
        "p": p_val,
        "d": d,
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
    
    # Store results
    results = {
        "G1": {l: [] for l in LAYERS},  # Direct Fail
        "G2": {l: [] for l in LAYERS},  # Direct Success
        "G3": {l: [] for l in LAYERS},  # CoT Fail
        "G4": {l: [] for l in LAYERS}   # CoT Success
    }
    
    # For effective dimension dynamics plot
    eff_dim_dynamics = {"G1": [], "G4": []}
    
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
        
        if end > start + 3:
            for l in LAYERS:
                h = out.hidden_states[l][0, start:end, :].float().cpu().numpy()
                metrics = compute_all_metrics(h)
                results[group_d][l].append(metrics)
                
                # Track dynamics for mid-layer
                if l == 13 and group_d in ["G1"]:
                    dims = compute_effective_dim_over_time(h)
                    if len(dims) > 0:
                        eff_dim_dynamics["G1"].append(dims)
        
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
        
        if end > start + 3:
            for l in LAYERS:
                h = out.hidden_states[l][0, start:end, :].float().cpu().numpy()
                metrics = compute_all_metrics(h)
                results[group_c][l].append(metrics)
                
                if l == 13 and group_c in ["G4"]:
                    dims = compute_effective_dim_over_time(h)
                    if len(dims) > 0:
                        eff_dim_dynamics["G4"].append(dims)
    
    print("\n\nGenerating Report and Figures...")
    
    # --- Generate Effective Dimension Dynamics Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Average dynamics per group
    for group, label, color in [("G4", "CoT Success", "#2ecc71"), ("G1", "Direct Failure", "#e74c3c")]:
        dynamics = eff_dim_dynamics[group]
        if len(dynamics) > 0:
            # Pad to same length
            max_len = max(len(d) for d in dynamics)
            padded = [d + [np.nan]*(max_len - len(d)) for d in dynamics]
            arr = np.array(padded)
            mean_dyn = np.nanmean(arr, axis=0)
            std_dyn = np.nanstd(arr, axis=0)
            
            x = np.arange(len(mean_dyn))
            ax.plot(x, mean_dyn, label=label, color=color, linewidth=2)
            ax.fill_between(x, mean_dyn - std_dyn, mean_dyn + std_dyn, alpha=0.2, color=color)
    
    ax.set_xlabel("Token Position (in analysis window)", fontsize=12)
    ax.set_ylabel("Effective Dimension", fontsize=12)
    ax.set_title("Effective Dimension Dynamics (Layer 13)", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.savefig(os.path.join(OUTPUT_DIR, "exp11_effective_dim_dynamics.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # --- Generate Correlation Matrix ---
    # Collect all metrics from G4 at layer 13 for correlation
    metric_names = ["speed", "dir_consistency", "stabilization", "turning_angle", "dir_autocorr", "tortuosity", "effective_dim"]
    g4_data = results["G4"][13]
    
    if len(g4_data) > 10:
        corr_matrix = np.zeros((len(metric_names), len(metric_names)))
        for i, m1 in enumerate(metric_names):
            for j, m2 in enumerate(metric_names):
                vals1 = np.array([x[m1] for x in g4_data], dtype=float)
                vals2 = np.array([x[m2] for x in g4_data], dtype=float)

                # pearsonr returns nan if either vector is constant; guard explicitly.
                if np.nanstd(vals1) < 1e-12 or np.nanstd(vals2) < 1e-12:
                    r = np.nan
                else:
                    r, _ = stats.pearsonr(vals1, vals2)
                corr_matrix[i, j] = r
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        
        ax.set_xticks(np.arange(len(metric_names)))
        ax.set_yticks(np.arange(len(metric_names)))
        ax.set_xticklabels(metric_names, rotation=45, ha='right')
        ax.set_yticklabels(metric_names)
        
        # Add correlation values
        for i in range(len(metric_names)):
            for j in range(len(metric_names)):
                ax.text(j, i, f"{corr_matrix[i, j]:.2f}", ha="center", va="center", 
                       color="white" if abs(corr_matrix[i, j]) > 0.5 else "black", fontsize=9)
        
        plt.colorbar(im)
        ax.set_title("Metric Correlation Matrix (G4, Layer 13)", fontsize=14)
        
        fig.savefig(os.path.join(OUTPUT_DIR, "exp11_correlation_matrix.png"), dpi=150, bbox_inches='tight')
        plt.close()
    
    # --- Generate Report ---
    report = [
        "# Experiment 11: Extended Metric Suite Report",
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
    
    new_metrics = ["turning_angle", "dir_autocorr", "tortuosity", "effective_dim"]
    
    for l in LAYERS:
        for m in new_metrics:
            g4_vals = [x[m] for x in results["G4"][l]]
            g1_vals = [x[m] for x in results["G1"][l]]
            result = run_comparison(g4_vals, g1_vals)
            if result:
                sig = "✓" if result["sig"] else ""
                report.append(f"| {l} | {m} | {result['mean1']:.4f} | {result['mean2']:.4f} | {result['d']:.2f} | {result['p']:.4f} | {sig} |")
    
    report.extend([
        "",
        "---",
        "",
        "## 3. Factorial Decomposition (New Metrics)",
        "",
        "### 3.1 Success Effect Within CoT (G4 vs G3)",
        "",
        "| Layer | Metric | G4 Mean | G3 Mean | Cohen's d | p | Sig? |",
        "|---|---|---|---|---|---|---|",
    ])
    
    for l in LAYERS:
        for m in new_metrics:
            g4_vals = [x[m] for x in results["G4"][l]]
            g3_vals = [x[m] for x in results["G3"][l]]
            result = run_comparison(g4_vals, g3_vals)
            if result:
                sig = "✓" if result["sig"] else ""
                report.append(f"| {l} | {m} | {result['mean1']:.4f} | {result['mean2']:.4f} | {result['d']:.2f} | {result['p']:.4f} | {sig} |")
    
    report.extend([
        "",
        "### 3.2 Success Effect Within Direct (G2 vs G1)",
        "",
        "| Layer | Metric | G2 Mean | G1 Mean | Cohen's d | p | Sig? |",
        "|---|---|---|---|---|---|---|",
    ])
    
    for l in LAYERS:
        for m in new_metrics:
            g2_vals = [x[m] for x in results["G2"][l]]
            g1_vals = [x[m] for x in results["G1"][l]]
            result = run_comparison(g2_vals, g1_vals)
            if result:
                sig = "✓" if result["sig"] else ""
                report.append(f"| {l} | {m} | {result['mean1']:.4f} | {result['mean2']:.4f} | {result['d']:.2f} | {result['p']:.4f} | {sig} |")
    
    report.extend([
        "",
        "### 3.3 Prompting Effect (G3 vs G1 - both failed)",
        "",
        "| Layer | Metric | G3 Mean | G1 Mean | Cohen's d | p | Sig? |",
        "|---|---|---|---|---|---|---|",
    ])
    
    for l in LAYERS:
        for m in new_metrics:
            g3_vals = [x[m] for x in results["G3"][l]]
            g1_vals = [x[m] for x in results["G1"][l]]
            result = run_comparison(g3_vals, g1_vals)
            if result:
                sig = "✓" if result["sig"] else ""
                report.append(f"| {l} | {m} | {result['mean1']:.4f} | {result['mean2']:.4f} | {result['d']:.2f} | {result['p']:.4f} | {sig} |")
    
    report.extend([
        "",
        "### 3.4 Prompting Effect (G4 vs G2 - both succeeded)",
        "",
        "| Layer | Metric | G4 Mean | G2 Mean | Cohen's d | p | Sig? |",
        "|---|---|---|---|---|---|---|",
    ])
    
    for l in LAYERS:
        for m in new_metrics:
            g4_vals = [x[m] for x in results["G4"][l]]
            g2_vals = [x[m] for x in results["G2"][l]]
            result = run_comparison(g4_vals, g2_vals)
            if result:
                sig = "✓" if result["sig"] else ""
                report.append(f"| {l} | {m} | {result['mean1']:.4f} | {result['mean2']:.4f} | {result['d']:.2f} | {result['p']:.4f} | {sig} |")
    
    # --- Correlation with DC ---
    report.extend([
        "",
        "---",
        "",
        "## 4. Metric Independence",
        "",
        "Correlation between new metrics and Directional Consistency (Layer 13, G4):",
        "",
        "| Metric | r(DC) |",
        "|---|---|",
    ])
    
    if len(g4_data) > 10:
        dc_vals = [x["dir_consistency"] for x in g4_data]
        for m in new_metrics:
            m_vals = [x[m] for x in g4_data]
            r, _ = stats.pearsonr(dc_vals, m_vals)
            report.append(f"| {m} | {r:.3f} |")
    
    report.extend([
        "",
        "![Correlation Matrix](exp11_correlation_matrix.png)",
        "",
        "---",
        "",
        "## 5. Effective Dimension Dynamics",
        "",
        "Does effective dimension decrease over generation (explore→commit)?",
        "",
        "![Effective Dimension Dynamics](exp11_effective_dim_dynamics.png)",
        "",
    ])
    
    # Compute slope of d_eff for G4 vs G1
    for group in ["G4", "G1"]:
        dynamics = eff_dim_dynamics[group]
        if len(dynamics) > 5:
            # Average dynamics
            max_len = max(len(d) for d in dynamics)
            padded = [d + [np.nan]*(max_len - len(d)) for d in dynamics]
            arr = np.array(padded)
            mean_dyn = np.nanmean(arr, axis=0)
            
            # Fit slope
            valid = ~np.isnan(mean_dyn)
            if np.sum(valid) > 3:
                slope, _, _, _, _ = stats.linregress(np.arange(np.sum(valid)), mean_dyn[valid])
                report.append(f"- **{group} mean slope**: {slope:.4f} (negative = contraction)")
    
    report.extend([
        "",
        "---",
        "",
        "## 6. Summary",
        "",
    ])
    
    # Count significant new metrics
    sig_counts = {"g4_g1": 0, "g4_g3": 0, "g2_g1": 0, "g3_g1": 0, "g4_g2": 0}
    
    for l in LAYERS:
        for m in new_metrics:
            g4 = [x[m] for x in results["G4"][l]]
            g3 = [x[m] for x in results["G3"][l]]
            g2 = [x[m] for x in results["G2"][l]]
            g1 = [x[m] for x in results["G1"][l]]
            
            r = run_comparison(g4, g1)
            if r and r["sig"]: sig_counts["g4_g1"] += 1
            
            r = run_comparison(g4, g3)
            if r and r["sig"]: sig_counts["g4_g3"] += 1
            
            r = run_comparison(g2, g1)
            if r and r["sig"]: sig_counts["g2_g1"] += 1
            
            r = run_comparison(g3, g1)
            if r and r["sig"]: sig_counts["g3_g1"] += 1
            
            r = run_comparison(g4, g2)
            if r and r["sig"]: sig_counts["g4_g2"] += 1
    
    total = len(LAYERS) * len(new_metrics)
    
    report.extend([
        f"**New metrics with significant effects (p<0.05, |d|>0.5)**:",
        "",
        f"- G4 vs G1 (primary): {sig_counts['g4_g1']}/{total}",
        f"- G4 vs G3 (success within CoT): {sig_counts['g4_g3']}/{total}",
        f"- G2 vs G1 (success within Direct): {sig_counts['g2_g1']}/{total}",
        f"- G3 vs G1 (prompting effect): {sig_counts['g3_g1']}/{total}",
        f"- G4 vs G2 (prompting, successes): {sig_counts['g4_g2']}/{total}",
        "",
        "---",
        "",
        "*Report generated by run_exp11_analysis.py*",
    ])
    
    # Save report
    report_path = os.path.join(OUTPUT_DIR, REPORT_FILENAME)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"\nAnalysis Complete!")
    print(f"Report: {report_path}")
    print(f"Figures: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
