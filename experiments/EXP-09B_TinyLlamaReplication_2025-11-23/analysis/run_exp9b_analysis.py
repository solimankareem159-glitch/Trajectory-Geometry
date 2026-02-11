"""
Experiment 9B: Cross-Model Replication Analysis
Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0

Identical metrics and statistics to Exp 9:
- Speed: Mean L2 norm of h_{t+1} - h_t
- Directional Consistency: Norm of mean normalized direction vector
- Stabilization Rate: Slope of linear fit to displacement norms
- Curvature (Early): 1 - mean cosine similarity (tokens 1-16)

Statistics:
- Permutation test (1000 shuffles)
- Cohen's d (pooled std)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
import numpy as np
from scipy import stats

# --- Configuration ---
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_DIR = r"experiments/Experiment 9B/data"
DATA_FILENAME = "exp9b_dataset.jsonl"
OUTPUT_DIR = r"experiments/Experiment 9B/results"
REPORT_FILENAME = "exp9b_replication_report.md"

# TinyLlama has 22 layers. Select: 0, mid (9-11), final (21)
LAYERS = [0, 9, 10, 11, 21]
WINDOW_SIZE = 32

# Original Qwen2.5-0.5B results for comparison
ORIGINAL_RESULTS = {
    "speed": {"d": 3.0, "direction": "G4 > G1"},  # d ~ 2.66-4.25
    "dir_consistency": {"d": -2.6, "direction": "G4 < G1"},  # d ~ -2.58 to -2.63
    "stabilization": {"d": 1.1, "direction": "G4 > G1"},  # d ~ 0.69-2.03
    "curvature_early": {"d": 0.5, "direction": "G4 > G1"}  # d ~ 0.39-0.82
}

def load_data():
    path = os.path.join(DATA_DIR, DATA_FILENAME)
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

def compute_metrics(hidden_states):
    """Identical to Exp 9."""
    h = hidden_states
    delta = h[1:] - h[:-1]
    norms = np.linalg.norm(delta, axis=1)
    speed = np.mean(norms)
    
    if len(delta) < 2:
        return {"speed": 0, "curvature_early": 0, "stabilization": 0, "dir_consistency": 0}
    
    v1 = delta[:-1]
    v2 = delta[1:]
    
    # Normalize for cosine
    v1_norm = v1 / (np.linalg.norm(v1, axis=1, keepdims=True) + 1e-9)
    v2_norm = v2 / (np.linalg.norm(v2, axis=1, keepdims=True) + 1e-9)
    
    cos_sims = np.sum(v1_norm * v2_norm, axis=1)
    
    # Early curvature: first 16 tokens
    mid = min(len(cos_sims), 16)
    curv_early = 1 - np.mean(cos_sims[:mid]) if mid > 0 else 0
    
    # Stabilization: slope of norms
    slope, _, _, _, _ = stats.linregress(np.arange(len(norms)), norms)
    stabilization = slope
    
    # Directional consistency
    mean_dir = np.mean(v1_norm, axis=0)
    dir_consistency = np.linalg.norm(mean_dir)
    
    return {
        "speed": float(speed),
        "curvature_early": float(curv_early),
        "stabilization": float(stabilization),
        "dir_consistency": float(dir_consistency)
    }

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print("Loading Model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    data = load_data()
    print(f"Loaded {len(data)} samples.")
    
    # Groups: G1=Direct Fail, G2=Direct Success, G3=CoT Fail, G4=CoT Success
    results = {
        "G1": {l: [] for l in LAYERS},
        "G2": {l: [] for l in LAYERS},
        "G3": {l: [] for l in LAYERS},
        "G4": {l: [] for l in LAYERS}
    }
    
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
        
        if end > start + 1:
            for l in LAYERS:
                h = out.hidden_states[l][0, start:end, :].float().cpu().numpy()
                metrics = compute_metrics(h)
                results[group_d][l].append(metrics)
        
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
        
        if end > start + 1:
            for l in LAYERS:
                h = out.hidden_states[l][0, start:end, :].float().cpu().numpy()
                metrics = compute_metrics(h)
                results[group_c][l].append(metrics)
    
    # Statistical Analysis
    print("\nRunning Statistical Tests...")
    
    report_lines = [
        "# Experiment 9B: Cross-Model Replication Report",
        "",
        f"**Replication Model**: TinyLlama-1.1B-Chat",
        f"**Original Model**: Qwen2.5-0.5B",
        f"**N**: {len(data)} samples",
        "",
        "## Group Sizes",
        "",
        f"- G1 (Direct Fail): {len(results['G1'][LAYERS[0]])}",
        f"- G2 (Direct Success): {len(results['G2'][LAYERS[0]])}",
        f"- G3 (CoT Fail): {len(results['G3'][LAYERS[0]])}",
        f"- G4 (CoT Success): {len(results['G4'][LAYERS[0]])}",
        "",
        "## Primary Comparison: G4 vs G1",
        "",
        "| Layer | Metric | G4 Mean | G1 Mean | Cohen's d | p-value | Original d | Sign Match | Replicated |",
        "|---|---|---|---|---|---|---|---|---|"
    ]
    
    replication_summary = {"replicated": 0, "partial": 0, "failed": 0, "sign_reversed": 0}
    
    for l in LAYERS:
        for m in ["speed", "curvature_early", "stabilization", "dir_consistency"]:
            g4_vals = [x[m] for x in results["G4"][l]]
            g1_vals = [x[m] for x in results["G1"][l]]
            
            if len(g4_vals) < 5 or len(g1_vals) < 5:
                continue
            
            # Permutation test
            observed_diff = np.mean(g4_vals) - np.mean(g1_vals)
            combined = np.array(g4_vals + g1_vals)
            n_a = len(g4_vals)
            
            k = 1000
            diffs = []
            for _ in range(k):
                np.random.shuffle(combined)
                diffs.append(np.mean(combined[:n_a]) - np.mean(combined[n_a:]))
            
            p_val = np.mean(np.abs(diffs) >= np.abs(observed_diff))
            
            # Cohen's d
            var_g4 = np.var(g4_vals, ddof=1)
            var_g1 = np.var(g1_vals, ddof=1)
            s_pooled = np.sqrt(((len(g4_vals)-1)*var_g4 + (len(g1_vals)-1)*var_g1) / (len(g4_vals)+len(g1_vals)-2))
            d = observed_diff / s_pooled if s_pooled > 0 else 0
            
            # Compare with original
            orig_d = ORIGINAL_RESULTS[m]["d"]
            orig_direction = ORIGINAL_RESULTS[m]["direction"]
            
            # Check sign match
            new_direction = "G4 > G1" if d > 0 else "G4 < G1"
            sign_match = "✓" if new_direction == orig_direction else "✗"
            
            # Replication status
            if sign_match == "✓" and p_val < 0.05 and abs(d) >= 0.5:
                status = "**REPLICATED**"
                replication_summary["replicated"] += 1
            elif sign_match == "✓" and (p_val < 0.10 or abs(d) >= 0.3):
                status = "Partial"
                replication_summary["partial"] += 1
            elif sign_match == "✗":
                status = "REVERSED"
                replication_summary["sign_reversed"] += 1
            else:
                status = ""
                replication_summary["failed"] += 1
            
            report_lines.append(
                f"| {l} | {m} | {np.mean(g4_vals):.4f} | {np.mean(g1_vals):.4f} | "
                f"{d:.2f} | {p_val:.4f} | {orig_d:.2f} | {sign_match} | {status} |"
            )
    
    # Summary
    report_lines.extend([
        "",
        "## Replication Summary",
        "",
        f"- **Replicated** (same direction, p<0.05, |d|≥0.5): {replication_summary['replicated']}",
        f"- **Partial** (same direction, weaker): {replication_summary['partial']}",
        f"- **Failed** (no effect): {replication_summary['failed']}",
        f"- **Sign Reversed**: {replication_summary['sign_reversed']}",
        "",
        "## Interpretation",
        "",
    ])
    
    total_tests = sum(replication_summary.values())
    if replication_summary["replicated"] >= total_tests * 0.5:
        verdict = "**PASS**: Majority of effects replicated."
    elif replication_summary["replicated"] + replication_summary["partial"] >= total_tests * 0.5:
        verdict = "**PARTIAL**: Some effects replicated, some weakened."
    else:
        verdict = "**FAIL**: Most effects did not replicate."
    
    report_lines.append(verdict)
    
    # Save report
    with open(os.path.join(OUTPUT_DIR, REPORT_FILENAME), 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    
    print(f"Analysis Complete. Report saved to {REPORT_FILENAME}")

if __name__ == "__main__":
    print(f"PID: {os.getpid()}", flush=True)
    main()
