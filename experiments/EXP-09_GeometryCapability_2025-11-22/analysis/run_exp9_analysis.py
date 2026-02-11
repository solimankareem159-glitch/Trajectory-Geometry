import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
import numpy as np
from scipy.spatial.distance import cosine
from scipy import stats

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATA_DIR = r"experiments/Experiment 9/data"
DATA_FILENAME = "exp9_dataset.jsonl"
OUTPUT_DIR = r"experiments/Experiment 9/results"
REPORT_FILENAME = "exp9_geometry_capability_report.md"

LAYERS = [0, 10, 11, 12, 13, 14, 15, 16, 24]
WINDOW_SIZE = 32

def load_data():
    path = os.path.join(DATA_DIR, DATA_FILENAME)
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

def compute_metrics(hidden_states):
    # hidden_states: (Seq, D)
    # 1. Speed (Mean Norm of Delta)
    h = hidden_states
    delta = h[1:] - h[:-1]
    norms = np.linalg.norm(delta, axis=1)
    speed = np.mean(norms)
    
    # 2. Curvature (Mean Cosine Similarity of consecutive deltas)
    # Cosine sim = 1 - cosine_dist. 
    # High curvature = Low cosine sim (abrupt turns). 
    # Actually "Curvature" usually means 1 - cos. 
    # Let's use MEAN COSINE SIMILARITY as "Straightness". 
    # If the user asks for "Curvature", I will report 1 - MeanCos.
    # But usually researchers want to know if it's "straight" (high cos) or "jagged" (low cos).
    # Plan: Compute Mean Cosine Similarity.
    
    if len(delta) < 2:
        return {"speed": 0, "curvature": 0, "stabilization": 0, "dir_consistency": 0}
        
    v1 = delta[:-1]
    v2 = delta[1:]
    
    # Normalize for cosine
    v1_norm = v1 / (np.linalg.norm(v1, axis=1, keepdims=True) + 1e-9)
    v2_norm = v2 / (np.linalg.norm(v2, axis=1, keepdims=True) + 1e-9)
    
    cos_sims = np.sum(v1_norm * v2_norm, axis=1)
    curvature = 1 - np.mean(cos_sims) # 0 = straight, 2 = u-turn
    
    # Early vs Late Curvature
    # Split 1-16 vs 17-32 (approx mid point)
    mid = len(cos_sims) // 2
    curv_early = 1 - np.mean(cos_sims[:mid]) if mid > 0 else 0
    curv_late = 1 - np.mean(cos_sims[mid:]) if mid < len(cos_sims) else 0
    
    # 3. Stabilization Rate (Slope of log variance?)
    # "Rate of variance collapse".
    # Compute variance of hidden states in sliding window? Or just norm variance?
    # Let's use Variance of the tokens relative to the mean of the window.
    # Actually, simpler: Ratio of Late Variance / Early Variance of Delta Norms?
    # Or just slope of Delta Norms?
    # User def: "rate of variance collapse over tokens".
    # Let's calculate the Variance of hidden states at each step (if we had multiple samples).
    # But we only have one sample.
    # Proxy: "Variance of Delta Norms" over time?
    # Let's use the Slope of the norm of delta h. If it stabilizes, norm should decrease?
    # Or maybe "Variance of direction"?
    # Let's implement: Slope of linear fit to ||h_t - h_{t-1}||. Negative slope = stabilization.
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(len(norms)), norms)
    stabilization = slope
    
    # 4. Directional Consistency (Variance of Delta Directions)
    # Var(v / |v|)
    # Compute variance of normalized delta vectors per dimension, then sum?
    # "Directional Variance" usually means 1 - ||Mean Vector||.
    # if all vectors are same direction, Mean Vector has norm 1. 1-1=0 variance.
    # if uniform random, Mean Vector has norm 0. 1-0=1 variance.
    mean_dir = np.mean(v1_norm, axis=0)
    dir_consistency = np.linalg.norm(mean_dir) # 1 = perfect line, 0 = random walk
    
    return {
        "speed": speed,
        "curvature_early": curv_early,
        "curvature_late": curv_late,
        "stabilization": stabilization,
        "dir_consistency": dir_consistency
    }

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print("Loading Model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    
    data = load_data()
    print(f"Loaded {len(data)} samples.")
    
    # Store results: Groups G1, G2, G3, G4
    # G1: Direct Fail, G2: Direct Success, G3: CoT Fail, G4: CoT Success
    results = {
        "G1": {l: [] for l in LAYERS},
        "G2": {l: [] for l in LAYERS},
        "G3": {l: [] for l in LAYERS},
        "G4": {l: [] for l in LAYERS}
    }
    
    for i, rec in enumerate(data):
        print(f"Analyzing {i+1}/{len(data)}...", end='\r')
        
        # Condition A: Direct
        group_d = "G2" if rec["direct"]["correct"] else "G1"
        prompt_d = rec["direct"]["prompt"]
        resp_d = rec["direct"]["response"]
        full_d = prompt_d + resp_d
        
        # Forward pass Direct
        inputs = tokenizer(full_d, return_tensors="pt").to(model.device)
        # Find start of response
        len_prompt = len(tokenizer.encode(prompt_d, add_special_tokens=False))
        len_full = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            out = model(inputs.input_ids, output_hidden_states=True)
            
        # Extract Window (First 32 generated tokens)
        # Handle cases where response is short
        start = len_prompt
        end = min(len_full, start + WINDOW_SIZE)
        
        if end > start:
            for l in LAYERS:
                h = out.hidden_states[l][0, start:end, :].float().cpu().numpy()
                metrics = compute_metrics(h)
                results[group_d][l].append(metrics)
                
        # Condition B: CoT
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
        
        if end > start:
            for l in LAYERS:
                h = out.hidden_states[l][0, start:end, :].float().cpu().numpy()
                metrics = compute_metrics(h)
                results[group_c][l].append(metrics)

    # Statistical Analysis
    print("\nRunning Statistical Tests...")
    report_lines = ["# Experiment 9: Geometry-Capability Correlation Report\n"]
    
    # Table Header
    report_lines.append("| Layer | Metric | Comparison | CoT Success (G4) | Direct Fail (G1) | p-value | Cohen's d | Result |")
    report_lines.append("|---|---|---|---|---|---|---|---|")
    
    # Comparisons: G4 vs G1 (Primary)
    for l in LAYERS:
        for m in ["speed", "curvature_early", "stabilization", "dir_consistency"]:
            g4_vals = [x[m] for x in results["G4"][l]]
            g1_vals = [x[m] for x in results["G1"][l]]
            
            # Pilot check: allow small N
            if len(g4_vals) < 1 or len(g1_vals) < 1:
                continue
                
            # Permutation Test (approx via T-test for speed, or mannwhitney)
            # User asked for Permutation test.
            # Implementing simple permutation test for mean difference.
            observed_diff = np.mean(g4_vals) - np.mean(g1_vals)
            combined = np.array(g4_vals + g1_vals)
            n_a = len(g4_vals)
            
            # Fast permutation (1000 shuffles for speed)
            k = 1000
            diffs = []
            for _ in range(k):
                np.random.shuffle(combined)
                diffs.append(np.mean(combined[:n_a]) - np.mean(combined[n_a:]))
            
            p_val = np.mean(np.abs(diffs) >= np.abs(observed_diff))
            
            # Cohen's d
            s_pooled = np.sqrt(((len(g4_vals)-1)*np.var(g4_vals) + (len(g1_vals)-1)*np.var(g1_vals)) / (len(g4_vals)+len(g1_vals)-2))
            d = observed_diff / s_pooled if s_pooled > 0 else 0
            
            signif = "**SIGNIFICANT**" if p_val < 0.05 else ""
            
            report_lines.append(f"| {l} | {m} | G4 vs G1 | {np.mean(g4_vals):.4f} | {np.mean(g1_vals):.4f} | {p_val:.4f} | {d:.2f} | {signif} |")

    # Save Report
    with open(os.path.join(OUTPUT_DIR, REPORT_FILENAME), 'w') as f:
        f.write("\n".join(report_lines))
        
    print(f"Analysis Complete. Report saved to {REPORT_FILENAME}")

if __name__ == "__main__":
    import os
    print(f"PID: {os.getpid()}", flush=True)
    main()
