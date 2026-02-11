import numpy as np
import json
import os
import time
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.spatial.distance import cdist

# --- Configuration ---
UNMASKED_DIR = r"experiments/Experiment 6/data/full_2k"
MASKED_DIR = r"experiments/Experiment 7/data/exp7C_masked"
OUTPUT_DIR = r"experiments/Experiment 7/results/analysis_7c"
LAYERS = [0, 10, 11, 12, 13, 14, 15, 16, 24]
N_SPLITS = 5
N_PERMUTATIONS = 5

def ensure_dir(d):
    try:
        if not os.path.exists(d):
            os.makedirs(d)
    except OSError as e:
        if e.errno != 17:
            raise

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def load_data(data_dir, suffix=""):
    print(f"Loading data from {data_dir}...")
    try:
        # Check for memmap or npy
        f_path_npy = os.path.join(data_dir, f"features{suffix}.npy")
        f_path_mmap = os.path.join(data_dir, f"features{suffix}.memmap")
        
        if os.path.exists(f_path_mmap):
            features = np.memmap(f_path_mmap, dtype='float16', mode='r')
            features = features.reshape((2000, 25, 1792))
        elif os.path.exists(f_path_npy):
            try:
                features = np.load(f_path_npy, mmap_mode='r')
            except:
                # Fallback for raw memmap named as .npy
                features = np.memmap(f_path_npy, dtype='float16', mode='r')
                features = features.reshape((2000, 25, 1792))
        else:
            raise FileNotFoundError(f"No features file found in {data_dir}")

        labels = np.load(os.path.join(data_dir, f"labels{suffix}.npy"), allow_pickle=True)
        ids = np.load(os.path.join(data_dir, f"ids{suffix}.npy"), allow_pickle=True)
        
        return features, labels, ids
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

def compute_stability(X, labels, content_ids):
    unique_ops = np.unique(labels)
    unique_contents = np.unique(content_ids)
    op_stabilities = []
    
    for op in unique_ops:
        op_mask = (labels == op)
        if np.sum(op_mask) == 0: continue
        X_op = X[op_mask]
        c_op = content_ids[op_mask]
        
        c_centroids = []
        for c in unique_contents:
            c_mask = (c_op == c)
            if np.sum(c_mask) > 0:
                c_centroids.append(np.mean(X_op[c_mask], axis=0))
        
        if len(c_centroids) < 2: continue
        c_centroids = np.stack(c_centroids)
        sims = 1 - cdist(c_centroids, c_centroids, metric='cosine')
        n = len(c_centroids)
        off_diag_sum = np.sum(sims) - n
        mean_sim = off_diag_sum / (n * (n - 1))
        op_stabilities.append(mean_sim)
        
    return np.mean(op_stabilities) if op_stabilities else 0.0

def run_ncc(X_train, y_train, X_test, y_test):
    clf = NearestCentroid()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return f1_score(y_test, y_pred, average='macro')

def analyze_dataset(features, labels, ids, desc):
    print(f"\nAnalyzing {desc}...")
    results = {}
    content_ids = ids[:, 0]
    paraphrase_ids = ids[:, 1]
    
    for l in LAYERS:
        print(f"  Layer {l}...", end="\r")
        X = np.array(features[:, l, :], dtype=np.float32)
        
        # 1. Baseline NCC
        f1s = []
        sss = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=0.2, random_state=42)
        for tr, te in sss.split(X, labels):
            f1s.append(run_ncc(X[tr], labels[tr], X[te], labels[te]))
        mean_f1 = np.mean(f1s)
        
        # 2. Stability
        stab = compute_stability(X, labels, content_ids)
        
        # 3. LOPO (Masked only per spec? But good to have for comparo)
        # Spec says "LOPO Under Masking", but implies comparison "LOPO_unmasked vs LOPO_masked"
        unique_paras = np.unique(paraphrase_ids)
        lopo_f1s = []
        for p in unique_paras:
            test_mask = (paraphrase_ids == p)
            train_mask = ~test_mask
            if np.sum(test_mask) > 0:
                lopo_f1s.append(run_ncc(X[train_mask], labels[train_mask], X[test_mask], labels[test_mask]))
        mean_lopo = np.mean(lopo_f1s)
        
        # 4. Joint Holdout
        group_ids = content_ids * 1000 + paraphrase_ids
        unique_groups = np.unique(group_ids)
        joint_f1s = []
        for g in unique_groups:
             test_mask = (group_ids == g)
             train_mask = ~test_mask
             if np.sum(test_mask) > 0:
                 joint_f1s.append(run_ncc(X[train_mask], labels[train_mask], X[test_mask], labels[test_mask]))
        mean_joint = np.mean(joint_f1s)
        
        # 5. Permutation Control (Masked only per spec? We'll do simple check for all)
        perm_f1 = 0
        if desc == "MASKED": # Optimization: only run permutation on masked as requested
             perm_scores = []
             for _ in range(N_PERMUTATIONS):
                 y_perm = np.random.permutation(labels)
                 sss_p = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
                 for tr, te in sss_p.split(X, y_perm):
                     perm_scores.append(run_ncc(X[tr], y_perm[tr], X[te], y_perm[te]))
             perm_f1 = np.mean(perm_scores)

        results[l] = {
            "f1": mean_f1,
            "stability": stab,
            "lopo_f1": mean_lopo,
            "joint_f1": mean_joint,
            "perm_f1": perm_f1
        }
    print(f"  Layer {l} done.")
    return results

def main():
    start_time = time.time()
    ensure_dir(OUTPUT_DIR)
    
    # Load Data
    feat_un, lbl_un, ids_un = load_data(UNMASKED_DIR, "")
    feat_ma, lbl_ma, ids_ma = load_data(MASKED_DIR, "_masked")
    
    if feat_un is None or feat_ma is None:
        print("Data load failure.")
        return

    # Check Alignment
    print("Verifying alignment...")
    assert np.array_equal(lbl_un, lbl_ma), "Labels mismatch!"
    assert np.array_equal(ids_un, ids_ma), "IDs mismatch!"
    print("Alignment verified.")
    
    # Run Analysis
    res_un = analyze_dataset(feat_un, lbl_un, ids_un, "UNMASKED")
    res_ma = analyze_dataset(feat_ma, lbl_ma, ids_ma, "MASKED")
    
    # Synthesis & Reporting
    report_data = {
        "unmasked": res_un,
        "masked": res_ma,
        "comparison": {}
    }
    
    # Save raw results
    with open(os.path.join(OUTPUT_DIR, "exp7C_analysis_results.json"), 'w') as f:
        json.dump(report_data, f, indent=2, cls=NpEncoder)
        
    # Generate Report MD
    md = "# Experiment 7C: Masked vs Unmasked Geometry Analysis\n\n"
    
    md += "## Summary Table\n\n"
    md += "| Layer | F1 (Un) | F1 (Ma) | Ratio (F1) | Stab (Un) | Stab (Ma) | Ratio (Stab) |\n"
    md += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    
    best_mid_f1 = -1
    best_mid_layer = -1
    
    layer0_ratio = 0
    layer0_f1_ma = 0
    
    for l in LAYERS:
        u = res_un[l]
        m = res_ma[l]
        
        ratio_f1 = m['f1'] / u['f1'] if u['f1'] > 0 else 0
        ratio_stab = m['stability'] / u['stability'] if u['stability'] > 0 else 0
        
        if l == 0:
            layer0_ratio = ratio_f1
            layer0_f1_ma = m['f1']
            
        if l in [10, 11, 12, 13, 14, 15, 16]:
            if m['f1'] > best_mid_f1:
                best_mid_f1 = m['f1']
                best_mid_layer = l
        
        md += f"| {l} | {u['f1']:.3f} | {m['f1']:.3f} | {ratio_f1:.2f} | {u['stability']:.3f} | {m['stability']:.3f} | {ratio_stab:.2f} |\n"
        
    res_ma_0 = res_ma[0]
    res_ma_mid = res_ma[best_mid_layer]
    res_un_mid = res_un[best_mid_layer]
    
    mid_ratio = res_ma_mid['f1'] / res_un_mid['f1']
    
    md += "\n## Key Comparisons\n"
    md += f"- **Layer 0 Retention**: {layer0_ratio:.2f} (F1: {res_ma[0]['f1']:.3f})\n"
    md += f"- **Peak Mid-Layer ({best_mid_layer}) Retention**: {mid_ratio:.2f} (F1: {best_mid_f1:.3f})\n"
    md += f"- **Joint Holdout (Masked)**: {res_ma_mid['joint_f1']:.3f} (Mid) vs {res_ma_0['joint_f1']:.3f} (L0)\n"
    md += f"- **Permutation Baseline**: {res_ma_mid['perm_f1']:.3f}\n" 
    
    md += "\n## Conclusion\n\n"
    
    # Interpretation Logic
    outcome = ""
    # Outcome A: Mid Ratio >> L0 Ratio, Stability ~ 1, Masked F1 > Chance
    # Outcome B: Mid Ratio ~ L0 Ratio, Collapse
    # Outcome C: Partial
    
    if (mid_ratio > layer0_ratio + 0.2) and (res_ma_mid['stability'] > 0.8):
        outcome = "Outcome A — Latent Operator State Confirmed"
        rec = "Proceed to Experiment 8."
    elif (abs(mid_ratio - layer0_ratio) < 0.1) or (best_mid_f1 < 0.15): # Close to chance or similar drop
        outcome = "Outcome B — Instruction-Indexed Operator Geometry"
        rec = "Do NOT proceed to Experiment 8. Refine observable or scale model."
    else:
        outcome = "Outcome C — Partial Abstraction"
        rec = "Proceed with caution, or investigate larger models."
        
    md += f"**{outcome}**\n\n"
    md += f"**Recommendation**: {rec}\n"
    
    with open(os.path.join(OUTPUT_DIR, "exp7C_analysis_report.md"), 'w') as f:
        f.write(md)
        
    print(f"\nAnalysis complete. Report saved to {OUTPUT_DIR}")
    print(f"Total time: {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    main()
