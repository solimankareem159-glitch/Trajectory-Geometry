import numpy as np
import json
import os
import time
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.spatial.distance import cdist

# --- Configuration ---
DATA_DIR = r"experiments/Experiment 6/data/full_2k"
OUTPUT_DIR = r"experiments/Experiment 7/results/audit_7b"
LAYERS = [0, 10, 11, 12, 13, 14, 15, 16, 24]
N_SPLITS_BASELINE = 5
N_PERMUTATIONS = 5

def ensure_dir(d):
    try:
        if not os.path.exists(d):
            os.makedirs(d)
    except OSError as e:
        if e.errno != 17:  # File exists
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

def load_data():
    print("Loading data...")
    try:
        # 2000 * 25 * 1792 * 2 bytes = 179,200,000 bytes
        features = np.memmap(os.path.join(DATA_DIR, "features.memmap"), dtype='float16', mode='r')
        features = features.reshape((2000, 25, 1792))
        
        labels = np.load(os.path.join(DATA_DIR, "labels.npy"))
        ids = np.load(os.path.join(DATA_DIR, "ids.npy"))
        return features, labels, ids
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

def compute_stability(X, labels, content_ids):
    # Mean pairwise cosine similarity of operator centroids across contents
    unique_ops = np.unique(labels)
    unique_contents = np.unique(content_ids)
    
    op_stabilities = []
    
    for op in unique_ops:
        op_mask = (labels == op)
        if np.sum(op_mask) == 0: continue
        
        X_op = X[op_mask]
        c_op = content_ids[op_mask]
        
        # Centroids per content
        c_centroids = []
        for c in unique_contents:
            c_mask = (c_op == c)
            if np.sum(c_mask) > 0:
                c_centroids.append(np.mean(X_op[c_mask], axis=0))
        
        if len(c_centroids) < 2:
            continue
            
        c_centroids = np.stack(c_centroids)
        # 1 - cosine_dist = cosine_sim
        sims = 1 - cdist(c_centroids, c_centroids, metric='cosine')
        
        # Mean off-diagonal
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

def main():
    print("--- Experiment 7B: Full-Scale Invariance & Control Validation ---")
    start_time = time.time()
    ensure_dir(OUTPUT_DIR)
    
    features_mmap, labels, ids = load_data()
    if features_mmap is None:
        return

    # Load entirely into RAM as float32 for speed
    print("Converting to float32...")
    features = np.array(features_mmap, dtype=np.float32)
    
    content_ids = ids[:, 0]
    paraphrase_ids = ids[:, 1]
    
    report = {
        "layers": {},
        "controls": {}
    }
    
    permutation_results = {}

    # Check for masked features
    masked_path = os.path.join(DATA_DIR, "features_masked.memmap")
    masked_available = os.path.exists(masked_path)
    if not masked_available:
        print("Instruction masking at full scale not available — results below conditional.")
        report["controls"]["instruction_masking"] = "Not Available"
        # Save placeholder comparisons
        with open(os.path.join(OUTPUT_DIR, "masked_ncc_comparison.json"), 'w') as f:
            json.dump({"status": "Not Available"}, f, indent=2)
        with open(os.path.join(OUTPUT_DIR, "masked_stability_comparison.json"), 'w') as f:
            json.dump({"status": "Not Available"}, f, indent=2)

    # Main Layer Loop
    print(f"\nAnalyzing Layers: {LAYERS}")
    
    for l in LAYERS:
        print(f"\n--- Layer {l} ---")
        X = features[:, l, :]
        
        # --- Step 1: Baseline Geometry ---
        
        # Centroids
        unique_ops = np.sort(np.unique(labels))
        centroids = []
        for op in unique_ops:
            centroids.append(np.mean(X[labels == op], axis=0))
        centroids = np.stack(centroids)
        np.save(os.path.join(OUTPUT_DIR, f"baseline_centroids_layer_{l}.npy"), centroids)
        
        # NCC F1 (5 splits)
        baseline_f1s = []
        sss = StratifiedShuffleSplit(n_splits=N_SPLITS_BASELINE, test_size=0.2, random_state=42)
        for train_idx, test_idx in sss.split(X, labels):
            f1 = run_ncc(X[train_idx], labels[train_idx], X[test_idx], labels[test_idx])
            baseline_f1s.append(f1)
        mean_baseline_f1 = np.mean(baseline_f1s)
        
        # Stability
        stability = compute_stability(X, labels, content_ids)
        
        print(f"[Baseline] F1: {mean_baseline_f1:.3f} | Stable: {stability:.3f}")
        
        # Save Baseline Artifacts
        with open(os.path.join(OUTPUT_DIR, f"baseline_ncc_layer_{l}.json"), 'w') as f:
            json.dump({"layer": l, "mean_f1": mean_baseline_f1, "all_f1s": baseline_f1s}, f, cls=NpEncoder)
            
        with open(os.path.join(OUTPUT_DIR, f"baseline_stability_layer_{l}.json"), 'w') as f:
            json.dump({"layer": l, "stability": stability}, f, cls=NpEncoder)

        report["layers"][l] = {
            "baseline_f1_mean": mean_baseline_f1,
            "baseline_f1_std": np.std(baseline_f1s),
            "stability": stability
        }

        # --- Step 3: LOPO (Leave-One-Paraphrase-Out) ---
        unique_paras = np.sort(np.unique(paraphrase_ids))
        lopo_f1s = []
        for p in unique_paras:
            test_mask = (paraphrase_ids == p)
            train_mask = ~test_mask
            if np.sum(test_mask) == 0: continue
            
            f1 = run_ncc(X[train_mask], labels[train_mask], X[test_mask], labels[test_mask])
            lopo_f1s.append(f1)
            
        mean_lopo = np.mean(lopo_f1s)
        report["layers"][l]["lopo_f1_mean"] = mean_lopo
        report["layers"][l]["lopo_f1_std"] = np.std(lopo_f1s) if lopo_f1s else 0
        
        # Save LOPO
        with open(os.path.join(OUTPUT_DIR, f"lopo_results_layer_{l}.json"), 'w') as f:
            json.dump({"layer": l, "mean_f1": mean_lopo, "all_f1s": lopo_f1s}, f, cls=NpEncoder)

        # --- Step 4: Joint Holdout ---
        # content x paraphrase combinations
        group_ids = content_ids * 1000 + paraphrase_ids
        unique_groups = np.unique(group_ids)
        
        joint_f1s = []
        joint_stabilities = []
        
        for g in unique_groups:
            test_mask = (group_ids == g)
            train_mask = ~test_mask
            if np.sum(test_mask) == 0: continue
            
            # NCC
            f1 = run_ncc(X[train_mask], labels[train_mask], X[test_mask], labels[test_mask])
            joint_f1s.append(f1)
            
            # Stability on training data
            stab = compute_stability(X[train_mask], labels[train_mask], content_ids[train_mask])
            joint_stabilities.append(stab)
            
        mean_joint = np.mean(joint_f1s)
        mean_joint_stability = np.mean(joint_stabilities)
        
        report["layers"][l]["joint_f1_mean"] = mean_joint
        report["layers"][l]["joint_stability_mean"] = mean_joint_stability
        
        # Save Joint Holdout
        with open(os.path.join(OUTPUT_DIR, f"joint_holdout_results_layer_{l}.json"), 'w') as f:
            json.dump({
                "layer": l, 
                "mean_f1": mean_joint, 
                "all_f1s": joint_f1s,
                "mean_stability": mean_joint_stability,
                "all_stabilities": joint_stabilities
            }, f, cls=NpEncoder)

        # --- Step 5: Permutation (Accumulate) ---
        perm_f1s = []
        for _ in range(N_PERMUTATIONS):
            y_perm = np.random.permutation(labels)
            sss_perm = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
            for tr, te in sss_perm.split(X, y_perm):
                f1 = run_ncc(X[tr], y_perm[tr], X[te], y_perm[te])
                perm_f1s.append(f1)
        mean_perm = np.mean(perm_f1s)
        permutation_results[l] = mean_perm
        report["layers"][l]["permutation_f1"] = mean_perm

    # Save Permutation Control
    with open(os.path.join(OUTPUT_DIR, "permutation_control.json"), 'w') as f:
        json.dump(permutation_results, f, cls=NpEncoder)

    # --- Step 6: Final Report ---
    with open(os.path.join(OUTPUT_DIR, "exp7B_full_report.json"), 'w') as f:
        json.dump(report, f, indent=2, cls=NpEncoder)
        
    summary_md = "# Experiment 7B Summary\n\n"
    summary_md += "## Geometric Invariance Metrics\n\n"
    summary_md += "| Layer | Baseline F1 | LOPO F1 | Joint F1 | Stability | Permutation |\n"
    summary_md += "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
    
    # Identify Peaks
    best_layer = -1
    best_f1 = -1
    
    for l in LAYERS:
        r = report["layers"][l]
        f1 = r['baseline_f1_mean']
        if f1 > best_f1:
            best_f1 = f1
            best_layer = l
        summary_md += f"| {l} | {r['baseline_f1_mean']:.3f} | {r['lopo_f1_mean']:.3f} | {r['joint_f1_mean']:.3f} | {r['stability']:.3f} | {r.get('permutation_f1', 0):.3f} |\n"

    summary_md += f"\n## Conclusions\n\n"
    summary_md += f"- **Peak Geometry**: Layer {best_layer} (F1 = {best_f1:.3f})\n"
    
    # Simple heuristic checks
    mid_layers = [10, 11, 12, 13, 14, 15, 16]
    mid_means = [report["layers"][ml]['baseline_f1_mean'] for ml in mid_layers]
    l0_mean = report["layers"][0]['baseline_f1_mean']
    
    if max(mid_means) > l0_mean + 0.1:
         summary_md += "- **Layer Control**: Mid-layers significantly outperform Layer 0 (Embedding).\n"
    else:
         summary_md += "- **Layer Control**: Mid-layers show little to no advantage over geometric embedding geometry.\n"
         
    summary_md += f"- **Instruction Masking**: {report['controls']['instruction_masking']}\n"
    
    # Recommendation
    if best_f1 > 0.7:  # Arbitrary threshold from experience, but reasonable
        summary_md += "\n> [!TIP]\n> **Recommendation**: Proceed to Experiment 8 (Dynamic Trajectory Analysis). Robust geometric structure detected.\n"
    else:
        summary_md += "\n> [!WARNING]\n> **Recommendation**: Do NOT proceed. Geometric separation is weak.\n"

    with open(os.path.join(OUTPUT_DIR, "exp7B_summary.md"), 'w') as f:
        f.write(summary_md)

    print(f"\nSaved report to {OUTPUT_DIR}")
    print(f"Total time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
