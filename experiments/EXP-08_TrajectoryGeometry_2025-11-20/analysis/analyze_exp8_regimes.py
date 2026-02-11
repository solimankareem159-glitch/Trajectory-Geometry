import numpy as np
import json
import os
import time
print("Imports starting...", flush=True)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
print("Imports done.", flush=True)

# Use threadpool limits to avoid excessive CPU usage
os.environ["OMP_NUM_THREADS"] = "4"
print("Script starting...", flush=True)

# --- Configuration ---
DATA_DIR = r"experiments/Experiment 8/data"
OUTPUT_DIR = r"experiments/Experiment 8/results/analysis"
TRAJ_FILENAME = "exp8_trajectories_v4.npy"
CONTEXT_FILENAME = "exp8_context_v4.npy"
METADATA_FILENAME = "exp8_metadata_v4.jsonl"
LAYERS_OF_INTEREST = [0, 13, 24] # Baseline, Mid, Final
K_RANGE = range(4, 11) # 4 to 10
N_SAMPLES = 2000
L_LAYERS = 25
D_MODEL = 896
D_TOTAL = D_MODEL + 5

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

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
    print("Inside load_data", flush=True)
    print("Loading data...", flush=True)
    try:
        traj_path = os.path.join(DATA_DIR, TRAJ_FILENAME)
        ctx_path = os.path.join(DATA_DIR, CONTEXT_FILENAME)
        meta_path = os.path.join(DATA_DIR, METADATA_FILENAME)
        
        print("Opening memmaps...", flush=True)
        traj = np.memmap(traj_path, dtype='float32', mode='r', shape=(N_SAMPLES, L_LAYERS, D_TOTAL))
        ctx = np.memmap(ctx_path, dtype='float32', mode='r', shape=(N_SAMPLES, D_MODEL))
        print("Memmaps opened. Opening metadata...", flush=True)
        
        # Load Metadata
        metadata = []
        meta_path_temp = meta_path.replace(".jsonl", "_temp.jsonl")
        print(f"Reading metadata from {meta_path_temp}...", flush=True)
        with open(meta_path_temp, 'r') as f:
            lines = f.readlines()
        print(f"Read {len(lines)} lines. Parsing...", flush=True)
        for line in lines:
            if line.strip():
                try:
                    metadata.append(json.loads(line))
                except:
                    pass
        print(f"Loaded {len(metadata)} samples.", flush=True)
        return traj, ctx, metadata
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

def analyze_layer(layer_idx, traj_data, context_data, metadata, desc):
    print(f"\nAnalyzing Layer {layer_idx} ({desc})...")
    
    # 1. Prepare Features
    # Filter only samples with valid reasoning cues? 
    # User said "Anchor windows on: cue phrases OR first reasoning marker". "Identify a language span...".
    # If cue_found is False, the window is just the start.
    # Let's filter for cue_found=True to be strict about "reasoning regimes"?
    # Or use all to see if regimes emerge regardless?
    # User: "Data: Reuse existing... Multiple contents... Do not mask instructions".
    # Let's use all samples but maybe report if cue_found affects it.
    # For now, use ALL.
    
    # Filter only valid samples
    n_valid = len(metadata)
    print(f"  Valid samples: {n_valid}")
    
    if n_valid == 0:
        return None

    # Slice strictly the valid portion
    X_traj = np.array(traj_data[:n_valid, layer_idx, :], dtype=np.float32) # (N_valid, D_TOTAL)
    X_ctx = np.array(context_data[:n_valid], dtype=np.float32) # (N_valid, D_MODEL)
    
    # Check for NaN/Inf
    if np.any(np.isnan(X_traj)) or np.any(np.isinf(X_traj)):
        print("  Warning: NaNs or Infs in trajectory data. Replacing with 0.")
        X_traj = np.nan_to_num(X_traj)
        
    # Normalize Trajectory Features for Clustering
    scaler_traj = StandardScaler()
    X_traj_norm = scaler_traj.fit_transform(X_traj)
    
    # 2. Unsupervised Regime Discovery (Clustering)
    best_k = -1
    best_score = -1
    best_labels = None
    best_model = None
    
    scores = {}
    print("  Clustering (K=4..10)...")
    for k in K_RANGE:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_traj_norm)
        try:
            sil = silhouette_score(X_traj_norm, labels, sample_size=1000)
        except:
            sil = 0
        scores[k] = sil
        if sil > best_score:
            best_score = sil
            best_k = k
            best_labels = labels
            best_model = kmeans
            
    print(f"  Best K={best_k} (Silhouette={best_score:.3f})")
    
    # Save Centroids
    if best_model is not None:
        cen_path = os.path.join(OUTPUT_DIR, f"exp8_cluster_centroids_layer_{layer_idx}.npy")
        np.save(cen_path, best_model.cluster_centers_)
    
    # 3. Language -> Regime Prediction
    # Train Linear Probe: Context Embed -> Cluster ID
    # Use Stratified 5-Fold
    print("  Predicting Regimes from Language...")
    
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    X_ctx_norm = StandardScaler().fit_transform(X_ctx)
    
    # Cross Validation
    cv_scores = cross_val_score(clf, X_ctx_norm, best_labels, cv=5, scoring='accuracy')
    mean_acc = np.mean(cv_scores)
    
    # Chance Baseline
    # Most frequent class baseline
    vals, counts = np.unique(best_labels, return_counts=True)
    majority_acc = np.max(counts) / len(best_labels)
    
    print(f"  Prediction Accuracy: {mean_acc:.3f} (Chance: {majority_acc:.3f})")
    
    return {
        "layer": layer_idx,
        "best_k": best_k,
        "silhouette": best_score,
        "pred_acc": mean_acc,
        "chance_acc": majority_acc,
        "k_scores": scores
    }

def main():
    print("Entered main", flush=True)
    ensure_dir(OUTPUT_DIR)
    
    traj, ctx, meta = load_data()
    if traj is None: return
    
    # Ensure fully generated (wait for extraction to finish if running?)
    # This script assumes data is ready.
    # Check shape
    if len(meta) < N_SAMPLES:
        print(f"Warning: Only {len(meta)} samples found (Metadata).")
    
    results = {}
    
    # Analyze Layers of Interest
    # Layer 0 (Lexical Baseline)
    results[0] = analyze_layer(0, traj, ctx, meta, "Input Embeddings")
    
    # Layer 13 (Mid-Layer)
    results[13] = analyze_layer(13, traj, ctx, meta, "Mid-Layer")
    
    # Layer 24 (Output Baseline)
    results[24] = analyze_layer(24, traj, ctx, meta, "Final Layer")
    
    # Save Results
    with open(os.path.join(OUTPUT_DIR, "exp8_regime_stats.json"), 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
        
    # Generate Report MD
    md = "# Experiment 8: Regime Discovery Report\n\n"
    md += "| Layer | Best K | Silhouette | Prediction Acc | Chance Acc | Delta |\n"
    md += "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
    
    for l in [0, 13, 24]:
        r = results[l]
        delta = r['pred_acc'] - r['chance_acc']
        md += f"| {l} | {r['best_k']} | {r['silhouette']:.3f} | {r['pred_acc']:.3f} | {r['chance_acc']:.3f} | +{delta:.3f} |\n"
        
    md += "\n## Interpretation\n\n"
    
    # Logic
    mid = results[13]
    l0 = results[0]
    
    if mid['pred_acc'] > (l0['pred_acc'] + 0.05) and mid['silhouette'] > 0.15:
        res = "SUCCESS: Language reliably induces mid-layer dynamical regimes."
        rec = "Proceed to Experiment 9."
    elif mid['pred_acc'] > mid['chance_acc']:
        res = "PARTIAL: Regimes exist but prediction is weak or similar to Layer 0."
        rec = "Investigate further."
    else:
        res = "FAILURE: No predictable regimes found."
        rec = "Terminate."
        
    md += f"**Result**: {res}\n\n"
    md += f"**Recommendation**: {rec}\n"
    
    with open(os.path.join(OUTPUT_DIR, "exp8_regime_discovery_report.md"), 'w') as f:
        f.write(md)
        
    print(f"\nAnalysis complete. Report saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
