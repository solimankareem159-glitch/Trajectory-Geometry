import numpy as np
import json
import os
import time
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import f1_score
from scipy.spatial.distance import cdist
from scipy.linalg import subspace_angles

# --- Configuration ---
DATA_DIR = "experiments/experiment 6/data/full_2k"
OUTPUT_DIR = "experiments/experiment 7/results/geometry_metrics"
FEATURES_PATH = os.path.join(DATA_DIR, "features.memmap")
LABELS_PATH = os.path.join(DATA_DIR, "labels.npy")
IDS_PATH = os.path.join(DATA_DIR, "ids.npy")
PROGRESS_PATH = os.path.join(DATA_DIR, "progress.json")

N_MAX = 2000
L_LAYERS = 25
FEATURE_DIM = 1792
LAYERS_TO_ANALYZE = [0, 10, 11, 12, 13, 14, 15, 16, 24]
N_OPS = 10

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def main():
    print("--- Experiment 7 (Partial) - Operator Geometry ---")
    start_time = time.time()
    ensure_dir(OUTPUT_DIR)

    # 1. Load Data
    if not os.path.exists(PROGRESS_PATH):
        print("Progress file not found. Is the full run started?")
        return

    with open(PROGRESS_PATH, 'r') as f:
        prog = json.load(f)
        n_so_far = prog.get('completed_samples', 0)

    N = min(N_MAX, n_so_far)
    print(f"Loading first {N} samples (Available: {n_so_far})...")

    if N < 50:
        print("Not enough samples (<50) to run geometry analysis.")
        return

    # Load artifacts
    # Use mode='c' (copy-on-write) or 'r' to avoid locking issues if writer is active
    # Shape is (Total, L, D) but we only need first N
    # We must know the TOTAL size used in full run allocation to map it correctly?
    # In full run: shape=(2000, 25, 1792).
    TOTAL_ALLOC = 2000
    
    try:
        features_all = np.memmap(FEATURES_PATH, dtype='float16', mode='r', shape=(TOTAL_ALLOC, L_LAYERS, FEATURE_DIM))
        features = features_all[:N].copy().astype(np.float32) # Copy to memory + float32 for math
        del features_all # Release memmap handle
    except Exception as e:
        print(f"Error loading memmap: {e}")
        return

    labels = np.load(LABELS_PATH)[:N]
    ids = np.load(IDS_PATH)[:N] # col0=content, col1=para
    content_ids = ids[:, 0]
    
    # Check Class Balance
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Class distribution: {dict(zip(unique, counts))}")

    report = {"n_samples": N, "layers": {}}

    # 2. Layer-wise Geometry
    for l in LAYERS_TO_ANALYZE:
        print(f"\nAnalyzing Layer {l}...")
        X = features[:, l, :] # (N, D)
        
        # A) Centroids
        centroids = []
        for op in range(N_OPS):
            mask = (labels == op)
            if np.sum(mask) > 0:
                centroids.append(np.mean(X[mask], axis=0))
            else:
                centroids.append(np.zeros(FEATURE_DIM)) # Should not happen if N is large enough
        centroids = np.stack(centroids) # (10, D)
        
        # Save Centroids
        np.save(os.path.join(OUTPUT_DIR, f"centroids_layer_{l}.npy"), centroids)
        
        # B) Distances
        # Cosine distance = 1 - cosine_similarity
        dist_cos = cdist(centroids, centroids, metric='cosine')
        dist_euc = cdist(centroids, centroids, metric='euclidean')
        
        np.save(os.path.join(OUTPUT_DIR, f"cosine_dist_layer_{l}.npy"), dist_cos)
        
        # Mean off-diagonal distance
        mask_off_diag = ~np.eye(dist_cos.shape[0], dtype=bool)
        mean_sep_cos = np.mean(dist_cos[mask_off_diag])
        
        # C) Stability (Cross-Content)
        # For each operator, compute centroid per content, measure global coherence
        stability_scores = []
        unique_contents = np.unique(content_ids)
        
        for op in range(N_OPS):
            op_mask = (labels == op)
            if np.sum(op_mask) < 2:
                stability_scores.append(0.0)
                continue
            
            # Get operator vectors
            X_op = X[op_mask]
            c_op = content_ids[op_mask]
            
            # Compute content-specific centroids
            content_centroids = []
            valid_c = []
            for c in unique_contents:
                c_mask = (c_op == c)
                if np.sum(c_mask) > 0:
                    content_centroids.append(np.mean(X_op[c_mask], axis=0))
                    valid_c.append(c)
            
            if len(content_centroids) < 2:
                stability_scores.append(0.0) # Can't measure stability with 1 content
                continue
                
            content_centroids = np.stack(content_centroids)
            # Pairwise cosine similarity between all content centroids for this operator
            # 1 - cosine_dist
            sim_matrix = 1 - cdist(content_centroids, content_centroids, metric='cosine')
            # Mean off-diagonal similarity
            n_c = len(content_centroids)
            if n_c > 1:
                off_diag_sim = np.sum(sim_matrix) - n_c # Sum of all - trace(1s)
                mean_sim = off_diag_sim / (n_c * (n_c - 1))
            else:
                mean_sim = 1.0
            
            stability_scores.append(mean_sim)
            
        mean_stability = np.mean(stability_scores)
        
        # D) Subspace Geometry (Principal Angles)
        # PCA on each operator
        subspaces = []
        for op in range(N_OPS):
            mask = (labels == op)
            X_op = X[mask]
            if X_op.shape[0] >= 5:
                pca = PCA(n_components=5)
                pca.fit(X_op)
                subspaces.append(pca.components_) # (5, D)
            else:
                subspaces.append(None)
        
        # Pairwise max principal angle (or min similarity)
        # We'll compute min angle (closest approach) converted to cosine
        # subspace_angles returns angles in radians [0, pi/2]
        # We want similarity: cosmetic(mean_angle?) or just largest angle?
        # Let's save the angle matrix
        angle_matrix = np.zeros((N_OPS, N_OPS))
        for i in range(N_OPS):
            for j in range(N_OPS):
                if subspaces[i] is not None and subspaces[j] is not None:
                    # subspaces are (5, D). scipy expects (D, 5) column bases?
                    # "b[i] column is a vector". so transpose.
                    angles = subspace_angles(subspaces[i].T, subspaces[j].T)
                    # small angle = similar.
                    # metric mean angle
                    angle_matrix[i, j] = np.mean(angles)
                else:
                    angle_matrix[i, j] = np.pi/2 # max orthogonality
        
        np.save(os.path.join(OUTPUT_DIR, f"subspace_angles_layer_{l}.npy"), angle_matrix)
        
        # E) Nearest Centroid SANITY CHECK
        clf = NearestCentroid()
        # manual split (consistent)
        indices = np.arange(N)
        np.random.shuffle(indices) # actually shuffle for split
        split = int(0.7 * N)
        train_idx, test_idx = indices[:split], indices[split:]
        
        clf.fit(X[train_idx], labels[train_idx])
        y_pred = clf.predict(X[test_idx])
        f1 = f1_score(labels[test_idx], y_pred, average='macro')
        
        # Logging
        print(f"  Sep (Cos): {mean_sep_cos:.3f} | Stable: {mean_stability:.3f} | NCC F1: {f1:.3f}")
        
        report["layers"][l] = {
            "separation_cosine": float(mean_sep_cos),
            "stability_score": float(mean_stability),
            "ncc_f1": float(f1),
            "top_pairs_cos": [], # TODO if needed, but summary is enough
            "mean_subspace_angle": float(np.mean(angle_matrix[~np.eye(N_OPS, dtype=bool)]))
        }

    # Best geometry layer (simple heuristic max statbility * separation)
    best_l = max(report["layers"], key=lambda x: report["layers"][x]["stability_score"])
    report["best_stability_layer"] = best_l
    
    with open(os.path.join(OUTPUT_DIR, "exp7_partial_report.json"), 'w') as f:
        json.dump(report, f, indent=2)
        
    print(f"\nSaved results to {OUTPUT_DIR}")
    print(f"Total time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
