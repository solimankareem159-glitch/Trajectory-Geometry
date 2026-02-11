import numpy as np
import os
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score
from sklearn.random_projection import GaussianRandomProjection

# --- Configuration ---
DATA_PATH = "experiments/experiment 6/data/pilot/pilot_dataset.npz"
OUTPUT_PATH = "experiments/experiment 6/data/pilot/pilot_features.npz"
RANDOM_SEED = 42
TEST_SIZE_RATIO = 0.3
DO_PROJECTION = False # Set to True if memory/speed is an issue, but for 45 samples it's fine without.
PROJECTION_DIM = 256

def main():
    print("--- Pilot Probe Sanity Check ---")
    start_time = time.time()
    
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        return
        
    data = np.load(DATA_PATH)
    X_raw = data['hidden_states'] # (45, 25, 32, 896)
    y = data['operator_ids']      # (45,)
    
    print(f"Loaded dataset: X={X_raw.shape}, y={y.shape}")
    
    N, L, T, D = X_raw.shape
    
    # 2. Feature Extraction
    # We want features per layer.
    # Feature logic:
    # H: (T, D)
    # DH[t] = H[t] - H[t-1]  (t=1..31) -> (31, D)
    # DHn[t] = DH[t] / (norm + 1e-8)
    # feat = [mean(DHn), var(DHn)] -> (2D)
    
    print("Extracting features...")
    
    feat_start = time.time()
    
    # Vectorized computation preferred, but loop over N, L ok for clarity/memory
    # Or vectorize over N and L if fits in memory.
    # X_raw: (N, L, T, D)
    
    # Shift for deltas
    # H[t] - H[t-1]
    # Slice 1: (N, L, 1..31, D)
    # Slice 2: (N, L, 0..30, D)
    
    H_t = X_raw[:, :, 1:, :] 
    H_t_minus_1 = X_raw[:, :, :-1, :]
    
    DH = H_t - H_t_minus_1 # (N, L, 31, D)
    
    # Normalize
    # Norm over dim D
    norms = np.linalg.norm(DH, axis=-1, keepdims=True) # (N, L, 31, 1)
    DHn = DH / (norms + 1e-8)
    
    # Aggregate
    mu = np.mean(DHn, axis=2) # (N, L, D)
    var = np.var(DHn, axis=2) # (N, L, D)
    
    # Concat
    features = np.concatenate([mu, var], axis=-1) # (N, L, 2D) (N, L, 1792)
    
    print(f"Features extracted in {time.time() - feat_start:.2f}s. Shape: {features.shape}")
    
    # Optional Projection
    if DO_PROJECTION:
        print(f"Projecting to {PROJECTION_DIM} dims...")
        # Reshape to (N*L, 2D) for projection fitting
        flat_feat = features.reshape(-1, features.shape[-1])
        grp = GaussianRandomProjection(n_components=PROJECTION_DIM, random_state=RANDOM_SEED)
        flat_proj = grp.fit_transform(flat_feat)
        features = flat_proj.reshape(N, L, PROJECTION_DIM)
        print(f"Projected shape: {features.shape}")
    
    # 3. Training Loop (Per Layer)
    print("\n--- Probe Results (Layer-wise) ---")
    print(f"{'Layer':<6} | {'Acc':<6} | {'F1-Mac':<8}")
    print("-" * 28)
    
    results = []
    
    # Split once for consistency
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE_RATIO, random_state=RANDOM_SEED)
    train_idx, test_idx = next(sss.split(features[:, 0, :], y))
    
    y_train, y_test = y[train_idx], y[test_idx]
    
    best_layer = -1
    best_f1 = -1.0
    
    for layer_idx in range(L):
        X_layer = features[:, layer_idx, :]
        X_train, X_test = X_layer[train_idx], X_layer[test_idx]
        
        clf = LogisticRegression(class_weight='balanced', solver='lbfgs', max_iter=200, random_state=RANDOM_SEED)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        
        results.append((layer_idx, acc, f1))
        
        if f1 > best_f1:
            best_f1 = f1
            best_layer = layer_idx
            
        print(f"{layer_idx:<6} | {acc:.2f}   | {f1:.2f}")
        
    print("-" * 28)
    print(f"Best Layer: {best_layer} (F1: {best_f1:.2f})")
    print(f"Chance Baseline: {1.0/3.0:.2f}")
    
    # 4. Save Features
    print(f"\nSaving features to {OUTPUT_PATH}...")
    np.savez_compressed(OUTPUT_PATH, features=features, labels=y)
    
    print(f"Total time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
