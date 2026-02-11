import numpy as np
import os
import time
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import shuffle

# --- Configuration ---
DATASET_PATH = "experiments/experiment 6/data/pilot/pilot_dataset.npz"
FEATURES_PATH = "experiments/experiment 6/data/pilot/pilot_features.npz"
OUTPUT_PATH = "experiments/experiment 6/data/pilot/pilot_lopo_results.json"
RANDOM_SEED = 42

def main():
    print("--- Pilot Cross-Paraphrase Probe (Leave-One-Paraphrase-Out) ---")
    start_time = time.time()
    
    # 1. Load Data
    if not os.path.exists(DATASET_PATH) or not os.path.exists(FEATURES_PATH):
        print("ERROR: Missing dataset or features file.")
        return

    # Load IDs from dataset
    ds = np.load(DATASET_PATH)
    # We need paraphrase_ids from metadata since it wasn't saved as a top-level array in pilot_dataset.npz originally?
    # Wait, in pilot_run_extraction.py I saved: operator_ids, content_ids, metadata (json string)
    # I did NOT save paraphrase_ids as a top level array. I need to parse metadata.
    
    operator_ids = ds['operator_ids']
    metadata_json = ds['metadata']
    
    # Parse metadata to get paraphrase_ids
    # Metadata is a string of json list
    meta_list = json.loads(str(metadata_json))
    paraphrase_ids = np.array([m['paraphrase_id'] for m in meta_list])
    
    # Load features
    ft = np.load(FEATURES_PATH)
    features = ft['features'] # (N, L, F)
    
    # Validation
    N, L, F = features.shape
    print(f"Loaded: N={N}, L={L}, F={F}")
    print(f"Unique Paraphrases: {np.unique(paraphrase_ids)}")
    print(f"Unique Operators: {np.unique(operator_ids)}")
    
    assert len(paraphrase_ids) == N
    assert len(operator_ids) == N
    
    # 2. LOPO Cross-Validation
    unique_paras = np.sort(np.unique(paraphrase_ids)) # [0, 1, 2]
    num_folds = len(unique_paras)
    
    # Structure to hold results: layer -> list of fold metrics
    layer_metrics = {l: {'acc': [], 'f1': []} for l in range(L)}
    fold_details = []
    
    print(f"\nStarting {num_folds}-fold LOPO evaluation...")
    
    for fold_idx, hold_para in enumerate(unique_paras):
        # Split
        test_mask = (paraphrase_ids == hold_para)
        train_mask = ~test_mask
        
        X_train_fold = features[train_mask] # (N_train, L, F)
        y_train_fold = operator_ids[train_mask]
        
        X_test_fold = features[test_mask]   # (N_test, L, F)
        y_test_fold = operator_ids[test_mask]
        
        # Track best per fold
        fold_best_f1 = -1
        fold_best_layer = -1
        
        # Train per layer
        for l in range(L):
            clf = LogisticRegression(
                class_weight='balanced', 
                solver='lbfgs', 
                max_iter=200, 
                random_state=RANDOM_SEED
            )
            
            clf.fit(X_train_fold[:, l, :], y_train_fold)
            y_pred = clf.predict(X_test_fold[:, l, :])
            
            acc = accuracy_score(y_test_fold, y_pred)
            f1 = f1_score(y_test_fold, y_pred, average='macro')
            
            layer_metrics[l]['acc'].append(acc)
            layer_metrics[l]['f1'].append(f1)
            
            if f1 > fold_best_f1:
                fold_best_f1 = f1
                fold_best_layer = l
        
        print(f"Fold {fold_idx+1}/{num_folds} (Para {hold_para}): Best Layer {fold_best_layer} (F1: {fold_best_f1:.2f})")
        
        fold_details.append({
            'fold': int(fold_idx),
            'held_out_paraphrase': int(hold_para),
            'best_layer': int(fold_best_layer),
            'best_f1': float(fold_best_f1)
        })

    # 3. Aggregation & Summary
    print("\n--- Summary Results (Mean across 3 folds) ---")
    print(f"{'Layer':<6} | {'Mean F1':<8} | {'Std F1':<8}")
    print("-" * 30)
    
    agg_results = {}
    best_layer_overall = -1
    best_mean_f1 = -1
    
    for l in range(L):
        mean_f1 = np.mean(layer_metrics[l]['f1'])
        std_f1 = np.std(layer_metrics[l]['f1'])
        
        agg_results[l] = {
            'mean_f1': float(mean_f1),
            'std_f1': float(std_f1),
            'mean_acc': float(np.mean(layer_metrics[l]['acc']))
        }
        
        marker = ""
        if mean_f1 > best_mean_f1:
            best_mean_f1 = mean_f1
            best_layer_overall = l
            marker = "*"
            
        print(f"{l:<6} | {mean_f1:.3f}    | {std_f1:.3f} {marker}")

    print("-" * 30)
    print(f"Best Layer Overall: {best_layer_overall} (Mean F1: {best_mean_f1:.3f})")
    print(f"Chance Baseline: {1.0/3.0:.2f}")

    # 4. Permutation Control (Sanity Check)
    print("\n--- Permutation Control (Para 0 held out) ---")
    # Hold out para 0 again
    hold_para = unique_paras[0]
    test_mask = (paraphrase_ids == hold_para)
    train_mask = ~test_mask
    
    # Shuffle labels
    y_train_shuffled = shuffle(operator_ids[train_mask], random_state=RANDOM_SEED)
    y_test_true = operator_ids[test_mask]
    
    # Train on BEST LAYER only
    l = best_layer_overall
    clf_perm = LogisticRegression(class_weight='balanced', max_iter=200, random_state=RANDOM_SEED)
    clf_perm.fit(features[train_mask][:, l, :], y_train_shuffled)
    y_pred_perm = clf_perm.predict(features[test_mask][:, l, :])
    
    perm_f1 = f1_score(y_test_true, y_pred_perm, average='macro')
    print(f"Permuted F1 (Layer {l}): {perm_f1:.3f} (Expected ~0.33)")

    # 5. Save Artifacts
    full_output = {
        'fold_details': fold_details,
        'layer_metrics_raw': layer_metrics,
        'aggregate_results': agg_results,
        'permutation_control': {
            'layer': int(l),
            'f1': float(perm_f1)
        }
    }
    
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(full_output, f, indent=2)
        
    print(f"\nResults saved to {OUTPUT_PATH}")
    print(f"Total Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
