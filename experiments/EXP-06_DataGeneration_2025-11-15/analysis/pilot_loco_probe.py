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
OUTPUT_PATH = "experiments/experiment 6/data/pilot/pilot_loco_results.json"
RANDOM_SEED = 42

def main():
    print("--- Pilot Cross-Topic Probe (Leave-One-Content-Out) ---")
    start_time = time.time()
    
    # 1. Load Data
    if not os.path.exists(DATASET_PATH) or not os.path.exists(FEATURES_PATH):
        print("ERROR: Missing dataset or features file.")
        return

    # Load IDs from dataset
    ds = np.load(DATASET_PATH)
    content_ids = ds['content_ids']
    operator_ids = ds['operator_ids']
    
    # Load features
    ft = np.load(FEATURES_PATH)
    features = ft['features'] # (N, L, F)
    
    # Validation
    N, L, F = features.shape
    print(f"Loaded: N={N}, L={L}, F={F}")
    print(f"Unique Contents: {np.unique(content_ids)}")
    print(f"Unique Operators: {np.unique(operator_ids)}")
    
    assert len(content_ids) == N
    assert len(operator_ids) == N
    
    # 2. LOCO Cross-Validation
    unique_contents = np.sort(np.unique(content_ids))
    num_folds = len(unique_contents)
    
    # Structure to hold results: layer -> list of fold metrics
    layer_metrics = {l: {'acc': [], 'f1': []} for l in range(L)}
    fold_details = []
    
    print(f"\nStarting {num_folds}-fold LOCO evaluation...")
    
    for fold_idx, hold_content in enumerate(unique_contents):
        # Split
        test_mask = (content_ids == hold_content)
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
        
        print(f"Fold {fold_idx+1}/{num_folds} (Content {hold_content}): Best Layer {fold_best_layer} (F1: {fold_best_f1:.2f})")
        
        fold_details.append({
            'fold': int(fold_idx),
            'held_out_content': int(hold_content),
            'best_layer': int(fold_best_layer),
            'best_f1': float(fold_best_f1)
        })

    # 3. Aggregation & Summary
    print("\n--- Summary Results (Mean across 5 folds) ---")
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

    # 4. Permutation Control (Sanity Check)
    print("\n--- Permutation Control (Content 0 held out) ---")
    # Hold out content 0 again
    hold_content = unique_contents[0]
    test_mask = (content_ids == hold_content)
    train_mask = ~test_mask
    
    # Shuffle labels
    y_train_shuffled = shuffle(operator_ids[train_mask], random_state=RANDOM_SEED)
    y_test_true = operator_ids[test_mask]
    
    # Train on BEST LAYER only for efficiency
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
