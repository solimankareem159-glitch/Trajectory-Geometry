"""
Generate Visualization Data (Qwen 0.5B - Exp 14/15)
===================================================
Aggregates metrics, hidden states, and analysis extensions into a single JSON file.
Output: visualisation_tool/data/trajectory_data.json

Inputs:
- Experiment 14 metrics (exp14_metrics.csv)
- Experiment 14 metadata (metadata.csv)
- Experiment 14 hidden states (*.npy)
- Experiment 9 dataset (exp9_dataset.jsonl) - for Truth/Difficulty values
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from tqdm import tqdm

# Paths
ROOT_DIR = os.getcwd()
VIS_DATA_DIR = os.path.join(ROOT_DIR, "visualisation_tool", "data")
EXP14_DIR = os.path.join(ROOT_DIR, "experiments", "EXP-14_UniversalSignature_2025-12-03", "data")
EXP9_DIR = os.path.join(ROOT_DIR, "experiments", "EXP-09_GeometryCapability_2025-11-22", "data")

METRICS_FILE = os.path.join(EXP14_DIR, "exp14_metrics.csv")
METADATA_FILE = os.path.join(EXP14_DIR, "metadata.csv")
HIDDEN_DIR = os.path.join(EXP14_DIR, "hidden_states")
DATASET_FILE = os.path.join(EXP9_DIR, "exp9_dataset.jsonl")

OUTPUT_FILE = os.path.join(VIS_DATA_DIR, "trajectory_data.json")

def load_data():
    print("Loading datasets...")
    
    # Load Metadata
    df_meta = pd.read_csv(METADATA_FILE)
    print(f"Loaded {len(df_meta)} metadata records")
    
    # Load Metrics
    df_metrics = pd.read_csv(METRICS_FILE)
    print(f"Loaded {len(df_metrics)} metric rows")
    
    # Load Dataset (for Difficulty)
    difficulty_map = {}
    if os.path.exists(DATASET_FILE):
        print(f"Loading dataset from {DATASET_FILE}")
        with open(DATASET_FILE, 'r') as f:
            for line in f:
                rec = json.loads(line)
                pid = rec['id']
                truth = abs(rec['truth'])
                
                # Assign tier
                if truth < 100:
                    tier = "Small"
                elif truth < 1000:
                    tier = "Medium"
                else:
                    tier = "Large"
                difficulty_map[pid] = tier
        print("Loaded difficulty tiers from dataset")
    else:
        print("Warning: Dataset file not found")
        
    return df_meta, df_metrics, difficulty_map

def compute_global_pca(df_meta):
    print("\nComputing Global PCA...")
    
    X_all = []
    sample_registry = []
    
    print("Loading all hidden states for PCA...")
    sample_lengths = [] # Track number of vectors per sample

    for i, rec in enumerate(tqdm(df_meta.to_dict('records'))):
        npy_path = os.path.join(HIDDEN_DIR, rec['filename'])
        try:
            h = np.load(npy_path) # [25, T, 896]
            # Flatten layer and token dims: [25*T, 896]
            flat_h = h.reshape(-1, 896)
            X_all.append(flat_h) 
            sample_registry.append(rec)
            sample_lengths.append(flat_h.shape[0])
        except Exception as e:
            # print(f"Skipping {rec['filename']}: {e}")
            pass
            
    if not X_all:
        print("No hidden states loaded!")
        return {}

    X_concat = np.concatenate(X_all, axis=0)
    print(f"Data shape for PCA: {X_concat.shape}") 
    
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_concat)
    
    print(f"PCA Explained Variance: {pca.explained_variance_ratio_}")
    
    # Split back into samples
    pca_data_map = {}
    cursor = 0
    
    for i, rec in enumerate(sample_registry):
        length = sample_lengths[i]
        sample_pca = X_pca[cursor : cursor + length] # [L, 3]
        
        # Reshape back to [25, T, 3]
        # We know n_layers is 25. T = length / 25
        n_layers = 25
        n_tokens = length // n_layers
        
        try:
            sample_pca_reshaped = sample_pca.reshape(n_layers, n_tokens, 3)
            pca_data_map[(rec['problem_id'], rec['condition'])] = sample_pca_reshaped
        except ValueError as e:
             print(f"Error reshaping {rec['filename']} (len {length}): {e}")
        
        cursor += length
        
    return pca_data_map

def generate_json(df_meta, df_metrics, pca_map, difficulty_map):
    print("\nGenerating JSON structure...")
    
    output_data = {"trajectories": []}
    
    print("Indexing metrics...")
    df_metrics['lookup_key'] = list(zip(df_metrics['problem_id'], df_metrics['condition'], df_metrics['layer']))
    metrics_lookup = df_metrics.set_index('lookup_key').to_dict('index')
    
    for i, rec in enumerate(tqdm(df_meta.to_dict('records'), desc="Building JSON")):
        pid = rec['problem_id']
        cond = rec['condition']
        
        # Get PCA data
        if (pid, cond) not in pca_map:
            continue
            
        pca_cube = pca_map[(pid, cond)] # [25, 32, 3]
        
        traj_entry = {
            "id": f"problem_{pid:03d}_{cond}",
            "problem_id": pid,
            "condition": cond,
            "group": rec['group'],
            "correct": bool(rec['correct']),
            "difficulty": difficulty_map.get(pid, "Unknown"),
            "layers": []
        }
        
        for layer_idx in range(rec['n_layers']):
            met_key = (pid, cond, layer_idx)
            layer_metrics = metrics_lookup.get(met_key, {})
            clean_metrics = {k: v for k, v in layer_metrics.items() if k not in ['problem_id', 'condition', 'layer', 'lookup_key', 'group', 'correct']}
            
            pca_path = pca_cube[layer_idx].tolist()
            
            layer_entry = {
                "layer_idx": layer_idx,
                "pca_path": pca_path, 
                "metrics": clean_metrics
            }
            
            traj_entry["layers"].append(layer_entry)
            
        output_data["trajectories"].append(traj_entry)
        
    return output_data

def main():
    if not os.path.exists(VIS_DATA_DIR):
        os.makedirs(VIS_DATA_DIR)
        
    df_meta, df_metrics, diff_map = load_data()
    pca_map = compute_global_pca(df_meta)
    json_data = generate_json(df_meta, df_metrics, pca_map, diff_map)
    
    print(f"Saving {len(json_data['trajectories'])} trajectories to {OUTPUT_FILE}...")
    
    # Sanitize NaN/Infinity -> None (null in JSON)
    def sanitize_floats(obj):
        if isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return obj
        elif isinstance(obj, dict):
            return {k: sanitize_floats(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize_floats(v) for v in obj]
        return obj

    clean_data = sanitize_floats(json_data)
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(clean_data, f)
    
    print("Done!")

if __name__ == "__main__":
    main()
