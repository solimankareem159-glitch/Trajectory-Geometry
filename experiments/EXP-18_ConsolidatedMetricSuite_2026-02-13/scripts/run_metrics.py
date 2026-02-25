
import os
import glob
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import sys
import re
import traceback
from transformers import AutoTokenizer

# Add scripts to path
sys.path.append(os.path.join(os.path.dirname(__file__)))
from metric_suite import TrajectoryMetrics, cross_layer_metrics, compute_all_metrics

def get_numbers(s):
    return [int(n) for n in re.findall(r'\d+', s)]

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "Data")
    hidden_dir = os.path.join(data_dir, "hidden_states")
    metadata_path = os.path.join(data_dir, "metadata.csv")
    dataset_path = os.path.join(data_dir, "exp9_dataset.jsonl")
    
    unembedding_path = os.path.join(data_dir, "unembedding.npy")
    embedding_path = os.path.join(data_dir, "embedding.npy")
    tokenizer_dir = os.path.join(data_dir, "tokenizer")
    
    output_path = os.path.join(base_dir, "results", "exp18_metrics_full.csv")
    checkpoint_path = os.path.join(base_dir, "results", "exp18_metrics_full_checkpoint.csv")
    
    os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)

    print("Loading weights and tokenizer...")
    W_U = np.load(unembedding_path).astype(np.float32)
    W_E = np.load(embedding_path).astype(np.float32)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    
    print("Precomputing Unembedding Gram Matrix (Optimization)...")
    # W_U is (V, D). We want G = W_U.T @ W_U -> (D, D)
    # This turns O(V) projection into O(D) vector-matrix mult.
    G_U = W_U.T @ W_U
    
    tm = TrajectoryMetrics(vocab_unbedding=W_U, embedding_matrix=W_E, gram_matrix=G_U)

    print(f"Loading metadata from {metadata_path}...")
    df = pd.read_csv(metadata_path)
    
    with open(dataset_path, 'r') as f:
        problems = [json.loads(line) for line in f]
    problem_map = {p['id']: p for p in problems}

    # Pass 1: Centroids
    # ... assuming they are computed fast enough or skip if redundant?
    # Actually we need them for Family 8.
    print("Computing Success Centroids (Pass 1)...")
    success_mask = (df['condition'] == 'cot') & (df['correct'] == True)
    success_files = df[success_mask]['filename'].tolist()
    layer_sums = {} ; layer_counts = {}
    for fname in tqdm(success_files, desc="Centroids"):
        fpath = os.path.join(hidden_dir, fname)
        if not os.path.exists(fpath): continue
        h_stack = np.load(fpath).astype(np.float32)
        L, T, D = h_stack.shape
        for l in range(L):
            h = h_stack[l]
            if l not in layer_sums:
                layer_sums[l] = np.zeros((1024, D))
                layer_counts[l] = np.zeros((1024, 1))
            T_eff = min(T, 1024)
            layer_sums[l][:T_eff] += h[:T_eff]
            layer_counts[l][:T_eff] += 1
    centroids = {}
    for l in layer_sums:
        cnt = layer_counts[l]
        valid = cnt > 0
        mean = np.zeros_like(layer_sums[l])
        mean[np.squeeze(valid)] = layer_sums[l][np.squeeze(valid)] / cnt[np.squeeze(valid)]
        centroids[l] = mean

    # Load Resume State
    completed_files = set()
    if os.path.exists(output_path):
        c_df = pd.read_csv(output_path)
        completed_files = set(c_df['filename'].unique())
        print(f"Resuming from {len(completed_files)} completed trajectories.")
    elif os.path.exists(checkpoint_path):
        c_df = pd.read_csv(checkpoint_path)
        completed_files = set(c_df['filename'].unique())
        print(f"Resuming from {len(completed_files)} checkpointed trajectories.")

    # Pass 2: Metrics
    print("Computing All Metrics (Pass 2)...")
    results = []
    
    # Open file for writing if resuming
    file_exists = os.path.exists(output_path)
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Metrics"):
        fname = row['filename']
        if fname in completed_files: continue
        
        fpath = os.path.join(hidden_dir, fname)
        if not os.path.exists(fpath): continue
        
        try:
            h_stack = np.load(fpath).astype(np.float32)
            prob = problem_map[row['problem_id']]
            
            truth = str(prob['truth'])
            truth_id = tokenizer.encode(truth, add_special_tokens=False)[0]
            ops = get_numbers(prob['question'])
            op_ids = [tokenizer.encode(str(o), add_special_tokens=False)[0] for o in ops]
            
            intermediate_id = None
            if len(ops) >= 2:
                 match = re.search(r'\((\d+)\s*[\*\/]\s*(\d+)\)', prob['question'])
                 if match:
                      val = eval(match.group(0))
                      intermediate_id = tokenizer.encode(str(val), add_special_tokens=False)[0]
            
            context = {
                'truth_id': truth_id,
                'wrong_id': tokenizer.encode(str(int(prob['truth'])+1), add_special_tokens=False)[0],
                'operand_ids': op_ids,
                'intermediate_id': intermediate_id,
                'centroids': centroids
            }
            
            traj_results = compute_all_metrics(h_stack, tm, context)
            
            curr_batch = []
            for m_row in traj_results:
                entry = row.to_dict()
                entry.update(m_row)
                curr_batch.append(entry)
            
            # Save batch to disk immediately to avoid total loss
            batch_df = pd.DataFrame(curr_batch)
            batch_df.to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)
            completed_files.add(fname)
            
        except Exception as e:
            print(f"\nError processing {fname}: {e}")
            traceback.print_exc()
            continue

    print("Computation complete.")

if __name__ == "__main__":
    main()
