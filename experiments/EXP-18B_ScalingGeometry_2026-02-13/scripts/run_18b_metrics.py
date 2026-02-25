# IMPORTS: Minimal set for all processes
import os
import sys
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import gc

# PID printing for main and workers
print(f"PID: {os.getpid()} (Main/Worker)", flush=True)

# Add local path for metrics
sys.path.append(os.path.dirname(__file__))
try:
    from metric_suite import TrajectoryMetrics, compute_all_metrics
except ImportError:
    pass 

# --- Configuration ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# Model Configs
CONFIGS = [
    {
        "name": "qwen_0_5b",
        "model_id": "Qwen/Qwen2.5-0.5B",
        "hidden_dir": os.path.join(ROOT_DIR, "experiments", "EXP-14_UniversalSignature_2025-12-03", "data", "hidden_states_full"),
        "metadata_path": os.path.join(ROOT_DIR, "experiments", "EXP-14_UniversalSignature_2025-12-03", "data", "metadata_full.csv"),
    },
    {
        "name": "qwen_1_5b",
        "model_id": "Qwen/Qwen2.5-1.5B",
        "hidden_dir": os.path.join(ROOT_DIR, "experiments", "EXP-16B_Qwen15B_2025-12-08", "data", "hidden_states_clean"), 
        "metadata_path": os.path.join(ROOT_DIR, "experiments", "EXP-16B_Qwen15B_2025-12-08", "data", "metadata_reparsed.csv"), 
    },
    {
        "name": "pythia_70m",
        "model_id": "EleutherAI/pythia-70m",
        "hidden_dir": os.path.join(ROOT_DIR, "experiments", "EXP-16_Pythia70m_2025-12-07", "data", "hidden_states_clean"),
        "metadata_path": os.path.join(ROOT_DIR, "experiments", "EXP-16_Pythia70m_2025-12-07", "data", "metadata.csv"),
    }
]

# --- Global State for Workers ---
GLOBAL_TM = None

def init_worker(G_U):
    """Initialize the TrajectoryMetrics object once per process. G_U is small."""
    global GLOBAL_TM
    # Workers don't need the full W_U/W_E matrices if we pass vectors in ctx.
    GLOBAL_TM = TrajectoryMetrics(gram_matrix=G_U)

def run_single_file_task(f, hidden_dir, cfg_name, meta_map, t_id, centroids, correct_vec, embed_vec):
    """Task function using small injected vectors instead of full matrices."""
    global GLOBAL_TM
    try:
        clean_f = f.replace("problem_", "").replace(".npy", "")
        pid_str = clean_f.split('_')[0]
        try: pid_str = str(int(pid_str)) 
        except: pass
        
        cond = 'cot' if 'cot' in f else 'direct'
        info = meta_map.get((pid_str, cond), {'correct': False, 'truth': '0'})
        
        h_stack = np.load(os.path.join(hidden_dir, f))
        
        # Context with ONLY the necessary vectors for this file
        ctx = {
            'truth_id': t_id,
            'operand_ids': [], # Skip for now
            'centroids': centroids,
            'correct_vec': correct_vec,
            'embed_vec': embed_vec
        }
        
        row_metrics = compute_all_metrics(h_stack, GLOBAL_TM, ctx)
        for r in row_metrics:
            r['problem_id'] = pid_str
            r['condition'] = cond
            r['correct'] = info['correct']
            r['model'] = cfg_name
            r['filename'] = f
            
        return row_metrics
    except Exception as e:
        import traceback
        print(f"  Error in {f}: {e}")
        traceback.print_exc()
        return []

def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Starting Exp 18B Vector-Injected Multiprocessing...")
    # Cap worker count for stability on Windows with heavy models
    num_cpus = min(4, max(1, multiprocessing.cpu_count() - 2))
    print(f"Workers: {num_cpus}")

    for cfg in CONFIGS:
        out_csv = os.path.join(ROOT_DIR, "experiments", "EXP-18B_ScalingGeometry_2026-02-13", "results", f"{cfg['name']}_metrics_57.csv")
        print(f"\n{'='*40}")
        print(f"Model: {cfg['name']} -> {out_csv}")
        print(f"{'='*40}")
        
        if not os.path.exists(cfg['hidden_dir']):
            print(f"  Error: Hidden dir not found: {cfg['hidden_dir']}")
            continue
            
        # 1. Weights
        G_U, tokenizer = None, None
        W_U, W_E = None, None
        try:
            print(f"  Extracting vectors for {cfg['model_id']}...")
            tokenizer = AutoTokenizer.from_pretrained(cfg['model_id'])
            model = AutoModelForCausalLM.from_pretrained(cfg['model_id'], torch_dtype=torch.float32)
            W_E = model.get_input_embeddings().weight.detach().numpy().astype(np.float32)
            W_U = model.get_output_embeddings().weight.detach().numpy().astype(np.float32)
            G_U = (W_U.T @ W_U).astype(np.float32)
            print(f"  Gram matrix size: {G_U.shape}")
            del model
        except Exception as e:
            print(f"  Warning: Vector extraction failed: {e}")

        files = [f for f in os.listdir(cfg['hidden_dir']) if f.endswith('.npy')]
        print(f"  Found {len(files)} .npy files.")
        
        # RESUME LOGIC
        processed_files = set()
        if os.path.exists(out_csv):
            try:
                existing_df = pd.read_csv(out_csv)
                if 'filename' in existing_df.columns:
                    processed_files = set(existing_df['filename'].unique())
                    print(f"  Resuming: {len(processed_files)} files already processed.")
            except Exception as e:
                print(f"  Warning: Could not read existing CSV ({e}), starting fresh.")

        remaining_files = [f for f in files if f not in processed_files]
        if not remaining_files:
            print(f"  Model {cfg['name']} already fully processed.")
            continue

        print(f"  Remaining to process: {len(remaining_files)}")
        print("  Loading metadata...")
        meta_map = {}
        if os.path.exists(cfg['metadata_path']):
            df_meta = pd.read_csv(cfg['metadata_path'])
            for _, row in df_meta.iterrows():
                pid = str(row.get('problem_id', ''))
                if pid.endswith('.0'): pid = pid[:-2]
                cond = row.get('condition', '')
                meta_map[(pid, cond)] = {
                    'correct': row.get('correct', False), 
                    'truth': str(row.get('ground_truth', row.get('truth', row.get('answer', '0'))))
                }
            print(f"  Loaded {len(meta_map)} meta entries.")
        
        # Inject ground truth for Qwen 0.5B (which uses Exp 9 problems)
        if cfg['name'] == 'qwen_0_5b':
            exp9_path = os.path.join(ROOT_DIR, "experiments", "EXP-18_ConsolidatedMetricSuite_2026-02-13", "Data", "exp9_dataset.jsonl")
            if os.path.exists(exp9_path):
                print(f"  Injecting truth from {os.path.basename(exp9_path)}...")
                with open(exp9_path, 'r') as f:
                    for line in f:
                        item = json.loads(line)
                        pid = str(item['id'])
                        for cond in ['direct', 'cot']:
                            if (pid, cond) in meta_map:
                                meta_map[(pid, cond)]['truth'] = str(item['truth'])
                print(f"  Truth injection complete.")

        # 3. Centroids
        print("  Computing centroids...")
        accumulators = {}
        processed_centroids = 0
        for f in tqdm(files[:1500], desc="Centroids", leave=False):
            try:
                clean_f = f.replace("problem_", "").replace(".npy", "")
                pid = str(int(clean_f.split('_')[0]))
                cond = 'cot' if 'cot' in f else 'direct'
                info = meta_map.get((pid, cond))
                if info and cond == 'cot' and info['correct']:
                    h = np.load(os.path.join(cfg['hidden_dir'], f))
                    # Validate dim
                    if h.shape[-1] != G_U.shape[0]: continue
                    for l in range(len(h)):
                        if l not in accumulators: accumulators[l] = []
                        accumulators[l].append(np.mean(h[l], axis=0))
                    processed_centroids += 1
            except: pass
        centroids = {l: np.mean(v, axis=0) for l,v in accumulators.items()}
        print(f"  Centroids computed from {processed_centroids} correct CoT trajectories.")

        # 4. Vector Map (Pre-calculate specific vectors)
        print("  Mapping token vectors...")
        truth_vec_map = {}
        embed_vec_map = {}
        truth_id_map = {}
        if tokenizer and W_U is not None:
            all_truths = set(info['truth'] for info in meta_map.values())
            for t_text in all_truths:
                try:
                    ids = tokenizer.encode(t_text, add_special_tokens=False)
                    if ids: 
                        t_id = ids[0]
                        truth_id_map[t_text] = t_id
                        truth_vec_map[t_text] = W_U[t_id].copy()
                        embed_vec_map[t_text] = W_E[t_id].copy()
                except: pass
            print(f"  Mapped {len(truth_vec_map)} token vectors.")
        
        # Free heavy matrices before launching pool
        del W_U, W_E
        gc.collect()

        # 5. Parallel Batch Execution
        print(f"  Processing {len(remaining_files)} files on {num_cpus} cores...")
        
        with ProcessPoolExecutor(max_workers=num_cpus, initializer=init_worker, initargs=(G_U,)) as executor:
            task_list = []
            for f in remaining_files:
                clean_f = f.replace("problem_", "").replace(".npy", "")
                try:
                    pid = str(int(clean_f.split('_')[0]))
                except:
                    pid = clean_f.split('_')[0]
                cond = 'cot' if 'cot' in f else 'direct'
                info = meta_map.get((pid, cond), {'truth': '0'})
                t_id = truth_id_map.get(info['truth'], 0)
                t_vec = truth_vec_map.get(info['truth'])
                e_vec = embed_vec_map.get(info['truth'])
                
                task_list.append(executor.submit(run_single_file_task, f, cfg['hidden_dir'], cfg['name'], meta_map, t_id, centroids, t_vec, e_vec))

            # Incremental saving
            for fut in tqdm(as_completed(task_list), total=len(remaining_files), desc="Metrics"):
                res = fut.result()
                if res:
                    df_step = pd.DataFrame(res)
                    # Write header only if file doesn't exist
                    df_step.to_csv(out_csv, mode='a', index=False, header=not os.path.exists(out_csv))
        
        print(f"  Finished processing {cfg['name']}.")
        
        # FINAL CLEANUP for this model
        del G_U, tokenizer, meta_map, truth_vec_map, truth_id_map, embed_vec_map, centroids, accumulators
        gc.collect()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
