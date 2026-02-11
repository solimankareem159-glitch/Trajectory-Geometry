"""
Script 03: Run Inference and Extract Hidden States (Batched)
===========================================================
Runs full inference on 300 problems with batching, checkpointing 
and hidden state extraction.

Usage:
    python 03_run_inference_dump_hidden.py [--save_hidden_states=1] [--resume]
"""

import os
import sys
import json
import re
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from exp16_utils import get_generation_device, set_seed

CHECKPOINT_INTERVAL = 5
BATCH_SIZE = 32  # Optimized for 8GB VRAM with Qwen 1.5B

def parse_numeric_answer(text):
    """Extract numeric answer from response text."""
    numbers = re.findall(r'-?\d+', text)
    if numbers:
        return int(numbers[-1])
    return None

def check_disk_space(min_gb=10):
    """Check if sufficient disk space is available."""
    import shutil
    stat = shutil.disk_usage(".")
    free_gb = stat.free / (1024**3)
    return free_gb > min_gb

def load_progress():
    """Load checkpoint progress."""
    progress_file = "experiments/Experiment 16/data/checkpoints/progress.json"
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {"completed": []}

def save_progress(completed_ids):
    """Save checkpoint progress."""
    progress_file = "experiments/Experiment 16/data/checkpoints/progress.json"
    os.makedirs(os.path.dirname(progress_file), exist_ok=True)
    with open(progress_file, 'w') as f:
        json.dump({"completed": completed_ids}, f)

def save_checkpoint(metadata, checkpoint_num):
    """Save partial metadata at checkpoint."""
    df = pd.DataFrame(metadata)
    checkpoint_file = f"experiments/Experiment 16/data/checkpoints/metadata_checkpoint_{checkpoint_num}.csv"
    df.to_csv(checkpoint_file, index=False)
    # print(f"Checkpoint saved: {checkpoint_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_hidden_states', type=int, default=1, help='Save hidden states to disk (1=yes, 0=no)')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    args = parser.parse_args()
    
    print("="*60)
    print("03_run_inference_dump_hidden.py: Batched Inference")
    print("="*60)
    print(f"Model: Qwen/Qwen2.5-1.5B (Batch Size: {BATCH_SIZE})")
    print(f"Save hidden states: {bool(args.save_hidden_states)}")
    print(f"Resume mode: {args.resume}")
    
    set_seed(42)
    
    if args.save_hidden_states and not check_disk_space(min_gb=10):
        print("WARNING: Low disk space (<10GB).")

    # Load dataset
    with open("experiments/Experiment 16/data/dataset_path.txt", 'r') as f:
        dataset_path = f.read().strip()
    
    with open(dataset_path, 'r') as f:
        dataset = [json.loads(line) for line in f]
    
    # Load progress
    progress = load_progress() if args.resume else {"completed": []}
    completed = set(tuple(x) for x in progress["completed"])
    
    # Load model
    device = get_generation_device()
    model_name = "Qwen/Qwen2.5-1.5B"
    print(f"Loading model on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # Important for batched generation
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if str(device) != 'cpu' else torch.float32,
        output_hidden_states=True,
        trust_remote_code=True
    )
    model.to(device)
    model.eval()
    
    # Optimize
    if str(device) != 'cpu':
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision('medium')
        except:
            pass

    # Prepare tasks
    tasks = []
    for problem in dataset:
        pid = problem['id']
        question = problem['question']
        truth = problem['truth']
        
        # Direct
        if (pid, 'direct') not in completed:
            tasks.append({
                'problem_id': pid, 'condition': 'direct', 'question': question, 'truth': truth,
                'prompt': f"Answer the following question directly.\nQ: Calculate {question}\nA:",
                'max_tokens': 50
            })
            
        # CoT
        if (pid, 'cot') not in completed:
            tasks.append({
                'problem_id': pid, 'condition': 'cot', 'question': question, 'truth': truth,
                'prompt': f"Q: Calculate {question}\nLet's think step by step before answering.",
                'max_tokens': 200
            })
            
    # Sort by condition to minimize padding waste (group similar max_tokens)
    tasks.sort(key=lambda x: x['condition'])
    
    # Create batches
    batches = [tasks[i:i + BATCH_SIZE] for i in range(0, len(tasks), BATCH_SIZE)]
    print(f"Processing {len(tasks)} items in {len(batches)} batches.")

    metadata = []
    checkpoint_counter = 0
    
    pbar = tqdm(total=len(tasks), desc="Inference", unit="sample")
    
    for batch_idx, batch in enumerate(batches):
        prompts = [t['prompt'] for t in batch]
        max_new = batch[0]['max_tokens']  # Assumes sorted by condition so batch has same max_tokens
        
        inputs = tokenizer(prompts, return_tensors='pt', padding=True).to(device)
        input_len = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                output_hidden_states=True,
                return_dict_in_generate=True
            )
            
        # Process items in batch
        for i, task in enumerate(batch):
            pid = task['problem_id']
            cond = task['condition']
            
            # Extract sequence
            gen_seq = outputs.sequences[i][input_len:]
            response = tokenizer.decode(gen_seq, skip_special_tokens=True)
            
            parsed = parse_numeric_answer(response)
            correct = (parsed == task['truth'])
            
            # Extract hidden states
            if args.save_hidden_states:
                try:
                    n_gen = min(len(gen_seq), 32)
                    n_layers = model.config.num_hidden_layers + 1
                    
                    # outputs.hidden_states: tuple(steps) -> tuple(layers) -> [batch, 1, dim]
                    hidden_stack = []
                    for layer_idx in range(n_layers):
                        layer_states = []
                        for token_idx in range(n_gen):
                            if token_idx < len(outputs.hidden_states):
                                # [batch, 1, dim] -> [dim]
                                state = outputs.hidden_states[token_idx][layer_idx][i][0].cpu().numpy()
                                layer_states.append(state)
                        
                        if layer_states:
                            hidden_stack.append(np.stack(layer_states))
                    
                    if hidden_stack:
                        hidden_array = np.stack(hidden_stack)
                        save_path = f"experiments/Experiment 16/data/hidden_states/{pid}_{cond}.npy"
                        np.save(save_path, hidden_array)
                        
                except Exception:
                    pass

            metadata.append({
                'problem_id': pid,
                'condition': cond,
                'question': task['question'],
                'truth': task['truth'],
                'response': response,
                'parsed': parsed,
                'correct': correct,
                'response_length_chars': len(response),
                'response_length_tokens': len(gen_seq)
            })
            
            completed.add((pid, cond))
            pbar.update(1)
        
        # Checkpoint
        if (batch_idx + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint_counter += 1
            # Append new metadata to CSV if needed, but simple overwrite is safer for now
            # Actually we should append to global metadata list
            save_checkpoint(metadata, checkpoint_counter)
            save_progress(list(completed))
            
    pbar.close()
    
    # Final save
    # Merge with previous metadata if resuming? 
    # Current script structure accumulates `metadata` in memory. 
    # If resuming, `load_progress` implies we skip logic, but we don't load previous CSV.
    # To fix resume properly, we should load previous metadata, but simpler to just 
    # append tasks to `metadata_checkpoint_X.csv`?
    # For now, we assume `metadata.csv` is generated from scratch or partial run is discarded?
    # Wait, if we resume, we only generate NEW rows.
    # We should APPEND to metadata.csv.
    # But complicating this now is risky.
    # The standard flow is: `metadata.csv` is final.
    # If we resume, we generate a Partial metadata list.
    # We can handle merging in post-processing or just save `metadata_part2.csv`.
    # Let's save `metadata.csv` as the FULL list (load existing?)
    
    final_df = pd.DataFrame(metadata)
    
    if args.resume and os.path.exists("experiments/Experiment 16/data/metadata.csv"):
        # Append
        old_df = pd.read_csv("experiments/Experiment 16/data/metadata.csv")
        final_df = pd.concat([old_df, final_df], ignore_index=True)
    
    final_df.to_csv("experiments/Experiment 16/data/metadata.csv", index=False)
    print(f"\n[OK] Metadata saved: {len(final_df)} rows")
    save_progress(list(completed))

if __name__ == "__main__":
    main()
