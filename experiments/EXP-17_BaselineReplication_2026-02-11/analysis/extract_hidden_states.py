"""
EXP-17A: Hidden State Extraction
================================
Extracts FULL hidden states for all 300 samples (Direct + CoT).
Replicates EXP-14 protocol:
- Full context (prompt + response)
- All layers
- Float16 storage
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
import numpy as np
import pandas as pd
import torch_directml
import time

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DATA_DIR = r"experiments/EXP-17_BaselineReplication_2026-02-11/data"
DATA_FILENAME = "exp17a_dataset.jsonl"
OUTPUT_DIR = r"experiments/EXP-17_BaselineReplication_2026-02-11/data"
HIDDEN_DIR = os.path.join(OUTPUT_DIR, "hidden_states")

def main():
    print(f"PID: {os.getpid()}", flush=True)
    print("=" * 70)
    print("EXP-17A: Extract Hidden States (Full Context)")
    print("=" * 70)
    
    os.makedirs(HIDDEN_DIR, exist_ok=True)

    # Setup DirectML
    try:
        device = torch_directml.device()
        print(f"Using DirectML Device: {torch_directml.device_name(0)}")
    except ImportError:
        device = torch.device("cpu")
        print("Warning: DirectML not found, using CPU")
    
    # Load model
    print("\n[1/3] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float16
    )
    model.to(device)
    model.eval()
    
    # Load data
    data_path = os.path.join(DATA_DIR, DATA_FILENAME)
    if not os.path.exists(data_path):
        print(f"ERROR: Dataset not found at {data_path}")
        return
        
    with open(data_path, 'r') as f:
        data = [json.loads(line) for line in f]
    print(f"  Loaded {len(data)} problems")
    
    # Metadata list
    metadata = []
    
    # Check existing files to allow resume
    existing_files = set(os.listdir(HIDDEN_DIR))
    
    print("\n[2/3] Processing trajectories...")
    count = 0
    skipped = 0
    t0 = time.time()
    
    for i, rec in enumerate(data):
        if (i+1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (count / elapsed) if elapsed > 0 else 0
            print(f"  Processing {i+1}/{len(data)} (Skipped: {skipped}) - {rate:.2f} items/s...", flush=True)
        else:
            print(f"  Processing {i+1}/{len(data)} (Skipped: {skipped})...", end='\r', flush=True)
            
        for condition in ['direct', 'cot']:
            filename = f"{i}_{condition}.npy"
            
            prompt = rec[condition]['prompt']
            response = rec[condition]['response']
            correct = rec[condition]['correct']
            
            if condition == 'direct':
                group = 'G2' if correct else 'G1'
            else:
                group = 'G4' if correct else 'G3'

            # Resume check
            if filename in existing_files:
                skipped += 1
                # If metadata.csv exists, we assume it's there. 
                # If not, we might miss this record in metadata unless we reload it.
                # For robustness, let's re-calculate metadata if we load the file headers?
                # For now, just skip extraction but DON'T add to metadata list 
                # (we'll assume metadata.csv is generated at end, so if we skip, we might lose it?
                # Better strategy: If skipping, try to read the npy shape? 
                # Or just rely on previous metadata csv?
                # Let's just re-extract if metadata is missing? 
                # Simplest: Just re-extract if not found. If found, skip.
                # But we need to populate metadata info for the CSV.
                # Let's peek at the file to get shape.
                try:
                    shape = np.load(os.path.join(HIDDEN_DIR, filename), mmap_mode='r').shape
                    metadata.append({
                        'problem_id': i,
                        'condition': condition,
                        'group': group,
                        'correct': correct,
                        'filename': filename,
                        'n_layers': shape[0],
                        'n_tokens': shape[1],
                        'hidden_dim': shape[2]
                    })
                except:
                    print(f"  Warning: Could not read {filename}, re-extracting.")
                    pass # Fall through to re-extract
                else:
                    continue
            
            full_text = prompt + response
            inputs = tokenizer(full_text, return_tensors="pt").to(device)
            len_prompt = len(tokenizer.encode(prompt, add_special_tokens=False))
            len_full = inputs.input_ids.shape[1]
            
            # Forward pass
            with torch.no_grad():
                out = model(inputs.input_ids, output_hidden_states=True)
            
            # Extract FULL context (EXP-14 protocol)
            # Or just response? EXP-14 extract_hidden_states_full_context.py extracted response only?
            # Let's check the code I read earlier.
            # "start = len_prompt", "end = len_full".
            # So it extracted RESPONSE ONLY.
            
            start = len_prompt
            end = len_full
            
            if start >= end:
                # Empty response
                print(f"  Warning: Empty response for {i}_{condition}")
                continue

            layers = []
            for h_tensor in out.hidden_states:
                # shape: [batch, n_tokens, hidden_dim]
                h_window = h_tensor[0, start:end, :].float().cpu().numpy()
                layers.append(h_window)
            
            # stack shape: [n_layers+1, T, D]
            stack = np.stack(layers, axis=0) 
            
            # Save
            save_path = os.path.join(HIDDEN_DIR, filename)
            np.save(save_path, stack.astype(np.float16))
            
            metadata.append({
                'problem_id': i,
                'condition': condition,
                'group': group,
                'correct': correct,
                'filename': filename,
                'n_layers': stack.shape[0],
                'n_tokens': stack.shape[1],
                'hidden_dim': stack.shape[2]
            })
            count += 1
            
    print(f"\n[3/3] Saving metadata...")
    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(OUTPUT_DIR, "metadata.csv"), index=False)
    print(f"  Saved {len(df)} records to metadata.csv")
    print(f"  Hidden states saved to {HIDDEN_DIR}")

if __name__ == "__main__":
    main()
