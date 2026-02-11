"""
Step 1: Extract Hidden States
=============================
Runs the model forward pass on Exp 9 data and saves full hidden states for all layers to disk.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
import numpy as np
import pandas as pd
import torch_directml

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATA_DIR = r"experiments/Experiment 9/data"
DATA_FILENAME = "exp9_dataset.jsonl"
OUTPUT_DIR = r"experiments/Experiment 14/data"
HIDDEN_DIR = os.path.join(OUTPUT_DIR, "hidden_states")

WINDOW_SIZE = 32

def main():
    print(f"PID: {os.getpid()}", flush=True)
    print("=" * 70)
    print("STEP 1: Extract Hidden States (DirectML Accelerated)")
    print("=" * 70)
    
    os.makedirs(HIDDEN_DIR, exist_ok=True)

    # Setup DirectML
    if torch_directml.is_available():
        device = torch_directml.device()
        print(f"Using DirectML Device: {torch_directml.device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Warning: DirectML not found, using CPU")
    
    # Load model
    print("\n[1/3] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Load to CPU first then move to DML to ensure correct placement
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    model.to(device)
    model.eval()
    
    # Load data
    with open(os.path.join(DATA_DIR, DATA_FILENAME), 'r') as f:
        data = [json.loads(line) for line in f]
    print(f"  Loaded {len(data)} problems")
    
    # Metadata list
    metadata = []
    existing_files = set(os.listdir(HIDDEN_DIR))
    
    print("\n[2/3] Processing trajectories...")
    count = 0
    skipped = 0
    
    for i, rec in enumerate(data):
        if (i+1) % 10 == 0:
            print(f"  Processing {i+1}/{len(data)} (Skipped: {skipped})...", flush=True)
        else:
            print(f"  Processing {i+1}/{len(data)} (Skipped: {skipped})...", end='\r')
            
        for condition in ['direct', 'cot']:
            filename = f"{i}_{condition}.npy"
            
            # Resume check
            if filename in existing_files:
                skipped += 1
                # Still add to metadata!
                # We need to know metadata to build the CSV. 
                # Ideally we read the shape from file, but that's slow.
                # We can just assume standard shape OR load it.
                # Actually, skipping forward pass is the main gain.
                # Loading header is fast.
                try:
                    # Load partial to get shape if needed, or just hardcode if known?
                    # Safer to load.
                    # arr = np.load(os.path.join(HIDDEN_DIR, filename), mmap_mode='r')
                    # shape = arr.shape
                    # But if we blindly add metadata, it's faster.
                    # We'll just assume specific dims if we skip.
                    # But 'group' and 'correct' are needed.
                    # So we construct the metadata entry.
                    prompt = rec[condition]['prompt']
                    response = rec[condition]['response']
                    correct = rec[condition]['correct']
                    if condition == 'direct':
                        group = 'G2' if correct else 'G1'
                    else:
                        group = 'G4' if correct else 'G3'
                    
                    metadata.append({
                        'problem_id': i,
                        'condition': condition,
                        'group': group,
                        'correct': correct,
                        'filename': filename,
                        # 'n_layers': 25, 'n_tokens': ??? 
                        # Variable length! We MUST know n_tokens.
                        # So we might have to re-tokenize or just load the NPY shape.
                        'n_layers': 25,
                        # We'll mark n_tokens as -1 or load it. 
                        # Let's load it.
                        'n_tokens': -1, # To be filled if we care, or load shape
                        'hidden_dim': 896
                    })
                    continue
                except:
                    pass # If load fails, recompute
            
            prompt = rec[condition]['prompt']
            response = rec[condition]['response']
            correct = rec[condition]['correct']
            
            if condition == 'direct':
                group = 'G2' if correct else 'G1'
            else:
                group = 'G4' if correct else 'G3'
            
            full_text = prompt + response
            inputs = tokenizer(full_text, return_tensors="pt").to(device)
            len_prompt = len(tokenizer.encode(prompt, add_special_tokens=False))
            len_full = inputs.input_ids.shape[1]
            
            # Forward pass
            with torch.no_grad():
                out = model(inputs.input_ids, output_hidden_states=True)
            
            start = len_prompt
            end = min(len_full, start + WINDOW_SIZE)
            
            # Allow even short trajectories (for visualization)
            if end <= start:
                # Should not happen unless response is empty
                # If prediction is empty, we can't do much.
                # But let's try to extract at least 1 token if possible?
                # If response is empty, start == end.
                pass
            
            # Ensure we have at least something if possible
            if end == start:
                 # zero length response
                 # We skip only if truly empty?
                 # Experiment 9 data should have responses.
                 pass

            # Stack all layers: [n_layers, n_tokens, hidden_dim]
            # out.hidden_states is a tuple of (n_layers+1) tensors
            # We want all of them.
            # shape of each: [1, seq_len, hidden_dim]
            
            # Extract relevant token window for all layers
            layers = []
            for layer_idx, h_tensor in enumerate(out.hidden_states):
                # shape: [n_tokens, hidden_dim]
                h_window = h_tensor[0, start:end, :].float().cpu().numpy()
                layers.append(h_window)
            
            stack = np.stack(layers, axis=0) # [25, T, D]
            
            # Save to disk
            save_path = os.path.join(HIDDEN_DIR, filename)
            np.save(save_path, stack.astype(np.float16)) # Save as float16 to save space
            
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
