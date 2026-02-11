"""
Step 1B: Extract Hidden States (Full Context)
=============================================
Runs the model forward pass on Exp 9 data and saves FULL hidden states for all layers to disk.
Unlike the original EXP-14 extraction, this does NOT truncate to 32 tokens.
It captures the entire generated response.
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
DATA_DIR = r"experiments/EXP-09_GeometryCapability_2025-11-22/data"
DATA_FILENAME = "exp9_dataset.jsonl"
OUTPUT_DIR = r"experiments/Experiment 14/data"
HIDDEN_DIR = os.path.join(OUTPUT_DIR, "hidden_states_full") # NEW DIRECTORY

# WINDOW_SIZE = 32 # REMOVED: We want full context

def main():
    print(f"PID: {os.getpid()}", flush=True)
    print("=" * 70)
    print("STEP 1B: Extract Hidden States - FULL CONTEXT (DirectML Accelerated)")
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
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    model.to(device)
    model.eval()
    
    # Load data
    with open(os.path.join(DATA_DIR, DATA_FILENAME), 'r') as f:
        data = [json.loads(line) for line in f]
    print(f"  Loaded {len(data)} problems")
    
    # Metadata list
    metadata = []
    # Check existing files to allow resume
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
                # We still need metadata
                # Since we are lazy, we will just count skipped and append metadata
                # Note: We won't know exact n_tokens without loading, but that's fine for now.
                # We will mark n_tokens as -1.
                metadata.append({
                    'problem_id': i,
                    'condition': condition,
                    'group': group,
                    'correct': correct,
                    'filename': filename,
                    'n_layers': 25,
                    'n_tokens': -1, # Unknown when skipping
                    'hidden_dim': 896
                })
                continue
            
            full_text = prompt + response
            inputs = tokenizer(full_text, return_tensors="pt").to(device)
            len_prompt = len(tokenizer.encode(prompt, add_special_tokens=False))
            len_full = inputs.input_ids.shape[1]
            
            # Forward pass
            with torch.no_grad():
                out = model(inputs.input_ids, output_hidden_states=True)
            
            start = len_prompt
            end = len_full # NO TRUNCATION
            
            # Ensure valid range
            if start >= end:
                # Edge case: empty response?
                # We'll just save an empty array or skip?
                # Better to save 0-length to keep file consistency if possible, or 1 frame?
                # Let's skip saving if truly empty, but add metadata?
                pass

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
                'n_tokens': stack.shape[1], # FULL LENGTH
                'hidden_dim': stack.shape[2]
            })
            count += 1
            
    print(f"\n[3/3] Saving metadata...")
    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(OUTPUT_DIR, "metadata_full.csv"), index=False)
    print(f"  Saved {len(df)} records to metadata_full.csv")
    print(f"  Hidden states saved to {HIDDEN_DIR}")

if __name__ == "__main__":
    main()
