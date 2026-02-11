"""
EXP-17A Production (v2): Full Run (300 Samples)
===============================================
Purpose: Generate data and extract hidden states for EXP-17 Re-run.
- Sampling: 300 problems (75 per difficulty quartile).
- Prompt: "Provide ONLY the final numerical answer..."
- Features: DirectML support, Hidden State Extraction (Response Only).
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
import random
import re
import numpy as np
import pandas as pd
import time
import torch_directml

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DATA_DIR = r"experiments/EXP-17_BaselineReplication_2026-02-11/data"
INPUT_DATASET = os.path.join(DATA_DIR, "exp17a_dataset.jsonl")
OUTPUT_DIR = r"experiments/EXP-17_BaselineReplication_2026-02-11/data"
HIDDEN_DIR = os.path.join(OUTPUT_DIR, "hidden_states_v2")
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata_v2.csv")

N_SAMPLES_PER_BIN = 75 # Total 300

def get_magnitude(truth):
    try:
        return abs(float(truth))
    except:
        return 0.0

def load_and_stratify_data():
    print(f"Loading {INPUT_DATASET}...", flush=True)
    data = [json.loads(line) for line in open(INPUT_DATASET)]
    df = pd.DataFrame(data)
    
    # Calculate magnitude
    df['mag'] = df['truth'].apply(get_magnitude)
    
    # Quartiles
    df['bin'] = pd.qcut(df['mag'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    print("\nDifficulty Distribution (Magnitude Quartiles):", flush=True)
    print(df['bin'].value_counts(), flush=True)
    
    # Sample
    selected = []
    print("\nSampling...", flush=True)
    for b in ['Q1', 'Q2', 'Q3', 'Q4']:
        subset = df[df['bin'] == b]
        sample = subset.sample(n=min(len(subset), N_SAMPLES_PER_BIN), random_state=42)
        print(f"  Selected {len(sample)} from {b} (Mean Mag: {sample['mag'].mean():.1f})", flush=True)
        selected.extend(sample.to_dict('records'))
        
    return selected

def extract_answer_smart(text):
    if not text: return None
    lines = text.strip().split('\n')
    for line in lines:
        if re.match(r'^-?\d+$', line.strip()):
            return int(line.strip())
    nums = re.findall(r'-?\d+', text)
    if nums:
        return int(nums[-1])
    return None

def main():
    print(f"PID: {os.getpid()}", flush=True)
    
    os.makedirs(HIDDEN_DIR, exist_ok=True)
    
    # 1. Load Data
    dataset = load_and_stratify_data()
    print(f"Total Samples: {len(dataset)}", flush=True)

    # 2. Load Model
    try:
        device = torch_directml.device()
        print(f"Using DirectML Device: {torch_directml.device_name(0)}", flush=True)
    except ImportError:
        device = torch.device("cpu")
        print("Warning: DirectML not found, using CPU", flush=True)

    print(f"Loading Model {MODEL_NAME}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float16
    ).to(device)
    model.eval()
    
    metadata = []
    
    print("\nStarting Production Run...", flush=True)
    t0 = time.time()
    
    for i, rec in enumerate(dataset):
        q_str = rec['question']
        truth = rec['truth']
        
        if (i+1) % 5 == 0:
            elapsed = time.time() - t0
            rate = (i+1) / elapsed
            print(f"Processing {i+1}/{len(dataset)} ({rate:.2f} items/s)...", flush=True)
        
        # --- Prepare Prompts ---
        prompts = {
            'direct': f"Calculate {q_str}. Provide ONLY the final numerical answer on a single line. No steps, no explanation, no words — just the number.",
            'cot': f"Q: Calculate {q_str}\nLet's think step by step before answering."
        }
        
        token_limits = {
            'direct': 64,
            'cot': 256
        }
        
        for condition, prompt in prompts.items():
            # 1. Encode
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            len_prompt = inputs.input_ids.shape[1]
            
            # 2. Generate
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=token_limits[condition],
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    output_hidden_states=True,
                    return_dict_in_generate=True
                )
            
            # 3. Decode & Parse
            # out.sequences contains [prompt + generated]
            full_seq = out.sequences[0]
            generated_seq = full_seq[len_prompt:]
            
            text = tokenizer.decode(generated_seq, skip_special_tokens=True).strip()
            ans = extract_answer_smart(text)
            is_correct = (ans == truth)
            
            # 4. Extract Hidden States (Response Only)
            # out.hidden_states is a tuple (one per step)
            # each element is a tuple (one per layer)
            # values are [batch, 1, hidden_dim] (since it's generation step-by-step)
            # BUT model.generate returns hidden_states differently depending on config?
            # Wait, `output_hidden_states=True` in generate returns:
            # GenerateOutput.hidden_states: tuple (steps) of tuple (layers) of tensor [batch, 1, dim]
            # EXCEPT for the prompt pass?
            
            # Actually, standard `generate` creates hidden_states for generated tokens.
            # But we want the hidden states for the WHOLE sequence (or at least the response)?
            # The baseline `extract_hidden_states.py` did a FORWARD PASS on the full text.
            # This is safer to ensure we get the continuous trajectory.
            
            # So: Generate first, get text. Then Forward Pass to get hidden states.
            # This is 2x compute but reliable.
            
            full_text = prompt + text # Reconstruct full text? 
            # Or just use `tokenizer.decode(full_seq)`?
            # Better to use `full_seq` directly if possible, but we need to re-run forward pass.
            
            # Re-running forward pass
            inputs_full = tokenizer(tokenizer.decode(full_seq, skip_special_tokens=True), return_tensors="pt").to(device)
            # Note: skip_special_tokens might lose some layout, but prompt+text is safer?
            # Actually, `out.sequences` has the token IDs. Use them!
            
            with torch.no_grad():
                out_full = model(full_seq.unsqueeze(0), output_hidden_states=True)
            
            # Extract Response Portion
            # prompt length might have changed due to tokenization differences? 
            # Ideally use the same prompt tokens.
            # `full_seq` starts with `inputs.input_ids`.
            # So response starts at `len_prompt`.
            
            start_idx = len_prompt
            end_idx = full_seq.shape[0] # End of sequence
            
            layers = []
            for h_tensor in out_full.hidden_states:
                # shape: [batch, seq_len, dim]
                # Extract [0, start:end, :]
                h_window = h_tensor[0, start_idx:end_idx, :].float().cpu().numpy()
                layers.append(h_window)
            
            stack = np.stack(layers, axis=0) # [n_layers, n_tokens, dim]
            
            # 5. Save
            # Filename: {original_index}_{condition}.npy ??
            # But `rec` comes from stratification, so original index is lost unless we keep it.
            # The pilot used `enumerate` index. But for 300 samples, we want consistency?
            # Let's use `rec['id']` if available, or just a new running index `i`.
            # We'll use the running index `i` (0..299).
            # filename = f"{i}_{condition}.npy"
            
            filename = f"{i}_{condition}.npy"
            save_path = os.path.join(HIDDEN_DIR, filename)
            np.save(save_path, stack.astype(np.float16))
            
            # 6. Metadata
            # Determine group
            if condition == 'direct':
                group = 'G2' if is_correct else 'G1'
            else:
                group = 'G4' if is_correct else 'G3'
                
            metadata.append({
                'problem_id': i,
                'original_id': rec.get('id', -1),
                'question': q_str,
                'truth': truth,
                'condition': condition,
                'group': group,
                'response': text,
                'parsed': ans,
                'correct': is_correct,
                'filename': filename,
                'n_layers': stack.shape[0],
                'n_tokens': stack.shape[1]
            })

    # Save Metadata
    print(f"\nSaving metadata to {METADATA_FILE}...", flush=True)
    df_meta = pd.DataFrame(metadata)
    df_meta.to_csv(METADATA_FILE, index=False)
    print("Done!", flush=True)

if __name__ == "__main__":
    main()
