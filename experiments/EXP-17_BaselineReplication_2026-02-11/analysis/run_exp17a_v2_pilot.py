"""
EXP-17A Pilot (v2): Prompt Redesign & Difficulty Stratification
===============================================================
Purpose: Test if new direct prompt + higher token limits fix the "Reasoning Persistence" issue.
Sampling: 20 problems total (5 from each difficulty quartile of the original 300).
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

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DATA_DIR = r"experiments/EXP-17_BaselineReplication_2026-02-11/data"
INPUT_DATASET = os.path.join(DATA_DIR, "exp17a_dataset.jsonl")
OUTPUT_FILE = os.path.join(DATA_DIR, "exp17a_v2_pilot.jsonl")
N_SAMPLES_PER_BIN = 5

def get_magnitude(truth):
    try:
        return abs(float(truth))
    except:
        return 0.0

def load_and_stratify_data():
    print(f"Loading {INPUT_DATASET}...")
    data = [json.loads(line) for line in open(INPUT_DATASET)]
    df = pd.DataFrame(data)
    
    # Calculate magnitude
    df['mag'] = df['truth'].apply(get_magnitude)
    
    # Quartiles
    df['bin'] = pd.qcut(df['mag'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    print("\nDifficulty Distribution (Magnitude Quartiles):")
    print(df['bin'].value_counts())
    
    # Sample
    selected = []
    for b in ['Q1', 'Q2', 'Q3', 'Q4']:
        subset = df[df['bin'] == b]
        sample = subset.sample(n=min(len(subset), N_SAMPLES_PER_BIN), random_state=42)
        print(f"  Selected {len(sample)} from {b} (Mean Mag: {sample['mag'].mean():.1f})")
        selected.extend(sample.to_dict('records'))
        
    return selected

def extract_answer_smart(text):
    """Smart answer extraction for pilot analysis."""
    if not text: return None
    # 1. Look for single number on a line
    lines = text.strip().split('\n')
    for line in lines:
        if re.match(r'^-?\d+$', line.strip()):
            return int(line.strip())
            
    # 2. Look for last number
    nums = re.findall(r'-?\d+', text)
    if nums:
        return int(nums[-1])
    return None

def main():
    print(f"PID: {os.getpid()}", flush=True)
    
    # 1. Select Data
    max_retries = 3
    dataset = None
    for _ in range(max_retries):
        try:
            dataset = load_and_stratify_data()
            break
        except Exception as e:
            print(f"Error loading data: {e}")
            time.sleep(1)
            
    if not dataset:
        print("Failed to load dataset.")
        return

    # 2. Load Model
    try:
        import torch_directml
        device = torch_directml.device()
        print(f"Using DirectML Device: {torch_directml.device_name(0)}")
    except ImportError:
        device = torch.device("cpu")
        print("Warning: DirectML not found, using CPU")

    print(f"Loading Model {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float16
    ).to(device)
    model.eval()
    
    results = []
    
    print("\nStarting Inference...")
    for i, rec in enumerate(dataset):
        q_str = rec['question']
        truth = rec['truth']
        mag_bin = rec['bin']
        
        print(f"\nExample {i+1}/20 [{mag_bin}]: {q_str} = {truth}")
        
        # --- Condition A: Direct (New Prompt) ---
        # "Calculate {expr}. Provide ONLY the final numerical answer on a single line. No steps, no explanation, no words — just the number."
        prompt_direct = f"Calculate {q_str}. Provide ONLY the final numerical answer on a single line. No steps, no explanation, no words — just the number."
        
        inputs_d = tokenizer(prompt_direct, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_d = model.generate(
                **inputs_d, 
                max_new_tokens=64, # Increased from 32
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        text_d = tokenizer.decode(gen_d[0][inputs_d.input_ids.shape[1]:], skip_special_tokens=True).strip()
        ans_d = extract_answer_smart(text_d)
        is_correct_d = (ans_d == truth)
        
        print(f"  Direct Input: {prompt_direct}")
        print(f"  Direct Output: {repr(text_d)}")
        print(f"  Parsed: {ans_d} ({'OK' if is_correct_d else 'FAIL'})")
        
        # --- Condition B: CoT (Higher Limit) ---
        prompt_cot = f"Q: Calculate {q_str}\nLet's think step by step before answering."
        inputs_c = tokenizer(prompt_cot, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_c = model.generate(
                **inputs_c, 
                max_new_tokens=256, # Increased from 128
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        text_c = tokenizer.decode(gen_c[0][inputs_c.input_ids.shape[1]:], skip_special_tokens=True).strip()
        ans_c = extract_answer_smart(text_c)
        is_correct_c = (ans_c == truth)
        
        print(f"  CoT Len: {len(text_c)} chars")
        print(f"  CoT Output (last 50): ...{repr(text_c[-50:])}")
        print(f"  Parsed: {ans_c} ({'OK' if is_correct_c else 'FAIL'})")
        
        results.append({
            "original_id": rec['id'],
            "question": q_str,
            "truth": truth,
            "difficulty_bin": str(mag_bin),
            "direct": {
                "prompt": prompt_direct,
                "response": text_d,
                "parsed": ans_d,
                "correct": is_correct_d
            },
            "cot": {
                "prompt": prompt_cot,
                "response": text_c,
                "parsed": ans_c,
                "correct": is_correct_c
            }
        })
        
    # Save
    with open(OUTPUT_FILE, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            
    # Summary Stats
    d_acc = sum(1 for r in results if r['direct']['correct']) / 20
    c_acc = sum(1 for r in results if r['cot']['correct']) / 20
    
    # Check strict compliance (Direct)
    # Count how many responses are just a number (ignoring whitespace)
    strict_compliance = sum(1 for r in results if re.match(r'^-?\d+$', r['direct']['response'].strip()))
    
    print("\n" + "="*40)
    print("PILOT RESULTS Summary")
    print("="*40)
    print(f"Direct Accuracy: {d_acc*100:.1f}%")
    print(f"CoT Accuracy:    {c_acc*100:.1f}%")
    print(f"Direct Strict Format Compliance: {strict_compliance}/20 ({strict_compliance/20*100:.1f}%)")
    print(f"Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
