"""
EXP-17A: Data Generation
========================
Generates 300 arithmetic problems using Qwen2.5-3B-Instruct.
Replicates EXP-09 protocol:
- Operands: A[10-50], B[2-20], C[10-100]
- Structure: (A*B)±C and A*(B±C)
- Conditions: Direct (Greedy, 32 tok) and CoT (Greedy, 128 tok)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
import random
import re
import numpy as np
import time

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_DIR = r"experiments/EXP-17_BaselineReplication_2026-02-11/data"
OUTPUT_FILENAME = "exp17a_dataset.jsonl"
N_SAMPLES = 300

def generate_problem():
    # Format: (A * B) + C or similar
    # A, B in 10-50 to ensure 2-3 digit result.
    op = random.choice([(0, " + "), (1, " - ")])
    a = random.randint(10, 50)
    b = random.randint(2, 20)
    c = random.randint(10, 100)
    
    # Mix structure (50/50 balance)
    if random.random() < 0.5:
        q_str = f"({a} * {b}) {op[1]} {c}"
        ans = (a * b) + c if op[0] == 0 else (a * b) - c
    else:
        q_str = f"{a} * ({b} {op[1]} {c})"
        ans = a * (b + c) if op[0] == 0 else a * (b - c)
        
    return q_str, ans

def extract_answer(text):
    # Look for last number in text
    # This is a bit heuristic, but usually works for "The answer is X" or just "X"
    # Or CoT "Therefore, the result is X."
    numbers = re.findall(r'-?\d+', text)
    if not numbers:
        return None
    return int(numbers[-1])

def main():
    print(f"PID: {os.getpid()}", flush=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Setup DirectML
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
    )
    model.to(device)
    model.eval()
    
    records = []
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    
    # Resume check
    start_idx = 0
    if os.path.exists(out_path):
        with open(out_path, 'r') as f:
            existing = [json.loads(line) for line in f]
        start_idx = len(existing)
        records = existing
        print(f"Resuming from {start_idx} samples...")
    
    print(f"Generating problems {start_idx+1} to {N_SAMPLES}...")
    
    # Set seed for reproducibility relative to start_idx 
    # (though random generator state is not saved, so this is approximate)
    random.seed(42 + start_idx) 
    
    for i in range(start_idx, N_SAMPLES):
        if (i+1) % 10 == 0:
            print(f"Processing {i+1}/{N_SAMPLES}...", flush=True)
        else:
            print(f"Processing {i+1}/{N_SAMPLES}...", end='\r', flush=True)
            
        q_str, truth = generate_problem()
        
        # 1. Condition A: Direct
        prompt_direct = f"Answer the following question directly.\nQ: Calculate {q_str}\nA:"
        inputs_d = tokenizer(prompt_direct, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_d = model.generate(
                **inputs_d, 
                max_new_tokens=32, 
                do_sample=False
            )
        text_d = tokenizer.decode(gen_d[0][inputs_d.input_ids.shape[1]:], skip_special_tokens=True)
        ans_d = extract_answer(text_d)
        is_correct_d = (ans_d == truth)
        
        # 2. Condition B: CoT
        prompt_cot = f"Q: Calculate {q_str}\nLet's think step by step before answering."
        inputs_c = tokenizer(prompt_cot, return_tensors="pt").to(device)
        with torch.no_grad():
            # Allow longer generation for CoT
            gen_c = model.generate(
                **inputs_c, 
                max_new_tokens=128, 
                do_sample=False
            )
        text_c = tokenizer.decode(gen_c[0][inputs_c.input_ids.shape[1]:], skip_special_tokens=True)
        ans_c = extract_answer(text_c)
        is_correct_c = (ans_c == truth)
        
        record = {
            "id": i,
            "question": q_str,
            "truth": truth,
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
        }
        records.append(record)
        
        # Incremental Save
        with open(out_path, 'a') as f:
            f.write(json.dumps(record) + "\n")
            
    # Stats
    n_succ_d = sum(1 for r in records if r["direct"]["correct"])
    n_succ_c = sum(1 for r in records if r["cot"]["correct"])
    print(f"\nCompleted {N_SAMPLES} samples.")
    print(f"Direct Accuracy: {n_succ_d}/{N_SAMPLES} ({n_succ_d/N_SAMPLES:.2f})")
    print(f"CoT Accuracy:    {n_succ_c}/{N_SAMPLES} ({n_succ_c/N_SAMPLES:.2f})")

if __name__ == "__main__":
    main()
