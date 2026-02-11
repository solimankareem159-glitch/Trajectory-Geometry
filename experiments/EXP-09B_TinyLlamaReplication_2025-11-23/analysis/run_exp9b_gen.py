"""
Experiment 9B: Cross-Model Replication Study
Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0

Replicating Exp 9 methodology exactly:
- Same problem format: (A*B)+C or A*(B+C)
- Same operand ranges: A in [10,50], B in [2,20], C in [10,100]
- Same prompt wording
- Same decoding: greedy (temp=0)
- N=300 samples
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
import random
import re

# --- Configuration ---
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = r"experiments/Experiment 9B/data"
OUTPUT_FILENAME = "exp9b_dataset.jsonl"
N_SAMPLES = 300

def generate_problem():
    """Identical to Exp 9."""
    op = random.choice([(0, " + "), (1, " - ")])
    a = random.randint(10, 50)
    b = random.randint(2, 20)
    c = random.randint(10, 100)
    
    if random.random() < 0.5:
        q_str = f"({a} * {b}) {op[1]} {c}"
        ans = (a * b) + c if op[0] == 0 else (a * b) - c
    else:
        q_str = f"{a} * ({b} {op[1]} {c})"
        ans = a * (b + c) if op[0] == 0 else a * (b - c)
        
    return q_str, ans

def extract_answer(text):
    """Identical to Exp 9: extract last integer from text."""
    numbers = re.findall(r'-?\d+', text)
    if not numbers:
        return None
    return int(numbers[-1])

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"Loading Model {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    
    # Clear existing file
    if os.path.exists(out_path):
        os.remove(out_path)
    
    print(f"Generating {N_SAMPLES} problems...")
    
    n_succ_d = 0
    n_succ_c = 0
    
    for i in range(N_SAMPLES):
        print(f"Processing {i+1}/{N_SAMPLES}...", end='\r')
        q_str, truth = generate_problem()
        
        # --- Condition A: Direct Answer ---
        # EXACT prompt wording from Exp 9
        prompt_direct = f"Answer the following question directly.\nQ: Calculate {q_str}\nA:"
        inputs_d = tokenizer(prompt_direct, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            gen_d = model.generate(
                **inputs_d, 
                max_new_tokens=32, 
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        text_d = tokenizer.decode(gen_d[0][inputs_d.input_ids.shape[1]:], skip_special_tokens=True)
        ans_d = extract_answer(text_d)
        is_correct_d = (ans_d == truth)
        if is_correct_d:
            n_succ_d += 1
        
        # --- Condition B: Chain-of-Thought ---
        # EXACT prompt wording from Exp 9
        prompt_cot = f"Q: Calculate {q_str}\nLet's think step by step before answering."
        inputs_c = tokenizer(prompt_cot, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            gen_c = model.generate(
                **inputs_c, 
                max_new_tokens=128, 
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        text_c = tokenizer.decode(gen_c[0][inputs_c.input_ids.shape[1]:], skip_special_tokens=True)
        ans_c = extract_answer(text_c)
        is_correct_c = (ans_c == truth)
        if is_correct_c:
            n_succ_c += 1
        
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
        
        # Incremental save
        with open(out_path, 'a') as f:
            f.write(json.dumps(record) + "\n")
    
    print(f"\nGeneration Complete!")
    print(f"Direct Accuracy: {n_succ_d}/{N_SAMPLES} ({n_succ_d/N_SAMPLES:.2%})")
    print(f"CoT Accuracy:    {n_succ_c}/{N_SAMPLES} ({n_succ_c/N_SAMPLES:.2%})")

if __name__ == "__main__":
    print(f"PID: {os.getpid()}", flush=True)
    main()
