import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
import random
import re
import numpy as np

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
OUTPUT_DIR = r"experiments/Experiment 9/data"
OUTPUT_FILENAME = "exp9_dataset.jsonl"
N_SAMPLES = 50 

def generate_problem():
    # Format: (A * B) + C or similar
    # A, B in 10-50 to ensure 2-3 digit result.
    op = random.choice([(0, " + "), (1, " - ")])
    a = random.randint(10, 50)
    b = random.randint(2, 20)
    c = random.randint(10, 100)
    
    # Mix structure
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
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"Loading Model {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    
    records = []
    
    print(f"Generating {N_SAMPLES} problems...")
    
    count = 0
    # We want ~300. 
    # Let's actually run the loop until we hit 300.
    
    # To ensure balance, we might need more samples if performance is too high/low,
    # but the prompt says "Balanced difficulty (some failures expected)".
    # Qwen 0.5B usually struggles with arithmetic, so obtaining failures is easy.
    # Obtaining successes might be harder.
    # I'll stick to 10-50 range for operands.
    
        
    # Incremental consistency check
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    # Appending to existing file
    # if os.path.exists(out_path):
    #    os.remove(out_path)
        
    for i in range(N_SAMPLES):
        print(f"Processing {i+1}/{N_SAMPLES}...", end='\r')
        q_str, truth = generate_problem()
        
        # 1. Condition A: Direct
        prompt_direct = f"Answer the following question directly.\nQ: Calculate {q_str}\nA:"
        inputs_d = tokenizer(prompt_direct, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen_d = model.generate(**inputs_d, max_new_tokens=32, do_sample=False, temperature=None) # Greedy
        text_d = tokenizer.decode(gen_d[0][inputs_d.input_ids.shape[1]:], skip_special_tokens=True)
        ans_d = extract_answer(text_d)
        is_correct_d = (ans_d == truth)
        
        # 2. Condition B: CoT
        prompt_cot = f"Q: Calculate {q_str}\nLet's think step by step before answering."
        inputs_c = tokenizer(prompt_cot, return_tensors="pt").to(model.device)
        with torch.no_grad():
            # Allow longer generation for CoT
            gen_c = model.generate(**inputs_c, max_new_tokens=128, do_sample=False, temperature=None)
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
            
    # Stats
    n_succ_d = sum(1 for r in records if r["direct"]["correct"])
    n_succ_c = sum(1 for r in records if r["cot"]["correct"])
    print(f"Direct Accuracy: {n_succ_d}/{N_SAMPLES} ({n_succ_d/N_SAMPLES:.2f})")
    print(f"CoT Accuracy:    {n_succ_c}/{N_SAMPLES} ({n_succ_c/N_SAMPLES:.2f})")

if __name__ == "__main__":
    import os
    print(f"PID: {os.getpid()}", flush=True)
    main()
