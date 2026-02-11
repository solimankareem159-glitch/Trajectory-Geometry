"""
EXP-17A: Data Generation (Exact Replication + Optimization)
===========================================================
Generates 300 arithmetic problems using Qwen2.5-3B-Instruct.
Replicates EXP-09 protocol EXACTLY:
- Loads prompts/questions from EXP-09 dataset.
- Uses DirectML Optimization (Batching).
- Saves full response for manual parsing.

- Conditions: Direct (Greedy, 32 tok) and CoT (Greedy, 128 tok)
- Batch Size: 8
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
import re
import time
from tqdm import tqdm

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_DIR = r"experiments/EXP-17_BaselineReplication_2026-02-11/data"
OUTPUT_FILENAME = "exp17a_dataset.jsonl"
EXP09_DATA_PATH = r"experiments/EXP-09_GeometryCapability_2025-11-22/data/exp9_dataset.jsonl"
BATCH_SIZE = 4

def extract_answer(text):
    numbers = re.findall(r'-?\d+', text)
    if not numbers: return None
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
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left' 

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float16
    )
    model.to(device)
    model.eval()
    
    # Load EXP-09 Dataset (Exact Items)
    print(f"Loading EXP-09 items from {EXP09_DATA_PATH}...")
    if not os.path.exists(EXP09_DATA_PATH):
        print(f"ERROR: EXP-09 dataset not found at {EXP09_DATA_PATH}")
        return
        
    all_problems = []
    with open(EXP09_DATA_PATH, 'r') as f:
        for line in f:
            rec = json.loads(line)
            # We need id, question, truth. 
            # Rec has: id, question, truth, direct:{...}, cot:{...}
            all_problems.append({
                "id": rec['id'],
                "q": rec['question'],
                "a": rec['truth']
            })
    
    print(f"Loaded {len(all_problems)} problems for replication.")
        
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    
    # Resume check
    done_ids = set()
    if os.path.exists(out_path):
        with open(out_path, 'r') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done_ids.add(rec['id'])
                except: pass
    
    print(f"Resuming... {len(done_ids)} samples already done.")
    
    # Filter remaining problems
    todo_problems = [p for p in all_problems if p['id'] not in done_ids]
    
    if not todo_problems:
        print("All samples completed.")
        return

    # Process in batches
    print(f"Processing {len(todo_problems)} samples with batch size {BATCH_SIZE}...")
    
    for i in tqdm(range(0, len(todo_problems), BATCH_SIZE)):
        batch = todo_problems[i : i+BATCH_SIZE]
        
        # Prepare prompts (Exact wording from EXP-09 logic)
        # EXP-09 prompts were:
        # Direct: "Answer the following question directly.\nQ: Calculate {q_str}\nA:"
        # CoT: "Q: Calculate {q_str}\nLet's think step by step before answering."
        
        prompts_d = [f"Answer the following question directly.\nQ: Calculate {p['q']}\nA:" for p in batch]
        prompts_c = [f"Q: Calculate {p['q']}\nLet's think step by step before answering." for p in batch]
        
        # 1. Direct Condition
        inputs_d = tokenizer(prompts_d, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            gen_d = model.generate(
                **inputs_d, 
                max_new_tokens=32, 
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        responses_d = tokenizer.batch_decode(gen_d[:, inputs_d.input_ids.shape[1]:], skip_special_tokens=True)
        
        # 2. CoT Condition
        inputs_c = tokenizer(prompts_c, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            gen_c = model.generate(
                **inputs_c, 
                max_new_tokens=128, 
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        responses_c = tokenizer.batch_decode(gen_c[:, inputs_c.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Save Batch
        for j, p in enumerate(batch):
            resp_d = responses_d[j]
            resp_c = responses_c[j]
            
            ans_d = extract_answer(resp_d)
            ans_c = extract_answer(resp_c)
            
            record = {
                "id": p['id'],
                "question": p['q'],
                "truth": p['a'],
                "direct": {
                    "prompt": prompts_d[j],
                    "response": resp_d,
                    "parsed": ans_d,
                    "correct": (ans_d == p['a'])
                },
                "cot": {
                    "prompt": prompts_c[j],
                    "response": resp_c,
                    "parsed": ans_c,
                    "correct": (ans_c == p['a'])
                }
            }
            
            with open(out_path, 'a') as f:
                f.write(json.dumps(record) + "\n")

    print("\nDone.")

if __name__ == "__main__":
    main()
