"""
EXP-17B: Multi-Mode Prompting Data Generation
=============================================
Generates responses for 300 arithmetic problems using Qwen2.5-3B-Instruct across 8 distinct prompting modes.
Reuses exact questions from EXP-09 (replication set).
Uses DirectML Batch Optimization.

Modes:
1. Direct Retrieval (Baseline)
2. Standard CoT (Baseline)
3. Fixed-Step CoT ("Think in exactly 3 steps")
4. Few-Shot CoT (3 examples)
5. Constraint-Based ("Do not use the number 5 in intermediate steps") - tricky for arithmetic
   Wait, better constraint for arithmetic: "Show your work but do not use the word 'therefore'"?
   Or "Use a specific format".
   Let's use the plan definitions:
   - Constraint: "Answer without using the word 'carry' or 'borrow'." (or similar)
   Actually, let's look at implementation_plan.md for definitions.
6. Verification ("Double check your answer")
7. Creative Reframe ("Explain it to a 5 year old")
8. Analogy ("Think of this as...") - Might be weird for arithmetic.
   Maybe "System 2 vs System 1"?
   Let's stick to the plan.

If plan is vague, I will define robust ones here.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
import re
from tqdm import tqdm

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_DIR = r"experiments/EXP-17_BaselineReplication_2026-02-11/data"
OUTPUT_FILENAME = "exp17b_multimode_dataset.jsonl"
EXP09_DATA_PATH = r"experiments/EXP-09_GeometryCapability_2025-11-22/data/exp9_dataset.jsonl"
BATCH_SIZE = 8

# --- Prompt Templates ---
# {q} is the question, e.g. "(38 * 9) + 46"

TEMPLATES = {
    "direct": "Answer the following question directly.\nQ: Calculate {q}\nA:",
    "cot": "Q: Calculate {q}\nLet's think step by step before answering.",
    
    "fixed_step": "Q: Calculate {q}\nThink in exactly 3 steps. Step 1: ...", 
    # Attempt to force 3 steps. Model might ignore, but prompt is fixed.
    
    "few_shot": """Q: Calculate (10 * 5) + 2
A: 10 * 5 = 50. 50 + 2 = 52. The answer is 52.

Q: Calculate 2 * (3 + 4)
A: 3 + 4 = 7. 2 * 7 = 14. The answer is 14.

Q: Calculate (100 - 50) + 10
A: 100 - 50 = 50. 50 + 10 = 60. The answer is 60.

Q: Calculate {q}
A:""",

    "constraint": "Q: Calculate {q}\nShow your work, but you are forbidden from using the word 'then' or 'so'.",
    
    "verification": "Q: Calculate {q}\nFirst calculate the answer, then verify it by reverse calculation.",
    
    "creative": "Q: Calculate {q}\nExplain the solution as if you are teaching a beginner student.",
    
    "analogy": "Q: Calculate {q}\nThink of this calculation as a story." 
    # This might produce long text.
}

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
    
    # Load Questions
    print(f"Loading questions from {EXP09_DATA_PATH}...")
    try:
        all_problems = []
        with open(EXP09_DATA_PATH, 'r') as f:
            for line in f:
                rec = json.loads(line)
                all_problems.append({"id": rec['id'], "q": rec['question'], "a": rec['truth']})
    except Exception as e:
        print(f"Error loading data: {e}")
        return
        
    print(f"Loaded {len(all_problems)} problems.")
    
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    
    # Resume check
    # We need to know which (id, mode) pairs are done.
    done_set = set()
    if os.path.exists(out_path):
        with open(out_path, 'r') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    # Support resuming by storing (id, mode)
                    # The file structure is: {id, question, truth, mode, prompt, response...}
                    # Or flat? 
                    # Let's save one record per (problem, mode).
                    done_set.add((rec['id'], rec['mode']))
                except: pass
                
    print(f"Resuming... {len(done_set)} records already done.")
    
    # Generate tasks
    tasks = []
    modes = list(TEMPLATES.keys())
    
    for p in all_problems:
        for mode in modes:
            if (p['id'], mode) not in done_set:
                tasks.append({
                    "id": p['id'],
                    "q": p['q'],
                    "a": p['a'],
                    "mode": mode,
                    "prompt_text": TEMPLATES[mode].format(q=p['q'])
                })
                
    if not tasks:
        print("All tasks completed.")
        return

    print(f"Processing {len(tasks)} tasks with batch size {BATCH_SIZE}...")
    
    # Sort tasks by mode to minimize cache thrashing? No, independent samples.
    # Group by length maybe? sorting by prompt length improves batch efficiency.
    tasks.sort(key=lambda x: len(x['prompt_text']))
    
    for i in tqdm(range(0, len(tasks), BATCH_SIZE)):
        batch = tasks[i : i+BATCH_SIZE]
        prompts = [t['prompt_text'] for t in batch]
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            gen = model.generate(
                **inputs, 
                max_new_tokens=128,  # Uniform 128 for all modes (Direct gets less usually but 128 is safe caps)
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            
        responses = tokenizer.batch_decode(gen[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        for j, task in enumerate(batch):
            resp = responses[j]
            ans = extract_answer(resp)
            correct = (ans == task['a'])
            
            record = {
                "id": task['id'],
                "question": task['q'],
                "truth": task['a'],
                "mode": task['mode'],
                "prompt": task['prompt_text'],
                "response": resp,
                "parsed": ans,
                "correct": correct
            }
            
            with open(out_path, 'a') as f:
                f.write(json.dumps(record) + "\n")

    print("\nDone.")

if __name__ == "__main__":
    main()
