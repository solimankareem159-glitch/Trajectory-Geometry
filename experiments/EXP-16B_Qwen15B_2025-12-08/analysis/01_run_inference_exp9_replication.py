"""
Experiment 16B: Exact Replication of Experiment 9
===================================================
This script loads the EXACT questions from Experiment 9 and generates responses
using Qwen2.5-1.5B with IDENTICAL generation parameters to Experiment 9.

Goal: Create identical JSONL output format for visual comparison.
"""

import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import re

# --- Model Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- File Paths ---
EXP9_DATASET = r"experiments/Experiment 9/data/exp9_dataset.jsonl"
OUTPUT_DIR = r"experiments/Experiment 16B/data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "exp16b_dataset.jsonl")

# --- Experiment 9 Generation Parameters (EXACT MATCH) ---
DIRECT_MAX_TOKENS = 32
COT_MAX_TOKENS = 128
DO_SAMPLE = False
TEMPERATURE = None  # Greedy decoding

print(f"PID: {os.getpid()}", flush=True)
print("=" * 70)
print("Experiment 16B: Exact Replication of Experiment 9")
print("=" * 70)
print(f"Model: {MODEL_NAME}")
print(f"Device: {DEVICE}")
print(f"Direct Max Tokens: {DIRECT_MAX_TOKENS}")
print(f"CoT Max Tokens: {COT_MAX_TOKENS}")
print(f"Do Sample: {DO_SAMPLE}")
print(f"Temperature: {TEMPERATURE}")
print("=" * 70)

# --- Load Experiment 9 Dataset ---
print(f"\n[1/4] Loading Experiment 9 dataset from {EXP9_DATASET}...")
questions = []
with open(EXP9_DATASET, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        questions.append({
            'id': data['id'],
            'question': data['question'],
            'truth': data['truth']
        })
print(f"Loaded {len(questions)} questions")

# --- Load Model ---
print(f"\n[2/4] Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()
print(f"Model loaded on {DEVICE}")

# --- Helper Function: Parse Answer ---
def parse_answer(text):
    """Extract numerical answer from response text"""
    # Try to find explicit "answer is X" patterns
    answer_patterns = [
        r'(?:answer is|equals?|=)\s*(-?\d+)',
        r'(-?\d+)\s*$',  # Last number in string
        r'=\s*(-?\d+)',
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except:
                pass
    
    # Fallback: extract all numbers and take the last one
    numbers = re.findall(r'-?\d+', text)
    if numbers:
        try:
            return int(numbers[-1])
        except:
            pass
    
    return None

# --- Generate Responses ---
print(f"\n[3/4] Generating responses...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

results = []
for item in tqdm(questions, desc="Processing"):
    q_str = item['question']
    truth = item['truth']
    problem_id = item['id']
    
    # === DIRECT PROMPT (Experiment 9 Exact) ===
    direct_prompt = f"Answer the following question directly.\nQ: Calculate {q_str}\nA:"
    direct_inputs = tokenizer(direct_prompt, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        direct_outputs = model.generate(
            **direct_inputs,
            max_new_tokens=DIRECT_MAX_TOKENS,  # EXACT MATCH
            do_sample=DO_SAMPLE,               # EXACT MATCH
            temperature=TEMPERATURE,           # EXACT MATCH
            pad_token_id=tokenizer.eos_token_id
        )
    
    direct_response = tokenizer.decode(
        direct_outputs[0][direct_inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    direct_parsed = parse_answer(direct_response)
    direct_correct = (direct_parsed == truth)
    
    # === COT PROMPT (Experiment 9 Exact) ===
    cot_prompt = f"Q: Calculate {q_str}\nLet's think step by step before answering."
    cot_inputs = tokenizer(cot_prompt, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        cot_outputs = model.generate(
            **cot_inputs,
            max_new_tokens=COT_MAX_TOKENS,     # EXACT MATCH
            do_sample=DO_SAMPLE,               # EXACT MATCH
            temperature=TEMPERATURE,           # EXACT MATCH
            pad_token_id=tokenizer.eos_token_id
        )
    
    cot_response = tokenizer.decode(
        cot_outputs[0][cot_inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    cot_parsed = parse_answer(cot_response)
    cot_correct = (cot_parsed == truth)
    
    # === Save in EXACT Experiment 9 Format ===
    result = {
        "id": problem_id,
        "question": q_str,
        "truth": truth,
        "direct": {
            "prompt": direct_prompt,
            "response": direct_response,
            "parsed": direct_parsed,
            "correct": direct_correct
        },
        "cot": {
            "prompt": cot_prompt,
            "response": cot_response,
            "parsed": cot_parsed,
            "correct": cot_correct
        }
    }
    results.append(result)

# --- Save Results ---
print(f"\n[4/4] Saving results to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for result in results:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

# --- Print Summary Statistics ---
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

direct_accuracy = sum(1 for r in results if r['direct']['correct']) / len(results) * 100
cot_accuracy = sum(1 for r in results if r['cot']['correct']) / len(results) * 100

print(f"Total Problems: {len(results)}")
print(f"Direct Accuracy: {direct_accuracy:.2f}%")
print(f"CoT Accuracy: {cot_accuracy:.2f}%")
print(f"\nOutput saved to: {OUTPUT_FILE}")
print("=" * 70)
