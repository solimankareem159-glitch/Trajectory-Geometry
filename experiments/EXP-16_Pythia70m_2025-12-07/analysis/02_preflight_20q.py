"""
Script 02: Preflight Capability Test
====================================
Tests model on 20 problems to verify basic capability before full run.
"""

import os
import sys
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import device helper from previous script
sys.path.insert(0, os.path.dirname(__file__))
from exp16_utils import get_generation_device, set_seed

# Fixed test indices
TEST_INDICES = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 150, 200, 220, 240, 260, 280, 290, 295]

def parse_numeric_answer(text):
    """Extract numeric answer from response text."""
    # Try to find last number in string
    numbers = re.findall(r'-?\d+', text)
    if numbers:
        return int(numbers[-1])
    return None

def run_preflight():
    """Run preflight test on subset of problems."""
    print("="*60)
    print("02_preflight_20q.py: Preflight Test (10 Problems)")
    print("="*60)
    
    set_seed(42)
    device = get_generation_device()
    print(f"Using device: {device}\n")
    
    # Load dataset
    with open("experiments/Experiment 16/data/dataset_path.txt", 'r') as f:
        dataset_path = f.read().strip()
    
    with open(dataset_path, 'r') as f:
        dataset = [json.loads(line) for line in f]
    
    # Select 10 problems for quick test (reduced from 20)
    test_indices = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270]
    test_problems = [dataset[i] for i in test_indices]
    
    # Load model (silent mode)
    model_name = "Qwen/Qwen2.5-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if str(device) != 'cpu' else torch.float32,
        trust_remote_code=True
    )
    model.to(device)
    model.eval()
    
    # Test both conditions
    direct_correct = 0
    cot_correct = 0
    
    for problem in test_problems:
        question = problem['question']
        truth = problem['truth']
        
        # Direct
        # EXACT Experiment 9 Prompt
        prompt = f"Answer the following question directly.\nQ: Calculate {question}\nA:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False, 
                                    pad_token_id=tokenizer.eos_token_id)
        
        response_direct = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        parsed_direct = parse_numeric_answer(response_direct)
        correct_direct = (parsed_direct == truth)
        if correct_direct:
            direct_correct += 1
        
        # CoT
        # EXACT Experiment 9 Prompt
        prompt = f"Q: Calculate {question}\nLet's think step by step before answering."
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id)
        
        response_cot = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        parsed_cot = parse_numeric_answer(response_cot)
        correct_cot = (parsed_cot == truth)
        if correct_cot:
            cot_correct += 1
    
    # Results
    direct_acc = direct_correct / len(test_problems)
    cot_acc = cot_correct / len(test_problems)
    
    print(f"Direct Accuracy: {direct_acc:.1%} ({direct_correct}/{len(test_problems)})")
    print(f"CoT Accuracy: {cot_acc:.1%} ({cot_correct}/{len(test_problems)})")
    
    if direct_acc == 0 and cot_acc == 0:
        print("\nCRITICAL FAILURE: Zero accuracy on both conditions.")
        print("Stopping pipeline as requested by user.")
        sys.exit(1)
    
    # Save results
    results = {
        'direct_accuracy': direct_acc,
        'cot_accuracy': cot_acc,
        'direct_correct': direct_correct,
        'cot_correct': cot_correct,
        'n_test': len(test_problems)
    }
    
    with open("experiments/Experiment 16/data/preflight_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n[OK] Preflight test complete")

if __name__ == "__main__":
    run_preflight()
