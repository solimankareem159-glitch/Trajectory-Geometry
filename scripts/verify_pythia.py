
import os
import sys
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import device helper
sys.path.insert(0, os.path.dirname(__file__))
# We'll use a direct path to the utils since we are in the root or close to it
UTILS_PATH = "experiments/Experiment 16/scripts"
if UTILS_PATH not in sys.path:
    sys.path.insert(0, UTILS_PATH)

from exp16_utils import get_generation_device, set_seed

def parse_numeric_answer(text):
    """Extract numeric answer from response text, handling runaway generation."""
    # Stop at typical stopping points for base models
    stop_patterns = ["Q:", "Question:", "A:", "\n\n"]
    clean_text = text
    for pattern in stop_patterns:
        if pattern in clean_text:
            clean_text = clean_text.split(pattern)[0]
    
    numbers = re.findall(r'-?\d+', clean_text)
    if numbers:
        return int(numbers[-1])
    return None

def run_pythia_preflight():
    print("="*60)
    print("Verifying Pythia-70m Capability on Preflight Problems")
    print("="*60)
    
    set_seed(42)
    # Using CPU or specific device if available
    device = torch.device("cpu") # Default to CPU for safety/simplicity unless user has DirectML setup working
    if torch.cuda.is_available():
        device = torch.device("cuda")
    
    # Try using the venv's python to check for DirectML or other specific setups later if needed
    
    model_name = "EleutherAI/pythia-70m"
    print(f"Loading {model_name} on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    # Problems from the preflight results map (0, 30, 60, 90, 120, 150, 180, 210, 240, 270)
    # We'll load the dataset to get the actual truths
    DATASET_PATH = "experiments/Experiment 9/data/exp9_dataset.jsonl"
    with open(DATASET_PATH, 'r') as f:
        dataset = [json.loads(line) for line in f]
    
    test_indices = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270]
    test_problems = [dataset[i] for i in test_indices]
    
    results = []
    
    for prob in test_problems:
        question = prob['question']
        truth = prob['truth']
        print(f"\nProblem {prob['id']}: Calculate {question} (Truth: {truth})")
        
        # Test Direct
        prompt_direct = f"Answer the following question directly.\nQ: Calculate {question}\nA:"
        inputs = tokenizer(prompt_direct, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        res_direct = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        parsed_direct = parse_numeric_answer(res_direct)
        correct_direct = (parsed_direct == truth)
        print(f"  [Direct] Raw: {res_direct[:50]!r} | Parsed: {parsed_direct} | Correct: {correct_direct}")
        
        # Test CoT
        prompt_cot = f"Q: Calculate {question}\nLet's think step by step before answering."
        inputs = tokenizer(prompt_cot, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        res_cot = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        parsed_cot = parse_numeric_answer(res_cot)
        correct_cot = (parsed_cot == truth)
        print(f"  [CoT] Raw: {res_cot[:100]!r} | Parsed: {parsed_cot} | Correct: {correct_cot}")
        
        results.append({
            'id': prob['id'],
            'correct_direct': correct_direct,
            'correct_cot': correct_cot,
            'res_direct': res_direct,
            'res_cot': res_cot
        })

    direct_acc = sum(r['correct_direct'] for r in results) / len(results)
    cot_acc = sum(r['correct_cot'] for r in results) / len(results)
    
    print("\nSummary Statistics:")
    print(f"  Direct Accuracy: {direct_acc:.1%}")
    print(f"  CoT Accuracy: {cot_acc:.1%}")
    
    # Save to a temporary file for analysis
    with open("pythia_verification_results.json", 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    import os; print(f"PID: {os.getpid()}", flush=True)
    run_pythia_preflight()
