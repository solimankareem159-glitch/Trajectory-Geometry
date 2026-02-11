
import pandas as pd
import json
import os
import sys
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(__file__))
from exp16_utils import get_generation_device

def parse_numeric_answer(text):
    numbers = re.findall(r'-?\d+', text)
    if numbers:
        return int(numbers[-1])
    return None

def main():
    print("Recovering missing metadata...")
    
    # Load existing
    csv_path = "experiments/Experiment 16/data/metadata.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        existing = set(zip(df['problem_id'], df['condition']))
    else:
        df = pd.DataFrame()
        existing = set()
    
    # Load dataset
    with open("experiments/Experiment 16/data/dataset_path.txt") as f:
         path = f.read().strip()
    with open(path) as f:
         dataset = [json.loads(line) for line in f]
         
    # Identify missing
    missing = []
    for p in dataset:
        if (p['id'], 'direct') not in existing:
             missing.append({'id': p['id'], 'condition': 'direct', 'question': p['question'], 'truth': p['truth']})
        if (p['id'], 'cot') not in existing:
             missing.append({'id': p['id'], 'condition': 'cot', 'question': p['question'], 'truth': p['truth']})
             
    print(f"Missing items: {len(missing)}")
    if not missing: 
        print("Nothing missing.")
        return

    # Load model
    device = get_generation_device()
    print(f"Loading model on {device}")
    model_name = "Qwen/Qwen2.5-1.5B"
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16 if str(device)!='cpu' else torch.float32).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    # Batch inference
    batch_size = 32
    print("Running inference...")
    
    new_rows = []
    for i in range(0, len(missing), batch_size):
        batch = missing[i:i+batch_size]
        prompts = []
        for t in batch:
            if t['condition'] == 'direct':
                prompts.append(f"Answer the following question directly.\nQ: Calculate {t['question']}\nA:")
            else:
                prompts.append(f"Q: Calculate {t['question']}\nLet's think step by step before answering.")
                
        # Use generous max tokens to cover CoT
        max_t = 200 
        inputs = tokenizer(prompts, return_tensors='pt', padding=True).to(device)
        input_len = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_t, pad_token_id=tokenizer.eos_token_id)
            
        for j, item in enumerate(batch):
            gen = outputs[j][input_len:]
            res = tokenizer.decode(gen, skip_special_tokens=True)
            parsed = parse_numeric_answer(res)
            
            new_rows.append({
                'problem_id': item['id'],
                'condition': item['condition'],
                'question': item['question'],
                'truth': item['truth'],
                'response': res,
                'parsed': parsed,
                'correct': parsed == item['truth'],
                'response_length_chars': len(res),
                'response_length_tokens': len(gen)
            })
            
    # Append
    new_df = pd.DataFrame(new_rows)
    final_df = pd.concat([df, new_df], ignore_index=True)
    # Sort
    final_df.sort_values(by=['problem_id', 'condition'], inplace=True)
    final_df.to_csv(csv_path, index=False)
    print(f"Recovered {len(new_df)} items. Total: {len(final_df)}")

if __name__ == "__main__": main()
