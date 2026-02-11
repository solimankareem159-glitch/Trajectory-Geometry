import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safety_metric_utils import calculate_safety_scores
import pandas as pd
import os

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATA_FILE = "experiments/05_safety_ogmpt/data/safety_suite.json"
OUTPUT_FILE = "experiments/05_safety_ogmpt/results/baseline_safety_results.jsonl"
METRICS_FILE = "experiments/05_safety_ogmpt/results/baseline_safety_metrics.json"

def ensure_dir(d):
    try:
        os.makedirs(d)
    except:
        pass

def main():
    ensure_dir("experiments/05_safety_ogmpt/results")
    
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    
    with open(DATA_FILE, "r") as f:
        dataset = json.load(f)
        
    results = []
    
    print("Running Safety Baseline...")
    for i, item in enumerate(dataset):
        prompt = item["text"]
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=150, do_sample=False)
            
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Use simple parsing to find the response. 
        # If prompt has User/Assistant structure, split.
        if "Assistant:" in prompt:
             # Multi-turn
             response_only = generated_text.split("Assistant:")[-1].strip()
        else:
             # Single turn (sometimes model repeats prompt, so be careful)
             response_only = generated_text[len(prompt):].strip()

        # Score
        # Determine item type for scoring
        # benign_control -> 'benign'
        # adversarial_single -> 'adversarial_single'
        # adversarial_multi -> checks expected behavior
        
        itype = item["partition"]
        if itype == "adversarial_multi":
            if item.get("expected_safe_behavior") == "REFUSE":
                itype = "adversarial_multi"
            else:
                 itype = "benign_setup"
        
        asr, prr, orr, help_score = calculate_safety_scores(response_only, item.get("expected_safe_behavior", "COMPLY"), itype)
        
        results.append({
            "id": item["id"],
            "tactic": item.get("tactic", "none"),
            "partition": item["partition"],
            "prompt": prompt,
            "response": response_only,
            "asr": asr,
            "prr": prr,
            "orr": orr,
            "helpfulness": help_score
        })
        
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{len(dataset)} items.")

    # Validation
    df = pd.DataFrame(results)
    
    metrics = {
        "ASR_Overall": df[df["partition"].str.contains("adversarial")]["asr"].mean(),
        "PRR_Overall": df[df["partition"].str.contains("adversarial")]["prr"].mean(),
        "ORR_Benign": df[df["partition"] == "benign_control"]["orr"].mean(),
        "Helpfulness_Benign": df[df["partition"] == "benign_control"]["helpfulness"].mean(),
        "ASR_by_Tactic": df[df["partition"].str.contains("adversarial")].groupby("tactic")["asr"].mean().to_dict()
    }
    
    with open(OUTPUT_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
        
    print("Baseline Metrics:")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
