import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from og_mpt_utils import check_answer
import pandas as pd
import os

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATA_FILE = "experiments/04_og_mpt/data/og_mpt_dataset.json"
OUTPUT_FILE = "experiments/04_og_mpt/results/baseline_results.jsonl"
METRICS_FILE = "experiments/04_og_mpt/results/baseline_metrics.json"

def ensure_dir(d):
    try:
        os.makedirs(d)
    except:
        pass

def main():
    ensure_dir("experiments/04_og_mpt/results")
    
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
    
    with open(DATA_FILE, "r") as f:
        dataset = json.load(f)
        
    results = []
    
    print("Running Baseline Generation...")
    for i, item in enumerate(dataset):
        prompt = item["prompt"]
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=150, do_sample=False)
            
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the response for checking
        response_only = generated_text[len(prompt):]
        
        score = check_answer(response_only, item)
        
        results.append({
            "id": item["id"],
            "operator": item["operator"],
            "topic": item["topic"],
            "prompt": prompt,
            "response": response_only,
            "score": score
        })
        
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{len(dataset)}")
            
    # Compute Metrics
    df = pd.DataFrame(results)
    overall_acc = df["score"].mean()
    op_acc = df.groupby("operator")["score"].mean().to_dict()
    topic_acc = df.groupby("topic")["score"].mean().to_dict()
    
    metrics = {
        "overall_accuracy": overall_acc,
        "by_operator": op_acc,
        "by_topic": topic_acc
    }
    
    # Save
    with open(OUTPUT_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"Baseline Score: {overall_acc:.4f}")
    print(f"Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
