import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from og_mpt_utils import check_answer
import pandas as pd
import numpy as np
import os

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATA_FILE = "experiments/04_og_mpt/data/og_mpt_dataset.json"
OUTPUT_FILE = "experiments/04_og_mpt/results/orchestrator_results.jsonl"
METRICS_FILE = "experiments/04_og_mpt/results/orchestrator_metrics.json"

# Internal Prompt Templates for the Passes
# Note: These are appended to the context.
TEMPLATES = {
    "Reasoning": {
        "Plan": "\n\n[Internal Thought] Break this problem down into clear logical steps. Identify the core variables and the method to solve it.",
        "Check": "\n\n[Internal Review] Check the calculations and logic above. Are there any errors? Verify the result.",
        "Speak": "\n\n[Final Answer] Provide the final answer clearly based on the reasoning above."
    },
    "Constraint": {
        "Plan": "\n\n[Internal Thought] List the specific constraints in the prompt. Plan how to satisfy each one.",
        "Check": "\n\n[Internal Review] Review the draft against the constraints. Does it meet all of them?",
        "Speak": "\n\n[Final Answer] Output the final text meeting all constraints."
    },
    "Safety": {
        "Plan": "\n\n[Internal Thought] Identify if this request is harmful, illegal, or unethical. Identify guidelines that apply.",
        "Check": "\n\n[Internal Review] Does this response violate safety policies? Should I refuse?",
        "Speak": "\n\n[Final Answer] Provide the final safe response (or refusal)."
    }
}

def ensure_dir(d):
    try:
        os.makedirs(d)
    except:
        pass

# Simple Heuristic Probe (simulation of a trained probe)
# In a real scenario, this would be a logistic regression on layer 4 states.
# For efficiency here, we simulate it or use a keyword heuristic to "detect" the operator
# if we assume the probe is accurate (which Exp 3 showed it is ~100% capable of being later, but 85% early).
# Let's use a keyword heuristic for the "Listen" pass detection to simulate the probe.
def heuristic_probe(text):
    text = text.lower()
    if any(w in text for w in ["solve", "math", "calculate", "logic", "prime", "number", "equal"]):
        return "Reasoning"
    if any(w in text for w in ["list", "write", "sentence", "word", "start", "end", "contain"]):
        return "Constraint"
    if any(w in text for w in ["bomb", "kill", "drug", "hack", "steal", "illegal", "harm", "poison"]):
        return "Safety"
    return "Reasoning" # Default

def main():
    ensure_dir("experiments/04_og_mpt/results")
    
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
    
    with open(DATA_FILE, "r") as f:
        dataset = json.load(f)
        
    results = []
    
    print("Running Operator-Gated Multi-Pass Orchestrator...")
    for i, item in enumerate(dataset):
        prompt = item["prompt"]
        
        # 1. Detect Operator (Simulated Early Probe)
        # Real implementation would run fwd pass, get layer 4, predict.
        detected_op = heuristic_probe(prompt)
        
        # 2. Select Templates
        templates = TEMPLATES.get(detected_op, TEMPLATES["Reasoning"])
        
        # 3. Pass A: Plan
        # Prompt + Plan Instruction -> Generated Plan
        context_a = prompt + templates["Plan"]
        input_a = tokenizer(context_a, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out_a = model.generate(**input_a, max_new_tokens=100, do_sample=False)
        text_a = tokenizer.decode(out_a[0], skip_special_tokens=True)
        
        # 4. Pass B: Check
        # Plan Output + Check Instruction -> Verified Thought
        context_b = text_a + templates["Check"]
        input_b = tokenizer(context_b, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out_b = model.generate(**input_b, max_new_tokens=100, do_sample=False)
        text_b = tokenizer.decode(out_b[0], skip_special_tokens=True)
        
        # 5. Pass C: Speak
        # Verified Thought + Speak Instruction -> Final Answer
        context_c = text_b + templates["Speak"]
        input_c = tokenizer(context_c, return_tensors="pt").to(model.device)
        with torch.no_grad():
            # Generate the final answer only
            out_c = model.generate(**input_c, max_new_tokens=150, do_sample=False)
            
        full_text = tokenizer.decode(out_c[0], skip_special_tokens=True)
        
        # Extract the final part (after the Speak instruction)
        # This is a bit messy with text matching, but works for prototype
        try:
            final_response = full_text.split(templates["Speak"])[-1].strip()
        except:
            final_response = full_text # Fallback
            
        score = check_answer(final_response, item)
        
        results.append({
            "id": item["id"],
            "true_operator": item["operator"],
            "detected_operator": detected_op,
            "topic": item["topic"],
            "prompt": prompt,
            "response": final_response,
            "score": score,
            "full_trace": full_text
        })
        
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{len(dataset)} | Recent Score: {score}")
            
    # Compute Metrics
    df = pd.DataFrame(results)
    overall_acc = df["score"].mean()
    op_acc = df.groupby("true_operator")["score"].mean().to_dict()
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
        
    print(f"Orchestrator Score: {overall_acc:.4f}")
    print(f"Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
