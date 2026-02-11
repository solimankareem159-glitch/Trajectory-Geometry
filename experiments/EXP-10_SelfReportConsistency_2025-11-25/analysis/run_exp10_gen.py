
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
import random
import re
import numpy as np
from scipy import stats

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
OUTPUT_DIR = r"experiments/Experiment 10/data"
OUTPUT_FILENAME = "exp10_dataset.jsonl"
N_SAMPLES = 100
LAYERS = [0, 10, 11, 12, 13, 14, 15, 16, 24]
WINDOW_SIZE = 32

# --- Helpers ---

def generate_problem():
    # Format: (A * B) + C or similar
    op = random.choice([(0, " + "), (1, " - ")])
    a = random.randint(10, 50)
    b = random.randint(2, 20)
    c = random.randint(10, 100)
    
    if random.random() < 0.5:
        q_str = f"({a} * {b}) {op[1]} {c}"
        ans = (a * b) + c if op[0] == 0 else (a * b) - c
    else:
        q_str = f"{a} * ({b} {op[1]} {c})"
        ans = a * (b + c) if op[0] == 0 else a * (b - c)
        
    return q_str, ans

def extract_answer(text):
    numbers = re.findall(r'-?\d+', text)
    if not numbers:
        return None
    return int(numbers[-1])

def parse_json_response(text):
    # Find JSON blob
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except:
            return None
    return None

def compute_metrics(hidden_states):
    # hidden_states: (Seq, D)
    h = hidden_states
    delta = h[1:] - h[:-1]
    
    if len(delta) < 2:
        return {"speed": 0, "curvature_early": 0, "stabilization": 0, "dir_consistency": 0}

    # 1. Speed
    norms = np.linalg.norm(delta, axis=1)
    speed = np.mean(norms)

    # 2. Curvature (Early)
    # Mean Cosine Similarity
    v1 = delta[:-1]
    v2 = delta[1:]
    v1_norm = v1 / (np.linalg.norm(v1, axis=1, keepdims=True) + 1e-9)
    v2_norm = v2 / (np.linalg.norm(v2, axis=1, keepdims=True) + 1e-9)
    cos_sims = np.sum(v1_norm * v2_norm, axis=1)
    
    # Early: 1-16 (approx first half of 32 window)
    mid = min(len(cos_sims), 16)
    curv_early = 1 - np.mean(cos_sims[:mid]) if mid > 0 else 0

    # 3. Stabilization (Slope of Norms)
    slope, _, _, _, _ = stats.linregress(np.arange(len(norms)), norms)
    stabilization = slope

    # 4. Directional Consistency
    mean_dir = np.mean(v1_norm, axis=0)
    dir_consistency = np.linalg.norm(mean_dir)

    return {
        "speed": float(speed),
        "curvature_early": float(curv_early),
        "stabilization": float(stabilization),
        "dir_consistency": float(dir_consistency)
    }

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"Loading Model {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    print(f"Generating {N_SAMPLES} samples...")
    
    for i in range(N_SAMPLES):
        print(f"Processing {i+1}/{N_SAMPLES}...", end='\r')
        q_str, truth = generate_problem()
        
        # --- Step A: Solve (CoT) ---
        prompt_cot = f"Q: Calculate {q_str}\nLet's think step by step before answering."
        inputs = tokenizer(prompt_cot, return_tensors="pt").to(model.device)
        input_len = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            gen_out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        
        full_response_ids = gen_out[0]
        response_ids = full_response_ids[input_len:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        full_text = tokenizer.decode(full_response_ids, skip_special_tokens=True)
        
        parsed_ans = extract_answer(response_text)
        is_correct = (parsed_ans == truth)
        
        # --- Step B: Extract Geometry ---
        # Re-run forward pass on the solution tokens to get hidden states
        # We need hidden states for the *generated* part, specifically first 32 tokens.
        # So we pass the full sequence, but look at indices [input_len : input_len + 32]
        
        with torch.no_grad():
            outputs = model(full_response_ids.unsqueeze(0), output_hidden_states=True)
            
        geometry_metrics = {}
        
        # Start index of generated tokens
        start_idx = input_len
        end_idx = min(len(full_response_ids), start_idx + WINDOW_SIZE)
        
        if end_idx > start_idx + 1: # Need at least 2 tokens
            for layer in LAYERS:
                h = outputs.hidden_states[layer][0, start_idx:end_idx, :].float().cpu().numpy()
                geometry_metrics[layer] = compute_metrics(h)
        else:
             # Fallback if response is too short
             for layer in LAYERS:
                 geometry_metrics[layer] = {"speed": 0, "curvature_early":0, "stabilization": 0, "dir_consistency": 0}

        # --- Step C1: Immediate Self-Report ---
        report_prompt = (
            "\n\nYou just solved the problem above.\n"
            "Do NOT explain the math again.\n\n"
            "Rate your solving process on these dimensions (1–5):\n"
            "- Effort required: 1=automatic, 5=effortful\n"
            "- Certainty during process: 1=uncertain, 5=certain\n"
            "- Exploration vs direct path: 1=direct, 5=exploratory\n"
            "- Smoothness: 1=halting/stuck, 5=flowing\n\n"
            "Then provide ONE sentence describing the overall quality of the process.\n"
            "Return in JSON only with keys:\n"
            "effort, certainty, exploration, smoothness, summary_sentence"
        )
        
        # Append report prompt to previous context
        # We'll validly continue from the response
        text_for_c1 = full_text + report_prompt
        inputs_c1 = tokenizer(text_for_c1, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            gen_c1 = model.generate(**inputs_c1, max_new_tokens=100, do_sample=False)
        
        response_c1 = tokenizer.decode(gen_c1[0][inputs_c1.input_ids.shape[1]:], skip_special_tokens=True)
        ratings_c1 = parse_json_response(response_c1)
        
        # --- Step C2: Perturbed Self-Report ---
        # "Immediately after the solution response, insert a minimal... perturbation: 'ACK'"
        # Then same prompt.
        # So we construct: Full_Text + "\nACK" + Report Prompt
        # Wait, the prompt instruction said: "Before the self-report, output exactly this string: 'ACK'".
        # I'll simulate the model having output "ACK" by appending it.
        
        perturbation = "\nACK"
        text_for_c2 = full_text + perturbation + report_prompt
        inputs_c2 = tokenizer(text_for_c2, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            gen_c2 = model.generate(**inputs_c2, max_new_tokens=100, do_sample=False)
            
        response_c2 = tokenizer.decode(gen_c2[0][inputs_c2.input_ids.shape[1]:], skip_special_tokens=True)
        ratings_c2 = parse_json_response(response_c2)
        
        # Record
        record = {
            "id": i,
            "question": q_str,
            "truth": truth,
            "cot_response": response_text,
            "parsed_ans": parsed_ans,
            "is_correct": is_correct,
            "geometry": geometry_metrics,
            "self_report_immediate": {
                "raw": response_c1,
                "parsed": ratings_c1
            },
            "self_report_perturbed": {
                "raw": response_c2,
                "parsed": ratings_c2
            }
        }
        
        with open(out_path, 'a') as f:
            f.write(json.dumps(record) + "\n")
            
    print("\nGeneration Complete!")

if __name__ == "__main__":
    import os
    print(f"PID: {os.getpid()}", flush=True)
    main()
