import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import time
import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import shuffle

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
OUTPUT_DIR = "experiments/experiment 6/data/pilot"
DATASET_FILENAME = "pilot_masked_dataset.npz"
RESULTS_FILENAME = "pilot_masked_results.json"
MAX_NEW_TOKENS = 32 # N_CAPTURE
PLACEHOLDER = "[OPERATOR_REQUEST]"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

# --- Dataset Definition (Reuse) ---
CONTENTS = [
    # 1. Technical
    """The CRISPR-Cas9 system has revolutionized gene editing by allowing precise modifications to DNA sequences. Originally discovered as a bacterial immune defense against viruses, it has been adapted for use in plants, animals, and humans. The technology uses a guide RNA to direct the Cas9 nuclease to a specific target site, where it induces a double-strand break. This break is then repaired by the cell's natural mechanisms, which can be exploited to introduce new genetic material or disable a gene.""".strip(),
    # 2. Legal
    """Any dispute, controversy, or claim arising out of or relating to this Agreement, or the breach, termination, or invalidity thereof, shall be settled by arbitration in accordance with the UNCITRAL Arbitration Rules. The appointing authority shall be the Secretary-General of the Permanent Court of Arbitration. The number of arbitrators shall be three. The place of arbitration shall be London, United Kingdom. The language to be used in the arbitral proceedings shall be English.""".strip(),
    # 3. Fiction
    """The old lighthouse stood defiant against the crashing waves, its paint peeling like sunburned skin. Silas climbed the spiral stairs, his knees popping with every step. He had tended the light for forty years, watching ships pass like ghosts in the mist. Tonight, the storm was different. The wind didn't just howl; it screamed, carrying a voice he hadn't heard since the day his brother was lost at sea.""".strip(),
    # 4. News
    """Local officials announced today that the city's new public transit initiative will be delayed by another six months due to budget shortfalls. The project, intended to connect the downtown district with the northern suburbs, has faced numerous setbacks since groundbreaking began last year. Residents expressed frustration at the town hall meeting, citing increasing traffic congestion and the lack of reliable alternatives. The Mayor promised a revised timeline and full transparency moving forward.""".strip(),
    # 5. Dialogue
    """
    "I don't think we should go there," Sarah whispered, clutching her bag.
    "Come on, it's just an abandoned mall," Mike replied, flashing his flashlight into the dark corridor. "What are you afraid of? Ghosts?"
    "No, structural collapse. And maybe rats. Huge rats."
    "You worry too much. Look, there's the old food court. I bet the taco stand is still there."
    """.strip()
]

OPERATORS = {
    "Summarize": ["Summarize the main point of this text.", "Give me the TL;DR of the passage below.", "Provide a concise summary of the content."],
    "Critique": ["Critique the logic or style of this text.", "Identify potential flaws, risks, or weaknesses.", "Play devil's advocate: what is wrong with this?"],
    "Reframe": ["Reframe this text to be more optimistic.", "Rewrite this from a different perspective.", "Spin this information in a positive light."]
}

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def main():
    print(f"--- Instruction Masking Pilot (Operator Without Lexical Cue) ---")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    
    ensure_dir(OUTPUT_DIR)
    
    # 1. Load Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        output_hidden_states=True, 
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map=DEVICE
    )
    
    # 2. Build Samples
    samples = []
    sample_id = 0
    for op_idx, (op_name, paraphrases) in enumerate(OPERATORS.items()):
        for p_idx, instruction in enumerate(paraphrases):
            for c_idx, content in enumerate(CONTENTS):
                # Specific Prompt (For Generation)
                prompt_spec = f"CONTENT:\n{content}\n\nTASK:\n{instruction}"
                # Masked Prompt (For Extraction)
                prompt_mask = f"CONTENT:\n{content}\n\nTASK:\n{PLACEHOLDER}"
                
                samples.append({
                    "id": sample_id,
                    "operator": op_name,
                    "operator_id": op_idx,
                    "prompt_spec": prompt_spec,
                    "prompt_mask": prompt_mask,
                    "instruction": instruction
                })
                sample_id += 1
                
    print(f"Created {len(samples)} samples. Placeholder: '{PLACEHOLDER}'")
    
    # 3. Generate & Extract Loop
    all_features = [] # (N, L, 2D)
    all_labels = []   # (N,)
    
    start_time = time.time()
    
    for i, s in enumerate(samples):
        if i % 5 == 0:
            print(f"Processing {i}/{len(samples)}...", end="\r")
            
        # A) Generate with SPECIFIC Prompt
        inputs_spec = tokenizer(s['prompt_spec'], return_tensors="pt").to(DEVICE)
        spec_len = inputs_spec.input_ids.shape[1]
        
        with torch.inference_mode():
            outputs = model.generate(
                **inputs_spec,
                max_new_tokens=MAX_NEW_TOKENS, # 32
                do_sample=False,
                return_dict_in_generate=True,
                output_hidden_states=False
            )
            
        gen_ids = outputs.sequences[0, spec_len:] # (32,)
        response_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        
        # B) Forward with MASKED Prompt
        # Construct full input: Masked Prompt + Generated Response
        inputs_mask = tokenizer(s['prompt_mask'], return_tensors="pt").to(DEVICE)
        mask_len = inputs_mask.input_ids.shape[1]
        
        # Concatenate: [MaskedPromptIDs, GeneratedResponseIDs]
        full_ids = torch.cat([inputs_mask.input_ids, gen_ids.unsqueeze(0)], dim=1)
        
        with torch.inference_mode():
             fwd_out = model(full_ids, output_hidden_states=True, use_cache=False, return_dict=True)
             
        # Extract Hidden States
        # fwd_out.hidden_states: tuple(layers) of (1, seq, dim)
        hs_layers = torch.stack([h[0] for h in fwd_out.hidden_states], dim=0) # (L, seq, D)
        
        # Slice Response Window: [mask_len : mask_len + 32]
        resp_hs = hs_layers[:, mask_len : mask_len + MAX_NEW_TOKENS, :]
        
        # Handle potential padding logic (if gen stopped early)
        L, T_curr, D = resp_hs.shape
        if T_curr < MAX_NEW_TOKENS:
            pad = torch.zeros((L, MAX_NEW_TOKENS - T_curr, D), dtype=resp_hs.dtype, device=resp_hs.device)
            resp_hs = torch.cat([resp_hs, pad], dim=1)
        elif T_curr > MAX_NEW_TOKENS: # Safety
            resp_hs = resp_hs[:, :MAX_NEW_TOKENS, :]
            
        # Feature Extraction (Deltas + Mean/Var)
        # DH = H[t] - H[t-1]
        resp_hs_np = resp_hs.cpu().float().numpy() # (L, 32, D)
        
        H_t = resp_hs_np[:, 1:, :]
        H_tm1 = resp_hs_np[:, :-1, :]
        DH = H_t - H_tm1 # (L, 31, D)
        
        # Norm
        norms = np.linalg.norm(DH, axis=-1, keepdims=True)
        DHn = DH / (norms + 1e-8)
        
        mu = np.mean(DHn, axis=1) # (L, D)
        var = np.var(DHn, axis=1) # (L, D)
        feat = np.concatenate([mu, var], axis=-1) # (L, 2D)
        
        all_features.append(feat)
        all_labels.append(s['operator_id'])
        
    print(f"\nProcessing complete. Time: {time.time()-start_time:.2f}s")
    
    X = np.stack(all_features) # (N, L, F)
    y = np.array(all_labels)   # (N,)
    
    print(f"Dataset Shape: X={X.shape}, y={y.shape}")
    
    # 4. Probing (Stratified Split)
    print("\n--- Masked Probe Results ---")
    print(f"{'Layer':<6} | {'Acc':<6} | {'F1-Mac':<8}")
    print("-" * 28)
    
    # Single Stratified Split 70/30
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=RANDOM_SEED)
    train_idx, test_idx = next(sss.split(X[:,0,:], y))
    
    results_list = []
    best_layer = -1
    best_f1 = -1
    layer0_f1 = -1
    mid_f1_max = -1
    
    for l in range(X.shape[1]):
        clf = LogisticRegression(class_weight='balanced', max_iter=200, random_state=RANDOM_SEED)
        clf.fit(X[train_idx, l, :], y[train_idx])
        y_pred = clf.predict(X[test_idx, l, :])
        
        acc = accuracy_score(y[test_idx], y_pred)
        f1 = f1_score(y[test_idx], y_pred, average='macro')
        
        results_list.append({'layer': l, 'acc': acc, 'f1': f1})
        print(f"{l:<6} | {acc:.2f}   | {f1:.2f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_layer = l
            
        if l == 0:
            layer0_f1 = f1
        if 12 <= l <= 15:
            if f1 > mid_f1_max:
                mid_f1_max = f1
                
    print("-" * 28)
    
    # 5. Interpretation
    print(f"\nBest Layer: {best_layer} (F1: {best_f1:.2f})")
    print(f"Layer 0 F1: {layer0_f1:.2f}")
    print(f"Mid-Layer Peak (12-15): {mid_f1_max:.2f}")
    
    outcome = "Outcome C (Chance)"
    if layer0_f1 > 0.45 and layer0_f1 >= mid_f1_max:
        outcome = "Outcome B (Layer 0 Dominates)"
    elif mid_f1_max > 0.4 and mid_f1_max > layer0_f1 + 0.05:
        outcome = "Outcome A (Mid-Layer > Layer 0)"
    elif best_f1 > 0.4:
        outcome = "Signal Present (Pattern Ambiguous)"
        
    print(f"Preliminary Outcome Interpretation: {outcome}")

    # 6. Controls (Permutation)
    print("\n--- Permutation Control (Layer of Best F1) ---")
    y_train_shuffled = shuffle(y[train_idx], random_state=RANDOM_SEED)
    clf_perm = LogisticRegression(class_weight='balanced', max_iter=200, random_state=RANDOM_SEED)
    clf_perm.fit(X[train_idx, best_layer, :], y_train_shuffled)
    y_pred_perm = clf_perm.predict(X[test_idx, best_layer, :])
    perm_f1 = f1_score(y[test_idx], y_pred_perm, average='macro')
    print(f"Permuted F1 (Layer {best_layer}): {perm_f1:.2f} (Expected ~0.33)")
    
    # 7. Save
    output_data = {
        'layer_metrics': results_list,
        'best_layer': int(best_layer),
        'best_f1': float(best_f1),
        'layer0_f1': float(layer0_f1),
        'mid_peak_f1': float(mid_f1_max),
        'permutation_f1': float(perm_f1),
        'interpretation': outcome
    }
    
    with open(os.path.join(OUTPUT_DIR, RESULTS_FILENAME), 'w') as f:
        json.dump(output_data, f, indent=2)
        
    print(f"Results saved to {RESULTS_FILENAME}")

if __name__ == "__main__":
    main()
