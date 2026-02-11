import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import time
import numpy as np
import json

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
OUTPUT_DIR = "pilot_run_data"
FILENAME = "pilot_dataset.npz"
MAX_NEW_TOKENS = 64 # Generate enough to capture 32
N_CAPTURE = 32
TEMPERATURE = 0.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Dataset Definition ---

# 5 Diverse Contents
CONTENTS = [
    # 1. Technical / Scientific
    """
    The CRISPR-Cas9 system has revolutionized gene editing by allowing precise modifications to DNA sequences. Originally discovered as a bacterial immune defense against viruses, it has been adapted for use in plants, animals, and humans. The technology uses a guide RNA to direct the Cas9 nuclease to a specific target site, where it induces a double-strand break. This break is then repaired by the cell's natural mechanisms, which can be exploited to introduce new genetic material or disable a gene.
    """.strip(),
    
    # 2. Legal / Formal
    """
    Any dispute, controversy, or claim arising out of or relating to this Agreement, or the breach, termination, or invalidity thereof, shall be settled by arbitration in accordance with the UNCITRAL Arbitration Rules. The appointing authority shall be the Secretary-General of the Permanent Court of Arbitration. The number of arbitrators shall be three. The place of arbitration shall be London, United Kingdom. The language to be used in the arbitral proceedings shall be English.
    """.strip(),
    
    # 3. Fiction / Narrative
    """
    The old lighthouse stood defiant against the crashing waves, its paint peeling like sunburned skin. Silas climbed the spiral stairs, his knees popping with every step. He had tended the light for forty years, watching ships pass like ghosts in the mist. Tonight, the storm was different. The wind didn't just howl; it screamed, carrying a voice he hadn't heard since the day his brother was lost at sea.
    """.strip(),
    
    # 4. News / Journalistic
    """
    Local officials announced today that the city's new public transit initiative will be delayed by another six months due to budget shortfalls. The project, intended to connect the downtown district with the northern suburbs, has faced numerous setbacks since groundbreaking began last year. Residents expressed frustration at the town hall meeting, citing increasing traffic congestion and the lack of reliable alternatives. The Mayor promised a revised timeline and full transparency moving forward.
    """.strip(),
    
    # 5. Dialogue / Conversational
    """
    "I don't think we should go there," Sarah whispered, clutching her bag.
    "Come on, it's just an abandoned mall," Mike replied, flashing his flashlight into the dark corridor. "What are you afraid of? Ghosts?"
    "No, structural collapse. And maybe rats. Huge rats."
    "You worry too much. Look, there's the old food court. I bet the taco stand is still there."
    """.strip()
]

# 3 Operators x 3 Paraphrases
OPERATORS = {
    "Summarize": [
        "Summarize the main point of this text.",
        "Give me the TL;DR of the passage below.",
        "Provide a concise summary of the content."
    ],
    "Critique": [
        "Critique the logic or style of this text.",
        "Identify potential flaws, risks, or weaknesses.",
        "Play devil's advocate: what is wrong with this?"
    ],
    "Reframe": [
        "Reframe this text to be more optimistic.",
        "Rewrite this from a different perspective.",
        "Spin this information in a positive light."
    ]
}

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def main():
    print(f"--- Pilot Run: Experiment 6 (45 Samples) ---")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    
    ensure_dir(OUTPUT_DIR)
    
    # 1. Load Model & Tokenizer
    try:
        start_load = time.time()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            output_hidden_states=True, 
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map=DEVICE
        )
        print(f"Model loaded in {time.time() - start_load:.2f}s")
    except Exception as e:
        print(f"CRITICAL ERROR loading model: {e}")
        return

    # 2. Build Sample List (Crossed Design)
    samples = []
    sample_id_counter = 0
    
    op_names = list(OPERATORS.keys())
    
    for op_idx, (op_name, paraphrases) in enumerate(OPERATORS.items()):
        for p_idx, instruction in enumerate(paraphrases):
            for c_idx, content in enumerate(CONTENTS):
                
                # Template:
                # CONTENT:
                # {content}
                #
                # TASK:
                # {instruction}
                
                full_prompt = f"CONTENT:\n{content}\n\nTASK:\n{instruction}"
                
                samples.append({
                    "sample_id": sample_id_counter,
                    "operator": op_name,
                    "operator_id": op_idx,
                    "content_id": c_idx,
                    "paraphrase_id": p_idx,
                    "instruction": instruction,
                    "prompt_text": full_prompt,
                    # Placeholders
                    "response_text": "",
                    "hidden_states": None
                })
                sample_id_counter += 1
                
    print(f"Created {len(samples)} sample definitions.")
    
    # 3. Execution Loop
    
    all_hidden_states = [] # Will be list of (L, T=32, D) arrays
    all_metadata = [] 
    
    start_time = time.time()
    
    for i, s in enumerate(samples):
        # Timing stats
        if i > 0 and i % 5 == 0:
            elapsed = time.time() - start_time
            avg_per = elapsed / i
            print(f"[{i}/{len(samples)}] Avg: {avg_per:.2f}s/sample | Last: {s['operator']}...", end="\r")
        else:
             print(f"[{i+1}/{len(samples)}] {s['operator']} (Op{s['operator_id']}) | Content {s['content_id']}...", end="\r")
        
        try:
            inputs = tokenizer(s['prompt_text'], return_tensors="pt").to(DEVICE)
            input_len = inputs.input_ids.shape[1]
            
            # A) Generate (Fast - No Hidden States)
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=N_CAPTURE, # 32
                    do_sample=False, 
                    return_dict_in_generate=True, 
                    output_hidden_states=False 
                )
            
            # Extract Text
            generated_ids = outputs.sequences[0, input_len:]
            response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            s['response_text'] = response_text
            
            # B) Forward Pass (Single Pass for Hidden States)
            full_ids = outputs.sequences # (1, total_len)
            
            with torch.inference_mode():
                fwd_out = model(full_ids, output_hidden_states=True, use_cache=False, return_dict=True)
            
            # Stack layers: fwd_out.hidden_states is tuple of (1, seq, dim)
            # We want (L, seq, D)
            # h[0] gets the (seq, dim) from (1, seq, dim)
            hs_layers = torch.stack([h[0] for h in fwd_out.hidden_states], dim=0) 
            
            # Slice response window
            # We want tokens corresponding to generated_ids
            # Indices: [input_len, input_len + N_CAPTURE]
            resp_hs = hs_layers[:, input_len : input_len + N_CAPTURE, :]
            
            # Robustness: Pad or key assert
            L, T_curr, D = resp_hs.shape
            
            if T_curr < N_CAPTURE:
                # This happens if generation stopped early (EOS)
                pad_amount = N_CAPTURE - T_curr
                padding = torch.zeros((L, pad_amount, D), dtype=resp_hs.dtype, device=resp_hs.device)
                resp_hs = torch.cat([resp_hs, padding], dim=1)
            elif T_curr > N_CAPTURE:
                 # Should not happen with slicing range above, but safety
                 resp_hs = resp_hs[:, :N_CAPTURE, :]
                
            # C) Assertions
            if resp_hs.shape != (L, N_CAPTURE, D):
                raise ValueError(f"Shape mismatch: {resp_hs.shape} != ({L}, {N_CAPTURE}, {D})")
                
            # Move to CPU numpy
            np_tensor = resp_hs.cpu().float().numpy() # (L, 32, D)
            
            all_hidden_states.append(np_tensor)
            
            # Clean metadata (remove big objects)
            meta = s.copy()
            del meta['hidden_states']
            all_metadata.append(meta)
            
        except Exception as e:
            print(f"\nError on sample {i}: {e}")
            import traceback
            traceback.print_exc()
            return
            
    total_elapsed = time.time() - start_time
    print(f"\nGeneration Complete in {total_elapsed:.2f}s ({total_elapsed/len(samples):.2f}s/sample). Saving...")
    
    # 4. Save
    # Stack all: (N, L, T, D)
    X = np.stack(all_hidden_states)
    
    # Extract labels
    y_op = np.array([m['operator_id'] for m in all_metadata])
    y_content = np.array([m['content_id'] for m in all_metadata])
    
    save_path = os.path.join(OUTPUT_DIR, FILENAME)
    
    np.savez_compressed(
        save_path,
        hidden_states=X,
        operator_ids=y_op,
        content_ids=y_content,
        metadata=json.dumps(all_metadata) # Save as string
    )
    
    # 5. Validation Summary
    print("\n--- Validation Summary ---")
    print(f"Total Samples Generated: {len(all_metadata)}")
    
    op_counts = {}
    for m in all_metadata:
        op_counts[m['operator']] = op_counts.get(m['operator'], 0) + 1
    print(f"Distribution by Operator: {op_counts}")
    
    print(f"Tensor Shape X (N, L, T, D): {X.shape}")
    
    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"Output Path: {os.path.abspath(save_path)}")
    print(f"Approx Disk Usage: {file_size_mb:.2f} MB")
    
    if X.shape[1:] == (25, 32, 896):
        print("Shape Validation: PASS")
    else:
        print("Shape Validation: FAIL")

if __name__ == "__main__":
    main()
