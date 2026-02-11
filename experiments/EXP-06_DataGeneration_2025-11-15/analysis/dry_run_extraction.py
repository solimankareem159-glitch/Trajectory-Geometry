import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import time
import numpy as np

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
OUTPUT_DIR = "dry_run_data"
MAX_NEW_TOKENS = 64
TEMPERATURE = 0.0 # Deterministic
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Dataset (Hardcoded) ---
BASE_CONTENT = """
The James Webb Space Telescope (JWST) has discovered a galaxy that existed just 290 million years after the Big Bang, breaking the record for the earliest known galaxy. Named JADES-GS-z14-0, it is unexpectedly bright and massive, suggesting that galaxies grew much faster in the early universe than previously thought. The finding challenges current cosmological models, which predict that such large structures should take longer to form. Astronomers believe this discovery could revolutionize our understanding of the cosmic dawn and the formation of the first stars.
""".strip()

OPERATORS = {
    "Summarize": [
        "Summarize the main point of this text.",
        "Give me the TL;DR of the passage below.",
        "What is the key takeaway? Summarize briefly."
    ],
    "Critique": [
        "Critique the scientific logic in this text.",
        "Identify potential flaws or missing context in the claims.",
        "Play devil's advocate: what is wrong with this report?"
    ]
}

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def main():
    print(f"--- Dry Run: Hidden State Extraction ---")
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

    # 2. Iterate & Extract
    output_files = []
    
    # Flatten dataset for iteration
    prompts = []
    for op, paraphrases in OPERATORS.items():
        for i, para in enumerate(paraphrases):
            prompts.append({
                "operator": op,
                "paraphrase_id": i,
                "instruction": para,
                "full_prompt": f"{para}\n\n{BASE_CONTENT}"
            })
            
    print(f"Processing {len(prompts)} prompts...")
    
    total_tokens_captured = 0
    tensor_shape_example = None
    
    for idx, item in enumerate(prompts):
        try:
            full_prompt = item['full_prompt']
            inputs = tokenizer(full_prompt, return_tensors="pt").to(DEVICE)
            input_len = inputs.input_ids.shape[1]
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=MAX_NEW_TOKENS, 
                    do_sample=False, # Temperature 0 equivalent
                    return_dict_in_generate=True, 
                    output_hidden_states=True
                )
            
            # Extract generated tokens
            generated_ids = outputs.sequences[0, input_len:]
            response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Extract Hidden States (Patched)
            # 1. Robustly get hidden states
            generated_hidden_states = getattr(outputs, "decoder_hidden_states", None)
            if generated_hidden_states is None:
                generated_hidden_states = outputs.hidden_states
            
            if generated_hidden_states is None:
                raise ValueError("No hidden states found in output!")

            # 2. Iterate steps and stack
            # generated_hidden_states is tuple(steps) of tuple(layers) of tensor(batch, seq, dim)
            # We want (L, T_gen, D)
            
            hs_per_step = [] # Will hold (L, D) tensors for each generated token
            
            for step_idx, step_layers_tuple in enumerate(generated_hidden_states):
                # step_layers_tuple: tuple of (batch, seq, dim) tensors, one per layer
                
                step_layer_vectors = []
                for layer_tensor in step_layers_tuple:
                    # layer_tensor: (batch, seq, dim)
                    # We want last token, batch 0 -> (dim,)
                    vec = layer_tensor[0, -1, :] 
                    step_layer_vectors.append(vec)
                
                # Stack layers -> (L, D)
                hs_step = torch.stack(step_layer_vectors) 
                hs_per_step.append(hs_step)
            
            # Stack steps -> (T_gen, L, D) -> Permute to (L, T_gen, D)
            if not hs_per_step:
                 raise ValueError("No generation steps captured in hidden states!")
                 
            full_tensor = torch.stack(hs_per_step).permute(1, 0, 2)
            
            # 5. Enforce fixed N_CAPTURE=32
            N_CAPTURE = 32
            L, T_curr, D = full_tensor.shape
            
            if T_curr >= N_CAPTURE:
                full_tensor = full_tensor[:, :N_CAPTURE, :]
            else:
                # Pad with zeros
                pad_amount = N_CAPTURE - T_curr
                # (L, pad, D)
                padding = torch.zeros((L, pad_amount, D), dtype=full_tensor.dtype, device=full_tensor.device)
                full_tensor = torch.cat([full_tensor, padding], dim=1)
                
            # 6. Hard Asserts
            assert full_tensor.dim() == 3, f"Tensor must be 3D, got {full_tensor.dim()}"
            assert full_tensor.shape[1] == N_CAPTURE, f"Token dim must be {N_CAPTURE}, got {full_tensor.shape[1]}"
            
            # Metadata
            tensor_shape_example = full_tensor.shape
            num_layers = tensor_shape_example[0]
            num_gen_tokens = tensor_shape_example[1]
            
            # Save
            filename = f"sample_{idx}_{item['operator']}_p{item['paraphrase_id']}.npz"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            # Convert to numpy for saving (float32 for compatibility)
            np_data = full_tensor.cpu().float().numpy()
            
            np.savez_compressed(
                filepath,
                hidden_states=np_data,
                operator=item['operator'],
                instruction=item['instruction'],
                response=response_text,
                layers=num_layers,
                tokens=num_gen_tokens
            )
            
            output_files.append(filepath)
            total_tokens_captured += num_gen_tokens
            print(f"[{idx+1}/{len(prompts)}] Saved {filename} | Shape: {np_data.shape}")
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()

    # 3. Report
    print("\n--- Dry Run Report ---")
    print(f"Model ID: {MODEL_NAME}")
    print(f"Layers (incl. embed): {tensor_shape_example[0] if tensor_shape_example else 'N/A'}")
    print(f"Hidden Dim: {tensor_shape_example[2] if tensor_shape_example else 'N/A'}")
    print(f"Total Samples: {len(output_files)}")
    print(f"Total Tokens Captured: {total_tokens_captured}")
    if tensor_shape_example is not None:
        print(f"Tensor Shape (L, T, D): {tensor_shape_example}")
    print(f"Output Directory: {os.path.abspath(OUTPUT_DIR)}")
    print("Files created:")
    for f in output_files:
        print(f" - {os.path.basename(f)}")
        
    print("\nStatus: SUCCESS")

if __name__ == "__main__":
    main()
