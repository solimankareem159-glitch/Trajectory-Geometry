import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import time
import numpy as np
import json
import psutil

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
OUTPUT_DIR = r"experiments/Experiment 7/data/exp7C_masked"
FEATURES_FILENAME = "features_masked.npy"  # User asked for .npy but memmap is likely safer/consistent. User spec said .npy, let's use memmap then save as npy at end or just use npy memmap as before? User spec says "features_masked.npy Shape: (N~2000, 25, 1792)". I will use memmap during run and rename/ensure format.
LABELS_FILENAME = "labels_masked.npy"
IDS_FILENAME = "ids_masked.npy"
MANIFEST_FILENAME = "manifest_masked.jsonl"
PROGRESS_FILENAME = "progress_masked.json"

N_OPS = 10
N_CONTENTS = 20
N_PARAS = 10
TOTAL_SAMPLES = N_OPS * N_CONTENTS * N_PARAS
N_CAPTURE = 32
L_LAYERS = 25 
D_MODEL = 896
FEATURE_DIM = D_MODEL * 2 # 1792

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_INTERVAL = 50

# --- Dataset Definitions (SAME AS EXP 6) ---
OPERATORS = [
    "Summarize", "Critique", "Reframe", "Decompose", "Compare",
    "Extend", "Translate", "Exemplify", "Formalize", "Question"
]

# We need the keys to iterate correctly, but values are IGNORED
PARAPHRASES = {
    "Summarize": [""] * 10, "Critique": [""] * 10, "Reframe": [""] * 10, "Decompose": [""] * 10, "Compare": [""] * 10,
    "Extend": [""] * 10, "Translate": [""] * 10, "Exemplify": [""] * 10, "Formalize": [""] * 10, "Question": [""] * 10
}

CONTENTS = [
    "The James Webb Space Telescope has captured the most detailed images of the early universe to date. Its infrared capabilities allow it to peer through dust clouds that obscured the view of Hubble. Astronomers are now analyzing these images to understand galaxy formation.",
    "Quantum computing utilizes qubits, which can exist in a superposition of states. This allows quantum computers to solve specific problems, like integer factorization, exponentially faster than classical supercomputers. However, maintaining coherence remains a major engineering challenge.",
    "CRISPR gene editing allows for precise alteration of DNA sequences in living organisms. Derived from a bacterial immune system, it uses a guide RNA to target specific genetic codes. Ethical concerns persist regarding its application in human germline editing.",
    "Machine learning models, particularly large language models, rely on vast amounts of training data. They learn statistical patterns to predict the next token in a sequence. Recent advancements have focused on alignment and reducing hallucinations in generated outputs.",
    "The Industrial Revolution marked a major turning point in history. It shifted production from hand tools to complex machinery, leading to urbanization. While it increased economic output, it also resulted in harsh working conditions and environmental pollution.",
    "The fall of the Roman Empire was a gradual process caused by internal strife and external invasions. Economic instability and political corruption weakened the state. The fracturing of the empire led to the Middle Ages in Europe.",
    "Democracy relies on the active participation of its citizens. Voting rights have expanded significantly over the centuries, though challenges to access remain. Free press and independent judiciary are essential checks on power.",
    "Globalization has interconnected economies and cultures worldwide. It facilitates the flow of goods, services, and information. Critics argue it can exacerbate inequality and erode local traditions.",
    "Photosynthesis is the process by which green plants use sunlight to synthesize nutrients. Chlorophyll in plant cells captures light energy to convert carbon dioxide and water into glucose. This process releases oxygen as a vital byproduct.",
    "The Amazon rainforest is often referred to as the lungs of the Earth. It hosts an immense biodiversity of flora and fauna. Deforestation poses a severe threat to this ecosystem and global climate stability.",
    "Ocean acidification is a consequence of increased carbon dioxide absorption by seawater. It impacts marine life, particularly organisms with calcium carbonate shells. Coral reefs are increasingly vulnerable to these chemical changes.",
    "Migration patterns of monarch butterflies span thousands of miles. They travel from Canada and the US to Mexico for the winter. Habitat loss and climate change are impacting their population numbers.",
    "Utilitarianism is an ethical theory that advocates for actions that maximize overall happiness. It assesses the moral worth of an act by its outcome. Critics argue it can justify the suffering of a minority for the majority's benefit.",
    "Existentialism emphasizes individual freedom and responsibility. It posits that existence precedes essence, meaning individuals define their own meaning. Philosophers like Sartre and Camus explored the absurdity of life.",
    "The Ship of Theseus is a thought experiment about identity. If all parts of a ship are replaced over time, is it still the same ship? It questions the nature of persistence and change.",
    "Epistemology is the branch of philosophy concerned with knowledge. It explores the distinction between justified belief and opinion. Skepticism challenges the possibility of certainty in our understanding of the world.",
    "Impressionism was a 19th-century art movement characterized by small, thin brush strokes. Artists sought to capture the changing qualities of light. Monet and Renoir are among its most famous practitioners.",
    "The Hero's Journey is a common narrative template in storytelling. It involves a hero who goes on an adventure, faces a crisis, and returns transformed. This structure is found in myths and modern movies alike.",
    "Minimalism in design focuses on simplicity and functionality. It strips away unnecessary elements to reveal the essential quality of a subject. 'Less is more' is a key mantra of this aesthetic.",
    "Haiku is a form of Japanese poetry consisting of three phrases. The structure typically follows a 5-7-5 syllable count. It often focuses on images from nature or a specific moment in time.",
]

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def main():
    print(f"--- Experiment 7C: Full-Scale Instruction Masking (Lexical Decontamination) ---")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {MODEL_NAME} | Device: {DEVICE}")
    print(f"Target: {TOTAL_SAMPLES} samples in {OUTPUT_DIR}/")

    ensure_dir(OUTPUT_DIR)

    # 1. Load Checkpoint or Init
    progress_path = os.path.join(OUTPUT_DIR, PROGRESS_FILENAME)
    start_idx = 0
    if os.path.exists(progress_path):
        with open(progress_path, 'r') as f:
            prog = json.load(f)
            start_idx = prog.get('completed_samples', 0)
            print(f"Resuming from sample {start_idx}...")

    # 2. Prepare Data Structures on Disk
    feat_path = os.path.join(OUTPUT_DIR, FEATURES_FILENAME)
    
    if start_idx == 0:
        # Create empty memmap
        fp = np.memmap(feat_path, dtype='float16', mode='w+', shape=(TOTAL_SAMPLES, L_LAYERS, FEATURE_DIM))
        labels_arr = np.zeros(TOTAL_SAMPLES, dtype=int)
        ids_arr = np.zeros((TOTAL_SAMPLES, 2), dtype=int)
        manifest = []
    else:
        fp = np.memmap(feat_path, dtype='float16', mode='r+', shape=(TOTAL_SAMPLES, L_LAYERS, FEATURE_DIM))
        labels_arr = np.load(os.path.join(OUTPUT_DIR, LABELS_FILENAME))
        ids_arr = np.load(os.path.join(OUTPUT_DIR, IDS_FILENAME))
        # Attempt to load manifest
        if os.path.exists(os.path.join(OUTPUT_DIR, MANIFEST_FILENAME)):
             with open(os.path.join(OUTPUT_DIR, MANIFEST_FILENAME), 'r') as f:
                manifest = [json.loads(line) for line in f]
        else:
             manifest = []

    # 3. Build Sample List (MATCHING EXP 6 ORDER)
    full_sample_list = []
    global_idx = 0
    
    # Placeholder String
    MASK_INSTRUCTION = "[PERFORM_THE_TASK]"

    for op_id, op_name in enumerate(OPERATORS):
        # We iterate N_PARAS times even though we don't use the text
        # This matches the loop structure of Exp 6
        for p_id in range(N_PARAS):
             for c_id, content in enumerate(CONTENTS):
                 # MASKED PROMPT
                 prompt = f"CONTENT:\n{content}\n\nTASK:\n{MASK_INSTRUCTION}"
                 
                 full_sample_list.append({
                     "sample_id": global_idx,
                     "operator": op_name,
                     "operator_id": op_id,
                     "content_id": c_id,
                     "paraphrase_id": p_id,
                     "instruction": "MASKED", # Don't store the original just in case, or store "MASKED"
                     "prompt": prompt
                 })
                 global_idx += 1
                 
    assert len(full_sample_list) == TOTAL_SAMPLES, f"Expected {TOTAL_SAMPLES}, got {len(full_sample_list)}"
    
    # 4. Load Model
    if start_idx < TOTAL_SAMPLES:
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            # Ensure model is on correct device
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, 
                output_hidden_states=True, 
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                device_map=DEVICE
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    else:
        print("Dataset already complete!")
        return

    # 5. Execution Loop
    start_time = time.time()
    batch_start = time.time()
    failure_count = 0
    
    print("\nStarting extraction...")
    
    for i in range(start_idx, TOTAL_SAMPLES):
        s = full_sample_list[i]
        
        # Logging
        if (i - start_idx) > 0 and (i - start_idx) % 50 == 0:
            now = time.time()
            elapsed = now - batch_start
            done_in_batch = (i - start_idx)
            avg_time = elapsed / done_in_batch
            remaining = (TOTAL_SAMPLES - i) * avg_time
            print(f"Samples: {i}/{TOTAL_SAMPLES} | Avg: {avg_time:.2f}s | ETA: {remaining:.1f}s | Failures: {failure_count}")

        try:
            # A. Generate
            inputs = tokenizer(s['prompt'], return_tensors="pt").to(DEVICE)
            input_len = inputs.input_ids.shape[1]
            
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=N_CAPTURE,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_hidden_states=False
                )
            
            # B. Forward Pass
            full_ids = outputs.sequences
            with torch.inference_mode():
                 fwd = model(full_ids, output_hidden_states=True, use_cache=False, return_dict=True)
            
            # Extract Hidden States
            hs_layers = torch.stack([h[0] for h in fwd.hidden_states], dim=0) # (L, seq, D)
            resp_hs = hs_layers[:, input_len : input_len + N_CAPTURE, :]
            
            # Padding
            curr_T = resp_hs.shape[1]
            if curr_T < N_CAPTURE:
                pad = torch.zeros((L_LAYERS, N_CAPTURE - curr_T, D_MODEL), dtype=resp_hs.dtype, device=resp_hs.device)
                resp_hs = torch.cat([resp_hs, pad], dim=1)
            elif curr_T > N_CAPTURE:
                resp_hs = resp_hs[:, :N_CAPTURE, :]
                
            # C. Deltas -> Features
            H_t = resp_hs[:, 1:, :]
            H_tm1 = resp_hs[:, :-1, :]
            DH = H_t - H_tm1 
            
            norm = torch.sqrt(torch.sum(DH**2, dim=-1, keepdim=True)) + 1e-8
            DHn = DH / norm
            
            mu = torch.mean(DHn, dim=1)
            var = torch.var(DHn, dim=1)
            
            feat = torch.cat([mu, var], dim=-1) # (L, 1792)
            feat_np = feat.cpu().numpy().astype('float16')
            
            # Write
            fp[i] = feat_np
            
            # Store metadata
            labels_arr[i] = s['operator_id']
            ids_arr[i, 0] = s['content_id']
            ids_arr[i, 1] = s['paraphrase_id']
            
            manifest.append({
                "sample_id": s['sample_id'],
                "operator_id": s['operator_id'],
                "content_id": s['content_id'],
                "ts": time.time()
            })
            
            # Checkpoint
            if (i + 1) % CHECKPOINT_INTERVAL == 0:
                fp.flush()
                np.save(os.path.join(OUTPUT_DIR, LABELS_FILENAME), labels_arr)
                np.save(os.path.join(OUTPUT_DIR, IDS_FILENAME), ids_arr)
                with open(os.path.join(OUTPUT_DIR, MANIFEST_FILENAME), 'w') as f:
                    for line_item in manifest[-(CHECKPOINT_INTERVAL+10):]: # Just write recent? No, write all? 
                        # To avoid massive file I/O on appending, let's just use append mode?
                        pass 
                    # Actually, simplistic approach: Rewrite all manifest? 2000 lines is small.
                with open(os.path.join(OUTPUT_DIR, MANIFEST_FILENAME), 'w') as f:
                    for item in manifest:
                        f.write(json.dumps(item) + "\n")
                        
                with open(progress_path, 'w') as f:
                    json.dump({"completed_samples": i + 1}, f)
                    
        except Exception as e:
            print(f"Error at sample {i}: {e}")
            failure_count += 1
            
    # Final cleanup
    fp.flush()
    np.save(os.path.join(OUTPUT_DIR, LABELS_FILENAME), labels_arr)
    np.save(os.path.join(OUTPUT_DIR, IDS_FILENAME), ids_arr)
    with open(os.path.join(OUTPUT_DIR, MANIFEST_FILENAME), 'w') as f:
        for item in manifest:
             f.write(json.dumps(item) + "\n")
    with open(progress_path, 'w') as f:
        json.dump({"completed_samples": TOTAL_SAMPLES}, f)

    # Conversion/Final checks
    print(f"\n--- Run Complete ---")
    print(f"Total Time: {time.time() - start_time:.1f}s")
    print(f"Features Shape: {fp.shape}")
    print(f"Labels Balance: {np.bincount(labels_arr)}")
    
    # Disk Usage
    feat_size = os.path.getsize(feat_path) / (1024*1024*1024)
    print(f"Features File Size: {feat_size:.2f} GB")

if __name__ == "__main__":
    main()
