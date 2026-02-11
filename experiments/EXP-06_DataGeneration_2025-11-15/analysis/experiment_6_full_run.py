import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import time
import numpy as np
import json
import psutil

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
OUTPUT_DIR = "experiments/experiment 6/data/full_2k"
FEATURES_FILENAME = "features.memmap"
LABELS_FILENAME = "labels.npy"
IDS_FILENAME = "ids.npy"
MANIFEST_FILENAME = "manifest.jsonl"
PROGRESS_FILENAME = "progress.json"

N_OPS = 10
N_CONTENTS = 20
N_PARAS = 10
TOTAL_SAMPLES = N_OPS * N_CONTENTS * N_PARAS
N_CAPTURE = 32
L_LAYERS = 25 # Qwen 0.5B has 24 layers + 1 embedding = 25
D_MODEL = 896
FEATURE_DIM = D_MODEL * 2 # Mean + Var = 1792

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_INTERVAL = 50

# --- Dataset Definitions ---

# 10 Operators
OPERATORS = [
    "Summarize", "Critique", "Reframe", "Decompose", "Compare",
    "Extend", "Translate", "Exemplify", "Formalize", "Question"
]

# 10 Paraphrases per Operator (Total 100)
PARAPHRASES = {
    "Summarize": [
        "Summarize the main point of this text.", "Give me the TL;DR of the passage below.", "Provide a concise summary of the content.",
        "Condense this text into its key takeaway.", "Briefly explain what this passage is about.", "Write a short summary of the following.",
        "What is the core message of this text?", "Reduce this text to its essential meaning.", "Give a brief overview of the main ideas.", "Synthesize the provided text into a summary."
    ],
    "Critique": [
        "Critique the logic or style of this text.", "Identify potential flaws, risks, or weaknesses.", "Play devil's advocate: what is wrong with this?",
        "Analyze the shortcomings of this argument.", "What are the limitations of this perspective?", "Evaluate the validity of the claims made.",
        "Find logical fallacies or gaps in reasoning.", "Assess the quality and reliability of this text.", "Offer a critical review of the content.", "Highlight any bias or inconsistency here."
    ],
    "Reframe": [
        "Reframe this text to be more optimistic.", "Rewrite this from a different perspective.", "Spin this information in a positive light.",
        "Change the tone of this text to be more formal.", "Reformulate this as if speaking to a child.", "Adapt this text for a skeptical audience.",
        "Rewrite this passage to be more persuasive.", "Shift the focus of this text to the future implications.", "Recast this statement as a question.", "Express this same idea using different words."
    ],
    "Decompose": [
        "Break this text down into a numbered list.", "Decompose this argument into its premises.", "List the key components mentioned here.",
        "Extract the individual steps or points.", "Separate the main idea from the supporting details.", "Outline the structure of this passage.",
        "Identify the distinct elements in this text.", "Unpack the complex ideas into simpler parts.", "Dissect this text into its constituent claims.", "Parse this content into a structured format."
    ],
    "Compare": [
        "Compare the ideas here with common knowledge.", "Contrast the views presented with an opposing view.", "How does this text relate to historical context?",
        "Identify similarities between this and other concepts.", "Draw a comparison metric for this content.", "Relate this text to a broader theme.",
        "What distinguishes this perspective from others?", "Juxtapose this argument with a counter-argument.", "Find an analogy that fits this situation.", "Evaluate this text against standard criteria."
    ],
    "Extend": [
        "Extend this text with a follow-up thought.", "Write a continuation of this passage.", "Elaborate on the implications of this.",
        "Add a concluding paragraph to this text.", "Expand on the ideas validation here.", "What would happen next based on this?",
        "Propose a next step derived from this.", "Flesh out the details mentioned briefly.", "Project the future consequences of this.", "Develop this idea further."
    ],
    "Translate": [
        "Translate this text into Spanish.", "Convert this text into French.", "Render this passage in German.",
        "Translate the core meaning into simple English.", "Translate this into a formal academic style.", "Convert this text into Python code comments.",
        "Translate this concept into a mathematical notation.", "Express this text in the style of Shakespeare.", "Translate this into a series of emojis.", "Convert this text into a JSON object."
    ],
    "Exemplify": [
        "Give an example that illustrates this point.", "Provide a concrete use case for this.", "Generate a scenario where this applies.",
        "Show how this works in practice with an example.", "Illustrate this concept with a story.", "Create a hypothetical situation based on this.",
        "Name a real-world entity that fits this description.", "Demonstrate this principle with a case study.", "Provide a counter-example to this claim.", "List specific instances of this phenomenon."
    ],
    "Formalize": [
        "Rewrite this text using formal logic.", "Convert this explanation into a mathematical proof.", "Standardize the terminology used here.",
        "Express this argument as a syllogism.", "Format this text as a legal clause.", "restructure this into a technical specification.",
        "Define the terms rigorously.", "Make this text unambiguous and precise.", "Codify these rules into a strict format.", "Translate this into a formal policy statement."
    ],
    "Question": [
        "Generate a question that this text answers.", "Ask a follow-up question based on this.", "What is the most critical question to ask here?",
        "Formulate a research question derived from this.", "Quiz me on the content of this passage.", "What doubt does this text raise?",
        "Interrogate the assumptions underlying this text.", "Pose a challenge to the author's view.", "Ask a rhetorical question inspired by this.", "Create a discussion prompt from this text."
    ]
}

# 20 Contents (Diverse Domains)
CONTENTS = [
    # 1. Tech/Science
    "The James Webb Space Telescope has captured the most detailed images of the early universe to date. Its infrared capabilities allow it to peer through dust clouds that obscured the view of Hubble. Astronomers are now analyzing these images to understand galaxy formation.",
    "Quantum computing utilizes qubits, which can exist in a superposition of states. This allows quantum computers to solve specific problems, like integer factorization, exponentially faster than classical supercomputers. However, maintaining coherence remains a major engineering challenge.",
    "CRISPR gene editing allows for precise alteration of DNA sequences in living organisms. Derived from a bacterial immune system, it uses a guide RNA to target specific genetic codes. Ethical concerns persist regarding its application in human germline editing.",
    "Machine learning models, particularly large language models, rely on vast amounts of training data. They learn statistical patterns to predict the next token in a sequence. Recent advancements have focused on alignment and reducing hallucinations in generated outputs.",
    # 2. History/Society
    "The Industrial Revolution marked a major turning point in history. It shifted production from hand tools to complex machinery, leading to urbanization. While it increased economic output, it also resulted in harsh working conditions and environmental pollution.",
    "The fall of the Roman Empire was a gradual process caused by internal strife and external invasions. Economic instability and political corruption weakened the state. The fracturing of the empire led to the Middle Ages in Europe.",
    "Democracy relies on the active participation of its citizens. Voting rights have expanded significantly over the centuries, though challenges to access remain. Free press and independent judiciary are essential checks on power.",
    "Globalization has interconnected economies and cultures worldwide. It facilitates the flow of goods, services, and information. Critics argue it can exacerbate inequality and erode local traditions.",
    # 3. Nature/Environment
    "Photosynthesis is the process by which green plants use sunlight to synthesize nutrients. Chlorophyll in plant cells captures light energy to convert carbon dioxide and water into glucose. This process releases oxygen as a vital byproduct.",
    "The Amazon rainforest is often referred to as the lungs of the Earth. It hosts an immense biodiversity of flora and fauna. Deforestation poses a severe threat to this ecosystem and global climate stability.",
    "Ocean acidification is a consequence of increased carbon dioxide absorption by seawater. It impacts marine life, particularly organisms with calcium carbonate shells. Coral reefs are increasingly vulnerable to these chemical changes.",
    "Migration patterns of monarch butterflies span thousands of miles. They travel from Canada and the US to Mexico for the winter. Habitat loss and climate change are impacting their population numbers.",
    # 4. Philosophy/Abstract
    "Utilitarianism is an ethical theory that advocates for actions that maximize overall happiness. It assesses the moral worth of an act by its outcome. Critics argue it can justify the suffering of a minority for the majority's benefit.",
    "Existentialism emphasizes individual freedom and responsibility. It posits that existence precedes essence, meaning individuals define their own meaning. Philosophers like Sartre and Camus explored the absurdity of life.",
    "The Ship of Theseus is a thought experiment about identity. If all parts of a ship are replaced over time, is it still the same ship? It questions the nature of persistence and change.",
    "Epistemology is the branch of philosophy concerned with knowledge. It explores the distinction between justified belief and opinion. Skepticism challenges the possibility of certainty in our understanding of the world.",
    # 5. Art/Literature
    "Impressionism was a 19th-century art movement characterized by small, thin brush strokes. Artists sought to capture the changing qualities of light. Monet and Renoir are among its most famous practitioners.",
    "The Hero's Journey is a common narrative template in storytelling. It involves a hero who goes on an adventure, faces a crisis, and returns transformed. This structure is found in myths and modern movies alike.",
    "Minimalism in design focuses on simplicity and functionality. It strips away unnecessary elements to reveal the essential quality of a subject. 'Less is more' is a key mantra of this aesthetic.",
    "Haiku is a form of Japanese poetry consisting of three phrases. The structure typically follows a 5-7-5 syllable count. It often focuses on images from nature or a specific moment in time.",
]


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def main():
    print(f"--- Experiment 6 Full Run (2000 Samples) ---")
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
    # Features: Memmap file
    feat_path = os.path.join(OUTPUT_DIR, FEATURES_FILENAME)
    
    # If starting from 0, create/overwrite
    if start_idx == 0:
        # Create empty memmap
        fp = np.memmap(feat_path, dtype='float16', mode='w+', shape=(TOTAL_SAMPLES, L_LAYERS, FEATURE_DIM))
        # Initial labels/ids arrays (in memory, save periodically)
        labels_arr = np.zeros(TOTAL_SAMPLES, dtype=int)
        ids_arr = np.zeros((TOTAL_SAMPLES, 2), dtype=int) # col0=content, col1=para
        manifest = []
    else:
        # Open existing
        fp = np.memmap(feat_path, dtype='float16', mode='r+', shape=(TOTAL_SAMPLES, L_LAYERS, FEATURE_DIM))
        # Load partial numpy arrays
        labels_arr = np.load(os.path.join(OUTPUT_DIR, LABELS_FILENAME))
        ids_arr = np.load(os.path.join(OUTPUT_DIR, IDS_FILENAME))
        # Load manifest
        with open(os.path.join(OUTPUT_DIR, MANIFEST_FILENAME), 'r') as f:
            manifest = [json.loads(line) for line in f]
            
    # 3. Build Sample List
    # Order: Ops -> Paras -> Contents (or any deterministic order)
    # We need a flat list of 2000 items to index into
    # Let's do Op -> Para -> Content to keep Op contiguous?
    # Actually, user ID spec is flexible, but crossed design.
    # Let's flatten simply.
    
    full_sample_list = []
    global_idx = 0
    for op_id, op_name in enumerate(OPERATORS):
        paras = PARAPHRASES[op_name] 
        # Safety check
        if len(paras) != N_PARAS:
            print(f"Warning: Operator {op_name} has {len(paras)} paraphrases, expected {N_PARAS}")
        
        for p_id, instruction in enumerate(paras):
             if p_id >= N_PARAS: break # Clip if necessary
             
             for c_id, content in enumerate(CONTENTS):
                 prompt = f"CONTENT:\n{content}\n\nTASK:\n{instruction}"
                 
                 full_sample_list.append({
                     "sample_id": global_idx,
                     "operator": op_name,
                     "operator_id": op_id,
                     "content_id": c_id,
                     "paraphrase_id": p_id,
                     "instruction": instruction,
                     "prompt": prompt
                 })
                 global_idx += 1
                 
    assert len(full_sample_list) == TOTAL_SAMPLES
    
    # 4. Load Model
    # Only if we have work to do
    if start_idx < TOTAL_SAMPLES:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            output_hidden_states=True, 
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map=DEVICE
        )
    else:
        print("Dataset already complete!")
        return

    # 5. Execution Loop
    start_time = time.time()
    batch_start = time.time()
    
    for i in range(start_idx, TOTAL_SAMPLES):
        s = full_sample_list[i]
        
        # Logging
        if i % 25 == 0 and i > start_idx:
            now = time.time()
            avg_time = (now - batch_start) / (i - start_idx)
            remaining = (TOTAL_SAMPLES - i) * avg_time
            print(f"Processing {i}/{TOTAL_SAMPLES} | Avg: {avg_time:.2f}s | ETA: {remaining/60:.1f}m | Mem: {psutil.virtual_memory().percent}%")
        
        try:
            # A. Generate (Fast)
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
            
            gen_ids = outputs.sequences[0, input_len:]
            
            # Metadata recording (minimal text)
            # We won't save full text to manifest to keep it small, or optional? 
            # User said "optional; can omit". We'll omit text to save space/time.
            
            # B. Forward Pass (Hidden States)
            full_ids = outputs.sequences
            with torch.inference_mode():
                 fwd = model(full_ids, output_hidden_states=True, use_cache=False, return_dict=True)
            
            # Extract
            # tuple of layers -> (1, seq, dim)
            hs_layers = torch.stack([h[0] for h in fwd.hidden_states], dim=0) # (L, seq, D)
            
            # Slice: input_len : input_len + N_CAPTURE
            resp_hs = hs_layers[:, input_len : input_len + N_CAPTURE, :]
            
            # Padding check
            curr_T = resp_hs.shape[1]
            if curr_T < N_CAPTURE:
                pad = torch.zeros((L_LAYERS, N_CAPTURE - curr_T, D_MODEL), dtype=resp_hs.dtype, device=resp_hs.device)
                resp_hs = torch.cat([resp_hs, pad], dim=1)
            elif curr_T > N_CAPTURE:
                resp_hs = resp_hs[:, :N_CAPTURE, :]
                
            # C. Feature Extraction (Deltas -> Mean/Var)
            # Move to CPU for numpy calc (or keep on GPU if fast enough? GPU is better for large matrix ops)
            # Computation on GPU to save transfer time
            
            H_t = resp_hs[:, 1:, :]
            H_tm1 = resp_hs[:, :-1, :]
            DH = H_t - H_tm1 # (L, 31, D)
            
            # Norm
            norm = torch.sqrt(torch.sum(DH**2, dim=-1, keepdim=True)) + 1e-8
            DHn = DH / norm
            
            mu = torch.mean(DHn, dim=1) # (L, D)
            var = torch.var(DHn, dim=1) # (L, D)
            
            # Concatenate [mu, var] -> (L, 2D)
            feat = torch.cat([mu, var], dim=-1) # (L, 1792)
            
            # To CPU Numpy float16
            feat_np = feat.cpu().numpy().astype('float16')
            
            # Write to memmap
            fp[i] = feat_np
            
            # Store labels/ids
            labels_arr[i] = s['operator_id']
            ids_arr[i, 0] = s['content_id']
            ids_arr[i, 1] = s['paraphrase_id']
            
            # Manifest entry
            manifest.append({
                "sample_id": s['sample_id'],
                "operator_id": s['operator_id'],
                "content_id": s['content_id'],
                "paraphrase_id": s['paraphrase_id'],
                "ts": time.time()
            })
            
            # Checkpoint
            if (i + 1) % CHECKPOINT_INTERVAL == 0:
                # Flush memmap
                fp.flush()
                # Save aux arrays
                np.save(os.path.join(OUTPUT_DIR, LABELS_FILENAME), labels_arr)
                np.save(os.path.join(OUTPUT_DIR, IDS_FILENAME), ids_arr)
                
                # Append/Rewrite manifest
                # To be safe, just write full list or append? 
                # Rewriting is safer for consistency if crash happens mid-write
                with open(os.path.join(OUTPUT_DIR, MANIFEST_FILENAME), 'w') as f:
                    for item in manifest:
                        f.write(json.dumps(item) + "\n")
                        
                # Update progress
                with open(progress_path, 'w') as f:
                    json.dump({"completed_samples": i + 1}, f)
                    
                # print(f"Checkpoint saved at {i+1}")
                
        except Exception as e:
            print(f"Error at sample {i}: {e}")
            import traceback
            traceback.print_exc()
            # Try to continue?
            
    # Final cleanup
    fp.flush()
    np.save(os.path.join(OUTPUT_DIR, LABELS_FILENAME), labels_arr)
    np.save(os.path.join(OUTPUT_DIR, IDS_FILENAME), ids_arr)
    with open(os.path.join(OUTPUT_DIR, MANIFEST_FILENAME), 'w') as f:
        for item in manifest:
             f.write(json.dumps(item) + "\n")
             
    with open(progress_path, 'w') as f:
        json.dump({"completed_samples": TOTAL_SAMPLES}, f)

    total_time = time.time() - start_time
    print(f"\n--- Run Complete ---")
    print(f"Total Samples: {TOTAL_SAMPLES}")
    print(f"Total Time: {total_time:.1f}s")
    print(f"Features Shape: {fp.shape} (memmap)")
    
    # Disk Usage
    feat_size = os.path.getsize(feat_path) / (1024*1024*1024)
    print(f"Features File Size: {feat_size:.2f} GB")

if __name__ == "__main__":
    main()
