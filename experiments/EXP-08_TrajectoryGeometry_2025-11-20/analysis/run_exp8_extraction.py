import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import time
import numpy as np
import json
import re
import random

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
OUTPUT_DIR = r"experiments/Experiment 8/data"
TRAJ_FILENAME = "exp8_trajectories_v4.npy"
CONTEXT_FILENAME = "exp8_context_v4.npy"
METADATA_FILENAME = "exp8_metadata_v4.jsonl"
PROGRESS_FILENAME = "exp8_progress_v4.json"

N_OPS = 10
N_CONTENTS = 20
N_PARAS = 10
TOTAL_SAMPLES = N_OPS * N_CONTENTS * N_PARAS
L_LAYERS = 25 
D_MODEL = 896
WINDOW_SIZE = 16
N_GENERATE = 48
FEATURE_DIM = D_MODEL + 5 

BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_INTERVAL = 32 

# --- Dataset Definitions ---
OPERATORS = [
    "Summarize", "Critique", "Reframe", "Decompose", "Compare",
    "Extend", "Translate", "Exemplify", "Formalize", "Question"
]
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
PARAPHRASES = {
    "Summarize": ["Summarize this.", "Give a summary.", "Shorten this.", "Briefly summarize.", "Provide a summary.", "Make a summary.", "Write a summary.", "Sum it up.", "Condense this.", "Give the gist."],
    "Critique": ["Critique this.", "Give a critique.", "Evaluate this.", "Find flaws.", "Analyze this.", "Review this.", "Criticize this.", "Assess this.", "Judge this.", "Give feedback."],
    "Reframe": ["Reframe this.", "Rewrite this.", "Paraphrase this.", "Reword this.", "Put in own words.", "Say it differently.", "Restate this.", "Express differently.", "Change wording.", "Spin this."],
    "Decompose": ["Decompose this.", "Break it down.", "Analyze parts.", "Split this.", "Dissect this.", "Show components.", "List parts.", "Separate elements.", "Structure this.", "Divide this."],
    "Compare": ["Compare this.", "Find similarities.", "Contrast this.", "Make comparison.", "Relate this.", "Show differences.", "Draw parallels.", "Match this.", "Juxtapose.", "Weigh this."],
    "Extend": ["Extend this.", "Continue this.", "Add more.", "Expand this.", "Lengthen this.", "Elaborate.", "Develop this.", "Go further.", "Write more.", "Complete this."],
    "Translate": ["Translate this.", "Put in Spanish.", "To French.", "To German.", "Convert language.", "Interpret this.", "Change language.", "To Italian.", "To Chinese.", "To Russian."],
    "Exemplify": ["Exemplify this.", "Give examples.", "Show instance.", "Illustrate.", "Demonstrate.", "Cite case.", "Provide instance.", "Make example.", "Show case.", "Detail example."],
    "Formalize": ["Formalize this.", "Make formal.", "More academic.", "Standardize.", "Official tone.", "Strict format.", "Structuring.", "Codify.", "Make professional.", "Elevate tone."],
    "Question": ["Question this.", "Ask about this.", "Interrogate.", "Probe this.", "Make inquiry.", "Query this.", "Challenge.", "Doubt this.", "Investigate.", "Ask why."]
} 

REASONING_MARKERS = [
    r"because", r"therefore", r"so", r"thus", r"hence", r"consequently",
    r"implies", r"means", r"reason", r"since", r"due to", 
    r"let's think", r"wait a", r"first", r"key issue", r"consider", 
    r"analyze", r"examine", r"look at", r"assume", r"suppose", 
    r"arguably", r"however", r"but", r"although", r"contrast",
    r"specifically", r"example", r"instance"
]
REGEX_PATTERN = "|".join(REASONING_MARKERS)

def compute_trajectory_descriptors(hidden_window):
    # hidden_window: (W, D)
    if hidden_window.shape[0] < 2:
        return torch.zeros((FEATURE_DIM,), device=hidden_window.device)

    # 1. Displacement Vectors
    h_t = hidden_window[1:, :]
    h_tm1 = hidden_window[:-1, :]
    delta_h = h_t - h_tm1 # (W-1, D)
    
    # A. Mean Displacement (D)
    mu_disp = torch.mean(delta_h, dim=0)
    
    # Norms
    norms = torch.norm(delta_h, dim=1) # (W-1,)
    
    # B/C. Speed Mean/Var (Scalars)
    speed_mean = torch.mean(norms).unsqueeze(0)
    speed_var = torch.var(norms).unsqueeze(0)
    
    # D. Curvature (Mean Cosine)
    if delta_h.shape[0] > 1:
        v1 = delta_h[:-1, :]
        v2 = delta_h[1:, :]
        v1_n = torch.nn.functional.normalize(v1, p=2, dim=1)
        v2_n = torch.nn.functional.normalize(v2, p=2, dim=1)
        cos_sim = torch.sum(v1_n * v2_n, dim=1)
        curvature = torch.mean(cos_sim).unsqueeze(0)
        dir_var = torch.mean(torch.var(v1_n, dim=0)).unsqueeze(0) 
    else:
        curvature = torch.tensor([1.0], device=hidden_window.device)
        dir_var = torch.tensor([0.0], device=hidden_window.device)

    # F. Stability (Inverse representational variance)
    rep_var = torch.mean(torch.var(hidden_window, dim=0))
    stability = (1.0 / (rep_var + 1e-6)).unsqueeze(0)
    
    return torch.cat([mu_disp, speed_mean, speed_var, curvature, dir_var, stability], dim=0)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Init Models
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map=DEVICE)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Prepare MMap
    feat_path = os.path.join(OUTPUT_DIR, TRAJ_FILENAME)
    ctx_path = os.path.join(OUTPUT_DIR, CONTEXT_FILENAME)
    D_TOTAL = D_MODEL + 5 
    
    if os.path.exists(feat_path):
        fp = np.memmap(feat_path, dtype='float32', mode='r+', shape=(TOTAL_SAMPLES, L_LAYERS, D_TOTAL))
        fp_ctx = np.memmap(ctx_path, dtype='float32', mode='r+', shape=(TOTAL_SAMPLES, D_MODEL))
        start_idx = 0
        if os.path.exists(os.path.join(OUTPUT_DIR, PROGRESS_FILENAME)):
            with open(os.path.join(OUTPUT_DIR, PROGRESS_FILENAME), 'r') as f:
                start_idx = json.load(f).get('completed_samples', 0)
    else:
        fp = np.memmap(feat_path, dtype='float32', mode='w+', shape=(TOTAL_SAMPLES, L_LAYERS, D_TOTAL))
        fp_ctx = np.memmap(ctx_path, dtype='float32', mode='w+', shape=(TOTAL_SAMPLES, D_MODEL))
        start_idx = 0
        
    print(f"Starting Exp 8 Data Extraction V3 (Randomized, Batch Size {BATCH_SIZE})...")
    
    meta_file = open(os.path.join(OUTPUT_DIR, METADATA_FILENAME), 'a' if start_idx > 0 else 'w')
    
    # Construct Samples List
    all_samples = []
    global_idx = 0
    for op_id, op_name in enumerate(OPERATORS):
        paras = PARAPHRASES.get(op_name, [""]*10)
        for p_id_iter in range(N_PARAS):
            instruction = paras[p_id_iter % len(paras)]
            for c_id, content in enumerate(CONTENTS):
                prompt = f"CONTENT:\n{content}\n\nTASK:\n{instruction}"
                all_samples.append({
                    "id": global_idx,
                    "prompt": prompt,
                    "operator": op_name,
                    "content_id": c_id,
                    "paraphrase_id": p_id_iter
                })
                global_idx += 1
    
    # SHUFFLE
    random.seed(42)
    random.shuffle(all_samples)
    
    # Reset IDs in metadata or keep original?
    # Keep original "id" to trace back, but write sequential index to metadata?
    # `fp[i]` corresponds to `i`-th iteration.
    # So `metadata` line `i` corresponds to `fp[i]`.
    # The `meta['id']` field will reflect the shuffled original ID.
                
    # Run Batches
    for i in range(start_idx, TOTAL_SAMPLES, BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, TOTAL_SAMPLES)
        # Sliced from shuffled list
        batch_samples = all_samples[i:batch_end]
        batch_prompts = [s['prompt'] for s in batch_samples]
        
        try:
            # 1. Tokenize
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(DEVICE)
            input_len = inputs.input_ids.shape[1]
            actual_batch_size = len(batch_prompts)
            
            # 2. Generate
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=N_GENERATE,
                    do_sample=False, 
                    top_k=40,
                    return_dict_in_generate=True,
                    output_hidden_states=False
                )
            
            full_ids = outputs.sequences # (B, Seq)
            
            # 3. Identify Windows
            windows = []
            cue_found_flags = []
            gen_texts = []
            
            new_ids = full_ids[:, input_len:] # (B, NewTokens)
            
            for b_idx in range(actual_batch_size):
                # Decode
                g_text = tokenizer.decode(new_ids[b_idx], skip_special_tokens=True)
                gen_texts.append(g_text)
                
                match = re.search(REGEX_PATTERN, g_text, re.IGNORECASE)
                cue_found_flags.append(bool(match))
                
                target_center = 8
                if match:
                    start_char = match.start()
                    pre_ids = tokenizer.encode(g_text[:start_char], add_special_tokens=False)
                    match_idx = len(pre_ids)
                    target_center = match_idx
                
                w_start = max(0, target_center - WINDOW_SIZE // 2)
                w_end = w_start + WINDOW_SIZE
                
                valid_len = (new_ids[b_idx] != tokenizer.pad_token_id).sum().item()
                if w_end > valid_len:
                    w_end = valid_len
                    w_start = max(0, w_end - WINDOW_SIZE)
                
                windows.append((w_start, w_end))

            # 4. Forward Pass to get Hidden States
            with torch.inference_mode():
                fwd = model(full_ids, output_hidden_states=True, use_cache=False, return_dict=True)
                
            hs_stack = torch.stack([h for h in fwd.hidden_states], dim=0).detach()
            
            # 5. Extract Details
            batch_traj = []
            batch_ctx = []
            
            for b_idx in range(actual_batch_size):
                w_s, w_e = windows[b_idx]
                abs_s = input_len + w_s
                abs_e = input_len + w_e
                
                # Window HS
                h_window = hs_stack[:, b_idx, abs_s:abs_e, :]
                
                # Context HS
                if abs_s > 0:
                    ctx_h = hs_stack[-1, b_idx, :abs_s, :]
                    ctx_vec = torch.mean(ctx_h, dim=0).cpu().numpy()
                else:
                    ctx_vec = np.zeros((D_MODEL,), dtype=np.float32)
                
                batch_ctx.append(ctx_vec)
                
                # Trajectory descriptors
                l_traj = []
                for l in range(L_LAYERS):
                    desc = compute_trajectory_descriptors(h_window[l])
                    l_traj.append(desc.cpu().numpy())
                
                batch_traj.append(np.stack(l_traj))

            # 6. Save Batch
            batch_traj_np = np.stack(batch_traj) # (B, L, D_TOT)
            batch_ctx_np = np.stack(batch_ctx)   # (B, D)
            
            fp[i:i+actual_batch_size] = batch_traj_np
            fp_ctx[i:i+actual_batch_size] = batch_ctx_np
            fp.flush()
            fp_ctx.flush()
            
            # Metadata
            for b_idx in range(actual_batch_size):
                 meta = {
                    "iter_idx": i + b_idx,
                    "original_id": batch_samples[b_idx]['id'],
                    "operator": batch_samples[b_idx]['operator'],
                    "gen_text": gen_texts[b_idx],
                    "cue_found": cue_found_flags[b_idx],
                }
                 meta_file.write(json.dumps(meta) + "\n")
            
            # Progress
            if (i // BATCH_SIZE) % 1 == 0:
                print(f"Processed batch {i}-{i+actual_batch_size}/{TOTAL_SAMPLES}")
                with open(os.path.join(OUTPUT_DIR, PROGRESS_FILENAME), 'w') as f:
                    json.dump({"completed_samples": i+actual_batch_size}, f)
                    
        except Exception as e:
            print(f"Error in batch {i}: {e}")
            import traceback
            traceback.print_exc()
            
    meta_file.close()
    print("Execution Complete.")

if __name__ == "__main__":
    main()
