import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
import numpy as np
import re
import random
from scipy import stats
import os
print(f"PID: {os.getpid()}", flush=True)

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATA_DIR = r"experiments/Experiment 8/data_prime"
METADATA_FILENAME = "exp8prime_gen.jsonl"
OUTPUT_DIR = r"experiments/Experiment 8/results/analysis"
REPORT_FILENAME = "exp8prime_transition_report.md"
SUMMARY_FILENAME = "exp8prime_summary.json"
TABLE_FILENAME = "exp8prime_effect_table.csv"

# Reuse Dataset Definitions from Extraction Script for Prompt Reconstruction
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
N_PARAS = 10

# Cues
CUE_FAMILIES = {
    "Uncertainty": ["wait", "hold on", "I'm not sure", "that doesn't seem"],
    "Planning": ["let's", "first", "step by step", "we can break this down"],
    "Perspective": ["however", "on the other hand", "from another angle"],
    "Resolution": ["therefore", "so", "in summary", "this means that"]
}
ALL_CUES_FLAT = [c.lower() for fam in CUE_FAMILIES.values() for c in fam]

def get_original_prompt(original_id):
    # Reconstruct the exact list (order matters)
    all_samples = []
    global_idx = 0
    for op_name in OPERATORS:
        paras = PARAPHRASES.get(op_name, [""]*10)
        for p_id_iter in range(N_PARAS):
            instruction = paras[p_id_iter % len(paras)]
            for content in CONTENTS:
                prompt = f"CONTENT:\n{content}\n\nTASK:\n{instruction}"
                all_samples.append({
                    "id": global_idx,
                    "prompt": prompt
                })
                global_idx += 1
    
    # Simple lookup (since we know the id is just the index)
    # But wait, original global_idx matches list index
    if 0 <= original_id < len(all_samples):
        return all_samples[original_id]['prompt']
    return None

def find_cue(text):
    text_lower = text.lower()
    for fam, cues in CUE_FAMILIES.items():
        for cue in cues:
            # Case insensitive exact match or word boundary?
            # User said "Detect first occurrence... case-insensitive".
            # Simple substring search or regex?
            # "wait" matches "waiting"?
            # User listed phrases. Assuming word boundaries for short ones?
            # "so" matches "also"? Yes if simple substring.
            # I should use word boundaries for short words.
            pattern = r"\b" + re.escape(cue) + r"\b"
            if len(cue.split()) > 1:
                pattern = re.escape(cue) # Phrases usually safe
            
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return fam, cue, match.end() # Return end or start? "Cue token index". 
                # Model predicts AFTER cue? Or we look AT cue?
                # User: "Let cue token index = t0. Extract [t0-8, ... t0+8]"
                # Usually t0 is the token *of* the cue.
                # I'll return start index to find token.
                return fam, cue, match.start()
    return None, None, None

def compute_metrics(h_window):
    # h_window: (17, D) (since 8 before, 8 after, + 1 center = 17)
    # Split at t0 (index 8).
    # "before vs after (split at t0)"
    # Before: 0..7? After: 8..16?
    # Or Before: 0..8 (inclusive of t0?), After: 8..16?
    # Delta = After - Before.
    # User: "mean ||dh|| before vs after"
    # I'll define Before = h[:8], After = h[9:] (excluding center to be clean? or center is t0)
    # Let's say t0 is 8.
    # Before: indices 0 to 7.
    # After: indices 9 to 16.
    # Center (t0) is the transition point.
    
    # 1. Speed (Mean Displacement)
    diffs = h_window[1:] - h_window[:-1] # (16, D)
    norms = np.linalg.norm(diffs, axis=1) # (16,)
    # Diffs indices: 0 maps to h[1]-h[0].
    # Split diffs:
    # Before diffs: 0..7 (up to h[8]-h[7])?
    # t0 is index 8.
    # diffs[7] is h[8]-h[7]. This is the step INTO t0.
    # diffs[8] is h[9]-h[8]. This is step OUT of t0.
    # Let's average steps leading up to t0 vs steps leading away.
    speed_before = np.mean(norms[:8])
    speed_after = np.mean(norms[8:])
    
    # 2. Curvature (Cosine)
    curvs = []
    if len(diffs) > 1:
        for i in range(len(diffs)-1):
            v1 = diffs[i]
            v2 = diffs[i+1]
            c = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-9)
            curvs.append(c)
        curvs = np.array(curvs)
        # curvs[i] is angle between diff[i] and diff[i+1].
        # diff[7] is into t0. diff[8] is out of t0.
        # curvs[7] is angle at t0.
        # Before: 0..6. After: 8..?
        # Let's just split in half.
        mid = len(curvs)//2
        curv_before = np.mean(curvs[:mid])
        curv_after = np.mean(curvs[mid:])
    else:
        curv_before = 0
        curv_after = 0

    # 3. Stability (Inverse Var)
    # Var of raw states
    var_before = np.mean(np.var(h_window[:8], axis=0))
    var_after = np.mean(np.var(h_window[9:], axis=0))
    stab_before = 1.0 / (var_before + 1e-6)
    stab_after = 1.0 / (var_after + 1e-6)
    
    # 4. Dir Var
    # Variance of normalized diffs
    # Normalize diffs
    diffs_norm = diffs / (norms[:, None] + 1e-9)
    dv_before = np.mean(np.var(diffs_norm[:8], axis=0))
    dv_after = np.mean(np.var(diffs_norm[8:], axis=0))
    
    return {
        "speed": speed_after - speed_before,
        "curvature": curv_after - curv_before,
        "stability": stab_after - stab_before,
        "dir_var": dv_after - dv_before
    }

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("Loading Model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    
    print("Loading Metadata...")
    meta_path = os.path.join(DATA_DIR, METADATA_FILENAME)
    # Safe copy to avoid lock issues
    import shutil
    import uuid
    meta_temp_name = f"{METADATA_FILENAME}.{uuid.uuid4()}.temp"
    meta_path_temp = os.path.join(DATA_DIR, meta_temp_name)
    print(f"Copying to {meta_path_temp}...", flush=True)
    try:
        shutil.copy(meta_path, meta_path_temp)
    except Exception as e:
        print(f"Warning: Could not copy metadata file ({e}). Trying direct read...")
        meta_path_temp = meta_path

    samples = []
    with open(meta_path_temp, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    samples.append(json.loads(line))
                except: pass
    
    # Cleanup temp
    if meta_path_temp != meta_path and os.path.exists(meta_path_temp):
        try:
            os.remove(meta_path_temp)
        except: pass
                
    print(f"Loaded {len(samples)} samples. Processing...")
    
    results = {fam: {l: {"speed": [], "curvature": [], "stability": [], "dir_var": []} for l in [0, 13, 24]} for fam in CUE_FAMILIES}
    control_results = {fam: {l: {"speed": [], "curvature": [], "stability": [], "dir_var": []} for l in [0, 13, 24]} for fam in CUE_FAMILIES}
    
    count = 0
    MAX_SAMPLES = 500
    
    for s in samples:
        if count >= MAX_SAMPLES: break
        
        oid = s.get("original_id")
        # Debug logging
        debug = (count == 0 and random.random() < 0.1) # low prob for random sampling debug? No, just print first few skips.
        # Actually let's print detailed reason for first 50 iterations
        iter_debug = (s.get("iter_idx", 0) < 20)
        
        orig_id = s.get("original_id")
        # For Prime data, 'prompt' is already in the record
        prompt = s.get("prompt")
        gen_text = s.get("gen_text", "")
        
        if not prompt: 
            if iter_debug: print(f"Skip {oid}: Prompt missing in data")
            continue
            
        # full_text for Prime is stored or can be reconstructed
        # The gen script stored 'full_text' key.
        full_text = s.get("full_text")
        if not full_text:
             full_text = prompt + gen_text 
        
        # 1. Find Cue
        fam, cue, char_idx = find_cue(gen_text) 
        if not fam: 
            if iter_debug: print(f"Skip {orig_id}: No cue found")
            continue
        
        # 2. Tokenize & align
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        prefix_ids = tokenizer.encode(gen_text[:char_idx], add_special_tokens=False)
        
        t0 = len(prompt_ids) + len(prefix_ids)
        
        # 3. Control Selection
        full_ids = tokenizer.encode(full_text, return_tensors="pt").to(model.device)
        total_len = full_ids.shape[1]
        
        valid_indices = []
        gen_start = len(prompt_ids)
        for i in range(gen_start, total_len - 8): 
            if abs(i - t0) > 20: 
                valid_indices.append(i)
                
        if not valid_indices: 
            if iter_debug: print(f"Skip {orig_id}: No valid control (t0={t0}, len={total_len})")
            continue

        tc = random.choice(valid_indices)
        
        if iter_debug: print(f"Processing {orig_id}: Cue={cue} at {t0}, Control at {tc}")
        
        # Check window bounds
        if t0 - 8 < 0 or t0 + 8 >= total_len: continue
        if tc - 8 < 0 or tc + 8 >= total_len: continue
        
        # 4. Run Model
        with torch.no_grad():
            out = model(full_ids, output_hidden_states=True)
            
        # 5. Extract & Compute
        layers = [0, 13, 24]
        for l in layers:
            h = out.hidden_states[l][0].float().cpu().numpy() # (Seq, D)
            
            # Cue Window
            h_cue = h[t0-8 : t0+9] # 17 tokens
            m_cue = compute_metrics(h_cue)
            
            # Control Window
            h_ctrl = h[tc-8 : tc+9]
            m_ctrl = compute_metrics(h_ctrl)
            
            for k in m_cue:
                results[fam][l][k].append(m_cue[k])
                control_results[fam][l][k].append(m_ctrl[k])
                
        count += 1
        if count % 10 == 0:
            print(f"Processed {count} samples...", end='\r')
            
    print(f"\nProcessing complete. N={count}")
    
    # 6. Statistical Analysis (Permutation Test)
    report = "# Experiment 8': Transition Test Report\n\n"
    report += "| Family | Layer | Metric | Mean Delta (Cue) | Mean Delta (Ctrl) | Cohen's d | p-value | Result |\n"
    report += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    
    final_summary = {}
    
    for fam in CUE_FAMILIES:
        for l in [0, 13, 24]:
            for metric in ["speed", "curvature", "stability", "dir_var"]:
                vals_cue = np.array(results[fam][l][metric])
                vals_ctrl = np.array(control_results[fam][l][metric])
                
                if len(vals_cue) < 5: continue
                
                mean_cue = np.mean(vals_cue)
                mean_ctrl = np.mean(vals_ctrl)
                
                # Paired Permutation Test
                diff = vals_cue - vals_ctrl
                obs_mean = np.mean(diff)
                
                n_perms = 10000
                signs = np.random.choice([-1, 1], size=(n_perms, len(diff)))
                perm_means = np.mean(diff * signs, axis=1)
                p_val = np.mean(np.abs(perm_means) >= np.abs(obs_mean))
                
                # Cohen's d
                # d = mean(diff) / std(diff)
                d = obs_mean / (np.std(diff, ddof=1) + 1e-9)
                
                res = "PASS" if (p_val < 0.05 and abs(d) > 0.5 and l in [13]) else "FAIL"
                # Strict check: if l=0 has effect, then l=13 is not specific.
                
                row = f"| {fam} | {l} | {metric} | {mean_cue:.4f} | {mean_ctrl:.4f} | {d:.3f} | {p_val:.4f} | {res} |\n"
                report += row
                
                final_summary[f"{fam}_L{l}_{metric}"] = {"d": d, "p": p_val, "obs": obs_mean}

    # Conclusion
    # Check for any PASS in L13 that is NOT matched by L0
    report += "\n## Conclusion\n"
    # Logic...
    
    with open(os.path.join(OUTPUT_DIR, REPORT_FILENAME), 'w') as f:
        f.write(report)
        
    print(f"Report saved to {REPORT_FILENAME}")

if __name__ == "__main__":
    main()
