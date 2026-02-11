import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
import random

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
OUTPUT_DIR = r"experiments/Experiment 8/data_prime" # Separate folder for Prime data
OUTPUT_FILENAME = "exp8prime_gen.jsonl"
MAX_SAMPLES = 50

# Reuse Dataset Definitions
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

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print("Loading Model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    
    # Construct Prompts
    all_prompts = []
    global_idx = 0
    for op_name in OPERATORS:
        paras = PARAPHRASES.get(op_name, [""]*10)
        for p_id_iter in range(N_PARAS):
            instruction = paras[p_id_iter % len(paras)]
            for content in CONTENTS:
                # Add CoT trigger
                base_prompt = f"CONTENT:\n{content}\n\nTASK:\n{instruction}"
                cot_prompt = base_prompt + "\n\nLet's think step by step."
                all_prompts.append({
                    "original_id": global_idx,
                    "base_prompt": base_prompt,
                    "prompt": cot_prompt
                })
                global_idx += 1
                
    # Shuffle and Select
    random.shuffle(all_prompts)
    selected_prompts = all_prompts[:MAX_SAMPLES]
    
    print(f"Generating {len(selected_prompts)} samples...")
    
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    
    with open(out_path, 'w') as f:
        for i, item in enumerate(selected_prompts):
            print(f"Generating {i+1}/{len(selected_prompts)}...", end='\r')
            
            prompt = item['prompt']
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
                
            gen_text = tokenizer.decode(gen_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # Save
            record = {
                "original_id": item['original_id'],
                "prompt": prompt, # The CoT prompt used
                "gen_text": gen_text,
                "full_text": prompt + gen_text # Simple concat
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            
    print(f"\nGeneration complete. Saved to {out_path}")

if __name__ == "__main__":
    main()
