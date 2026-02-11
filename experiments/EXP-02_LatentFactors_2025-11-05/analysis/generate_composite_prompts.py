import json

# Configuration
BASE_CONTENT = "Artificial intelligence is transforming science, medicine, and creativity."

# Single Operators (Controls)
SINGLE_TEMPLATES = {
    "Summarize": "Summarize the following text in one sentence: {text}",
    "Criticize": "Critically analyze the following statement, pointing out potential flaws: {text}",
    "Reframe Positively": "Reframe the following statement to emphasize its optimistic potential: {text}",
    "Reframe Negatively": "Reframe the following statement to emphasize its dystopian risks: {text}",
    "Poetic": "Write a short poem inspired by the essence of this statement: {text}",
    "Explain Technically": "Explain the technical implications of this statement for a computer scientist: {text}",
    "Ethical Risks": "Evaluate the ethical risks implied by this statement: {text}",
    "Opposite Stance": "Argue persuasively for the exact opposite of this statement: {text}",
    "Ask Question": "Ask a provocative question inspired by this statement: {text}"
}

# Composite Operators
COMPOSITE_TEMPLATES = {
    "Summarize + Criticize": "Summarize the following text in one sentence, but also critically analyze its flaws: {text}",
    "Reframe Negatively + Summarize": "Reframe this statement to emphasize dystopian risks, then summarize the main threat in one sentence: {text}",
    "Poetic + Criticize": "Write a critical analysis of this statement in the form of a short poem: {text}",
    "Explain Technically + Ethical Risks": "Explain the technical implications of this statement while evaluating its ethical risks: {text}",
    "Opposite Stance + Ask Question": "Argue for the opposite of this statement and then ask a provocative question about your new stance: {text}"
}

OUTPUT_FILE = "composite_prompts.json"

def main():
    prompts = []
    
    # Add Singles
    for op_name, template in SINGLE_TEMPLATES.items():
        prompts.append({
            "operator": op_name,
            "type": "single",
            "full_prompt": template.format(text=BASE_CONTENT)
        })
        
    # Add Composites
    for op_name, template in COMPOSITE_TEMPLATES.items():
        prompts.append({
            "operator": op_name,
            "type": "composite",
            "full_prompt": template.format(text=BASE_CONTENT)
        })
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(prompts, f, indent=2)
    
    print(f"Generated {len(prompts)} prompts (Singles & Composites) in {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
