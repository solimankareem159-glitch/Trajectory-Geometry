import json

# Configuration
BASE_CONTENT = "Artificial intelligence is transforming science, medicine, and creativity."

# Operator Templates
OPERATORS = {
    "Summarize": "Summarize the following text in one sentence: {text}",
    "Criticize": "Critically analyze the following statement, pointing out potential flaws: {text}",
    "Reframe Positively": "Reframe the following statement to emphasize its optimistic potential: {text}",
    "Reframe Negatively": "Reframe the following statement to emphasize its dystopian risks: {text}",
    "Ask Question": "Ask a provocative question inspired by this statement: {text}",
    "Explain Simply": "Explain the core idea of this statement to a five-year-old: {text}",
    "Explain Technically": "Explain the technical implications of this statement for a computer scientist: {text}",
    "Opposite Stance": "Argue persuasively for the exact opposite of this statement: {text}",
    "Poetic": "Write a short poem inspired by the essence of this statement: {text}",
    "Ethical Risks": "Evaluate the ethical risks implied by this statement: {text}"
}

OUTPUT_FILE = "operator_prompts.json"

def main():
    prompts = []
    
    for op_name, template in OPERATORS.items():
        prompt_text = template.format(text=BASE_CONTENT)
        prompts.append({
            "operator": op_name,
            "base_content": BASE_CONTENT,
            "full_prompt": prompt_text
        })
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(prompts, f, indent=2)
    
    print(f"Generated {len(prompts)} prompts in {OUTPUT_FILE}")
    for p in prompts:
        print(f"[{p['operator']}]: {p['full_prompt']}")

if __name__ == "__main__":
    main()
