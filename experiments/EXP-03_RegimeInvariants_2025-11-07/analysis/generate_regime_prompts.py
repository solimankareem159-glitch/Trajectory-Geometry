import json
import random

OPERATORS = [
    "Summarize", "Criticize", "Reframe Positively", "Reframe Negatively", 
    "Ask Question", "Explain Simply", "Explain Technically", 
    "Opposite Stance", "Poetic", "Ethical Risks"
]

TOPICS = {
    "AI": "Artificial intelligence is transforming science, medicine, and creativity.",
    "Climate": "Renewable energy and transition to a green economy are critical for the planet's future."
}

# Paraphrase templates for each operator
TEMPLATES = {
    "Summarize": [
        "Summarize this in one sentence: {text}",
        "Provide a brief summary of the following: {text}",
        "What is the main point here? Summarize: {text}",
        "Condense this into a single sentence: {text}",
        "Briefly recap the following statement: {text}",
        "Give me a one-sentence summary of: {text}",
        "Sum up the core message of this: {text}",
        "Write a concise summary of the following text: {text}",
        "What's the gist of this? Provide a one-sentence summary: {text}",
        "In one sentence, tell me what this is about: {text}"
    ],
    "Criticize": [
        "Critically analyze the following statement, pointing out potential flaws: {text}",
        "What are the weaknesses in this argument? {text}",
        "Provide a critique of the following: {text}",
        "Point out the logical flaws in this statement: {text}",
        "Critically evaluate the claims made here: {text}",
        "Identify the potential downsides or errors in this text: {text}",
        "Offer a critical perspective on the following: {text}",
        "What is wrong with this reasoning? {text}",
        "Challenge the assumptions in this statement: {text}",
        "Provide a critical assessment of the following text: {text}"
    ],
    "Reframe Positively": [
        "Reframe the following statement to emphasize its optimistic potential: {text}",
        "Look at the bright side of this development: {text}",
        "How can we see this in a more positive light? {text}",
        "Describe the utopian possibilities of this: {text}",
        "Emphasize the benefits and progress described here: {text}",
        "What are the most hopeful aspects of this statement? {text}",
        "Present an optimistic view of the following: {text}",
        "Highlight the positive impact of this: {text}",
        "How does this create a better future? {text}",
        "Reframe this in a way that inspires hope: {text}"
    ],
    "Reframe Negatively": [
        "Reframe the following statement to emphasize its dystopian risks: {text}",
        "What are the dangerous consequences of this? {text}",
        "Look at the dark side of this development: {text}",
        "Describe the dystopian risks implied here: {text}",
        "Emphasize the threats and negative potential of this: {text}",
        "What are the most alarming aspects of this statement? {text}",
        "Present a pessimistic view of the following: {text}",
        "Highlight the potential harm of this: {text}",
        "How could this lead to a catastrophic future? {text}",
        "Reframe this in a way that emphasizes risk and danger: {text}"
    ],
    "Ask Question": [
        "Ask a provocative question inspired by this statement: {text}",
        "What is a deep question we should ask about this? {text}",
        "Pose an interesting question regarding the following: {text}",
        "What does this text make you wonder? Ask a question: {text}",
        "Give me a thought-provoking question based on this: {text}",
        "What is the most important question raised by this? {text}",
        "Create an inquiry based on the following text: {text}",
        "What would you ask a researcher about this? {text}",
        "Pose a philosophical question inspired by: {text}",
        "What is an unanswered question in this statement? {text}"
    ],
    "Explain Simply": [
        "Explain this statement like I am five: {text}",
        "Provide a simple explanation of the following: {text}",
        "Make this easy for a child to understand: {text}",
        "Simplify the following concept: {text}",
        "What does this mean in plain English? {text}",
        "Give me a basic overview of this statement: {text}",
        "Explain the following in the simplest way possible: {text}",
        "How would you explain this to a non-expert? {text}",
        "Write a simple, jargon-free explanation of: {text}",
        "Break this down into very simple terms: {text}"
    ],
    "Explain Technically": [
        "Explain the technical implications of this statement for a computer scientist: {text}",
        "Provide a specialized technical analysis of: {text}",
        "Describe the underlying mechanisms involved here: {text}",
        "What are the engineering challenges related to this? {text}",
        "Explain this from a highly technical perspective: {text}",
        "Detail the systemic complexities of the following: {text}",
        "Analyze the formal properties of this statement: {text}",
        "What are the technical prerequisites for this? {text}",
        "Discuss the implementation details of: {text}",
        "Provide a deep dive into the technical aspects of: {text}"
    ],
    "Opposite Stance": [
        "Argue persuasively for the exact opposite of this statement: {text}",
        "What is the counter-argument to this? {text}",
        "Take the opposing view and defend it: {text}",
        "Why might the opposite of this be true? {text}",
        "Write a rebuttal to the following statement: {text}",
        "Defend the contrary position to: {text}",
        "What are the strongest arguments against this? {text}",
        "Contradict the main claim made here: {text}",
        "Argue that the following statement is false: {text}",
        "Promote the inversely related idea of: {text}"
    ],
    "Poetic": [
        "Write a short poem inspired by the essence of this statement: {text}",
        "Capture the spirit of this in a few verses: {text}",
        "Express the following sentiment poetically: {text}",
        "Write a haiku or short poem about: {text}",
        "Turn this statement into a work of poetry: {text}",
        "What is the poetic truth beneath this? {text}",
        "Write a lyrical interpretation of: {text}",
        "Compose a short verse based on the following: {text}",
        "If this statement were a poem, how would it go? {text}",
        "Write an evocative and poetic piece inspired by: {text}"
    ],
    "Ethical Risks": [
        "Evaluate the ethical risks implied by this statement: {text}",
        "What are the moral concerns regarding this? {text}",
        "Analyze the ethical dilemmas presented here: {text}",
        "What potential for injustice exists in this? {text}",
        "Discuss the social and ethical pitfalls of: {text}",
        "How might this impact human rights? {text}",
        "What ethical boundaries are being tested here? {text}",
        "Evaluate the fairness and bias in this statement: {text}",
        "What are the ethical responsibilities involved? {text}",
        "Analyze the following from an ethical standpoint: {text}"
    ]
}

OUTPUT_FILE = "experiments/03_regime_invariants/data/regime_invariants_prompts.json"

def main():
    dataset = []
    
    for operator in OPERATORS:
        templates = TEMPLATES.get(operator, [])
        for topic_name, topic_content in TOPICS.items():
            # For each operator and topic, we use the 10 templates (paraphrases)
            # This gives 10 operators * 2 topics * 10 paraphrases = 200 samples
            for template in templates:
                prompt_text = template.format(text=topic_content)
                dataset.append({
                    "operator_label": operator,
                    "topic_label": topic_name,
                    "prompt_text": prompt_text
                })
                
    random.shuffle(dataset)
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Generated {len(dataset)} prompts in {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
