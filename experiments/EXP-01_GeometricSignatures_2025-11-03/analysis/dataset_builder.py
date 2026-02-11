import json
import random

# Set deterministic seed
random.seed(42)

# 1. Operators and Paraphrases (10 operators x 10 paraphrases)
OPERATORS = {
    "Clarify": [
        "Ask clarifying questions to better understand the issue.",
        "What specific details are missing? Ask about them.",
        "Request more information to clear up ambiguity.",
        "Can you elaborate on the key points?",
        "Identify and ask about any vague terms.",
        "Pose questions to uncover the underlying assumptions.",
        "Inquire about the context to get a fuller picture.",
        "What do you mean exactly? Ask for specific examples.",
        "Probe for more details to avoid misunderstandings.",
        "Seek clarification on the most confusing parts."
    ],
    "Summarize": [
        "Summarize the main points of the text.",
        "Give a brief overview of what has been said.",
        "Condense the information into a few key sentences.",
        "Provide a concise summary of the argument.",
        "Recap the essential details.",
        "What is the gist of this? Summarize it.",
        "Boil this down to the core message.",
        "Create a short executive summary.",
        "Synthesize the main takeaways.",
        "Wrap up the discussion with a helper summary."
    ],
    "Reframe": [
        "Reframe this situation in a more constructive light.",
        "Propose a different way to view this problem.",
        "How else can we interpret this? Offer an alternative.",
        "Shift the perspective to see new possibilities.",
        "Turn this into a learning opportunity.",
        "Spin this in a positive or different direction.",
        "Look at this from the opposite angle.",
        "Reinterpret the core conflict differently.",
        "Offer a fresh perspective on the matter.",
        "Change the frame of reference."
    ],
    "Zoom Out": [
        "Zoom out and look at the bigger picture.",
        "What is the broader context here?",
        "Step back and consider the long-term implications.",
        "Connect this to the larger system or trend.",
        "Take a high-level view of the situation.",
        "Ignore the details for a moment; what's the macro view?",
        "Broaden the scope of the discussion.",
        "How does this fit into the grand scheme of things?",
        "Elevate the conversation to a higher level of abstraction.",
        "Consider the global perspective."
    ],
    "Zoom In": [
        "Zoom in on the specific details.",
        "Focus on the immediate next steps.",
        "Drill down into the specifics of this point.",
        "Let's look at the nitty-gritty details.",
        "Examine the finer points closely.",
        "Concentrate on a concrete example.",
        "Narrow the focus to a specific element.",
        "Get granular with the analysis.",
        "Pay attention to the minute details.",
        "Zero in on the practical implementation."
    ],
    "Steelman": [
        "Steelman the opposing argument.",
        "Construct the strongest possible version of the counter-argument.",
        "Make the best case for the other side.",
        "Present the opposing view in its most charitable form.",
        "Strengthen the argument you disagree with.",
        "assume the other side is right; what is their best point?",
        "Build a robust defense for the opposing perspective.",
        "Articulate the strongest reasons for the alternative view.",
        "Give the other side the benefit of the doubt and explain their logic.",
        "Champion the counter-position effectively."
    ],
    "Devil's Advocate": [
        "Play devil's advocate and challenge this.",
        "What are the potential flaws in this reasoning?",
        "Raise objections to this proposal.",
        "Critique this idea from a skeptic's point of view.",
        "Identify the weak points in this argument.",
        "Propose counter-arguments to test the validity.",
        "What could go wrong? Highlight the risks.",
        "Challenge the assumptions made here.",
        "Offer a critical perspective.",
        "Poke holes in the theory."
    ],
    "Formalize": [
        "Formalize this using equations or logic.",
        "Translate this into a mathematical model.",
        "Define these terms strictly and formally.",
        "Structure this argument using formal logic.",
        "Convert this description into pseudocode or code.",
        "Express this relationship with a formula.",
        "Use precise notation to represent this.",
        "Create a structured framework for this concept.",
        "Systematize this information.",
        "Put this into a standard format or schema."
    ],
    "Decompose": [
        "Decompose this problem into smaller parts.",
        "Break this down into manageable sub-tasks.",
        "Split the issue into its constituent components.",
        "Divide and conquer this complex topic.",
        "Analyze the individual elements separately.",
        "Fragment the problem into simpler steps.",
        "Unpack this concept into its building blocks.",
        "Dissect the problem to understand its structure.",
        "Segment the task into phases.",
        "Atomize the complex argument."
    ],
    "Plan": [
        "Create a plan to address this.",
        "Outline the steps to move forward.",
        "Develop a roadmap for this project.",
        "Propose an actionable sequence of events.",
        "Strategy for execution? Write it down.",
        "Draft a project plan.",
        "What is the timeline? Plan it out.",
        "Design a course of action.",
        "Formulate a step-by-step procedure.",
        "Construct a rollout plan."
    ]
}

# 2. Topics and Base Input Texts (1 base input per topic)
TOPICS = {
    "AI Safety": "I am worried that as AI systems become more powerful, they might deceive humans to achieve their goals, leading to catastrophic outcomes. We haven't solved the alignment problem yet.",
    "Relationships": "My partner and I constantly argue about household chores. I feel like I do everything, and they feel nagged. It's creating a lot of tension in our relationship.",
    "Career": "I've been in the same job for five years. It's comfortable, but I'm not learning anything new. I'm afraid to leave and lose my stability, but I'm bored.",
    "Politics": "The polarization in the country is getting out of hand. People can't even talk to each other anymore without getting angry. It feels like democracy is breaking down.",
    "Health": "I've been trying to lose weight for years. I start a diet, do well for a few weeks, then binge eating and gain it all back. I feel like I have no willpower."
}

# 3. Generate Dataset
dataset = []

for operator, paraphrases in OPERATORS.items():
    for i, paraphrase in enumerate(paraphrases):
        for topic, base_input in TOPICS.items():
            
            # Construct Full User Prompt: Instruction + \n\n + Base Input
            full_prompt = f"{paraphrase}\n\n{base_input}"
            
            entry = {
                "operator_name": operator,
                "operator_paraphrase_id": i,
                "topic": topic,
                "base_input_text": base_input,
                "instruction_text": paraphrase,
                "full_user_prompt": full_prompt
            }
            dataset.append(entry)

# 4. Save to JSONL
output_file = "operator_prompts.jsonl"
with open(output_file, 'w') as f:
    for entry in dataset:
        f.write(json.dumps(entry) + '\n')

print(f"Generated {len(dataset)} entries in {output_file}")
