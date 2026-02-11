import google.generativeai as genai
import json
import os
import time
import pandas as pd
import numpy as np

# Configuration
API_KEY = os.environ.get("GOOGLE_API_KEY", "YOUR_API_KEY_HERE")
INPUT_FILE = "turns.jsonl"
OUTPUT_FILE = "states.parquet"
CACHE_FILE = "embedding_cache.json"
MODEL_NAME = "models/text-embedding-004"

# Setup API
genai.configure(api_key=API_KEY)

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)

def get_embedding(text, cache):
    if text in cache:
        return np.array(cache[text])
    
    try:
        result = genai.embed_content(model=MODEL_NAME, content=text)
        emb = result['embedding']
        cache[text] = emb
        return np.array(emb)
    except Exception as e:
        print(f"Error embedding text: {e}")
        return None

def main():
    print("Loading data...")
    if not os.path.exists(INPUT_FILE):
        print(f"Input file {INPUT_FILE} not found.")
        return

    conversations = []
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            conversations.append(json.loads(line))
            
    print(f"Loaded {len(conversations)} turns.")
    
    cache = load_cache()
    states = []
    
    # Needs to track history for rolling window. 
    # Provided dataset is single-turn per prompt so "rolling window" might just be prompt + response if treated as one-shot.
    # However, the spec says "rolling window: last k=2 turns concatenated". 
    # In this specific dataset, each line is independent. 
    # I will interpret:
    # (a) Assistant-only: assistant_response
    # (b) Pair: user_prompt + "\n\n" + assistant_response
    # (c) Rolling window: effectively same as (b) for this one-shot dataset, or previous turn + current.
    # checking the prompt 3 output, it's independent tasks. 
    # BUT "Rolling window" implies a sequence.
    # If the dataset is just independent prompts, k=2 rolling window is effectively just the current turn (k=1 history + current, but history is empty).
    # I will implement it such that if there were previous turns it would include them, but here it might be just current.
    
    # Wait, the request says "loads turns.jsonl". The turns are generated from `operator_prompts.jsonl`.
    # These are effectively 500 independent conversations. 
    # So k=2 rolling window for a single-turn conversation is just that one turn.
    
    total_ops = len(conversations) * 3
    processed = 0
    
    for i, turn in enumerate(conversations):
        turn_id = i
        operator = turn['operator_name']
        topic = turn['topic']
        
        u_t = turn['user_prompt']
        a_t = turn['assistant_response']
        
        # Define texts
        text_a = a_t
        text_b = u_t + "\n\n" + a_t
        # For this dataset, text_c (rolling k=2) is the same as text_b because there is no t-1.
        text_c = text_b 
        
        definitions = {
            'a': text_a,
            'b': text_b,
            'c': text_c
        }
        
        for def_key, text in definitions.items():
            processed += 1
            if processed % 10 == 0:
                print(f"Processing state {processed}/{total_ops}...", flush=True)
                
            emb = get_embedding(text, cache)
            
            if emb is not None:
                states.append({
                    "turn_id": turn_id,
                    "operator_name": operator,
                    "topic": topic,
                    "state_def": def_key,
                    "embedding": emb.tolist() # Store as list for compatibility, or numpy array
                })
        
        # Save cache periodically
        if i % 50 == 0:
            save_cache(cache)
            
    save_cache(cache)
    
    # Save to Parquet
    df = pd.DataFrame(states)
    df.to_parquet(OUTPUT_FILE)
    print(f"Saved {len(states)} states to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
