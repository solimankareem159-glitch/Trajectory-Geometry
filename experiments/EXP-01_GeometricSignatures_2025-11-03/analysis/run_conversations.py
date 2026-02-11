import google.generativeai as genai
import json
import time
import os
from datetime import datetime

# Configuration
API_KEY = os.environ.get("GOOGLE_API_KEY", "YOUR_API_KEY_HERE")
MODEL_NAME = "models/gemini-2.0-flash"
TEMPERATURE = 0.2
INPUT_FILE = "operator_prompts.jsonl"
OUTPUT_FILE = "turns.jsonl"

# Setup API
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(
    MODEL_NAME,
    generation_config=genai.GenerationConfig(temperature=TEMPERATURE)
)

def run_conversations():
    print(f"Starting generation using {MODEL_NAME} with temp {TEMPERATURE}...")
    
    # Read inputs
    with open(INPUT_FILE, 'r') as f:
        inputs = [json.loads(line) for line in f]
    
    results = []
    
    # Process each prompt
    count = 0
    total = len(inputs)
    
    # Create/Clear output file
    with open(OUTPUT_FILE, 'w') as f:
        pass

    for entry in inputs:
        count += 1
        operator = entry['operator_name']
        topic = entry['topic']
        full_user_prompt = entry['full_user_prompt']
        
        print(f"[{count}/{total}] Processing {operator} on {topic}...", flush=True)
        
        try:
            # Generate content
            response = model.generate_content(full_user_prompt)
            assistant_response = response.text
            
            # Record result
            result_entry = {
                "operator_name": operator,
                "topic": topic,
                "user_prompt": full_user_prompt,
                "assistant_response": assistant_response,
                "model_name": MODEL_NAME,
                "timestamp": datetime.now().isoformat()
            }
            results.append(result_entry)

            # Append to file immediately
            with open(OUTPUT_FILE, 'a') as f:
                f.write(json.dumps(result_entry) + '\n')
            
            # Rate limit handling (simple sleep)
            time.sleep(1) 
            
        except Exception as e:
            print(f"Error processing item {count}: {e}", flush=True)
    
    # Save results
    with open(OUTPUT_FILE, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + '\n')
            
    print(f"Finished! Saved {len(results)} conversation turns to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_conversations()
