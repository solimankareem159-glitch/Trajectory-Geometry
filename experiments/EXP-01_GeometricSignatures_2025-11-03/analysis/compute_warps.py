import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import os

INPUT_FILE = "states.parquet"
OUTPUT_FILE = "warps.parquet"
PLOT_FILE = "warp_norm_over_turns.png"

def cosine_similarity(v1, v2):
    if v1 is None or v2 is None:
        return np.nan
    # cosine distance is 1 - similarity. 
    # But usually "cosine to prev warp" implies similarity (1.0 is aligned).
    # scipy cosine returns distance (0.0 is aligned).
    # I will return SIMILARITY: 1 - distance.
    try:
        dist = cosine(v1, v2)
        return 1.0 - dist
    except:
        return np.nan

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"File {INPUT_FILE} not found. Please run embed_states.py first.")
        return

    print("Loading states...")
    df = pd.read_parquet(INPUT_FILE)
    
    # Ensure turn_id is int
    df['turn_id'] = df['turn_id'].astype(int)
    
    # Process per state_def
    state_defs = df['state_def'].unique()
    all_warps = []
    
    for s_def in state_defs:
        print(f"Processing state definition: {s_def}")
        subset = df[df['state_def'] == s_def].copy()
        
        # Group by Topic to form "conversations"
        # Since the generation was 10 operators x 10 paraphrases x 5 topics,
        # we have 100 turns per topic.
        # We assume the turn_id reflects the order of generation.
        # Although generation was interleaved, the logical conversation is the sequence of events.
        # Let's verify by sorting by turn_id within Topic.
        
        subset = subset.sort_values(by=['topic', 'turn_id'])
        
        # Compute Warps
        topics = subset['topic'].unique()
        
        for topic in topics:
            topic_df = subset[subset['topic'] == topic].copy()
            
            # Embeddings series
            embeddings = topic_df['embedding'].values
            turn_ids = topic_df['turn_id'].values
            operators = topic_df['operator_name'].values
            
            # Delta_t = s_t - s_{t-1}
            # For t=0, warp is undefined or 0.
            
            prev_emb = None
            prev_warp = None
            
            for i in range(len(embeddings)):
                curr_emb = np.array(embeddings[i])
                
                if prev_emb is not None:
                    warp = curr_emb - prev_emb
                    warp_norm = np.linalg.norm(warp)
                    
                    if prev_warp is not None:
                        warp_cos_prev = cosine_similarity(warp, prev_warp)
                    else:
                        warp_cos_prev = np.nan
                        
                    all_warps.append({
                        "operator_name": operators[i],
                        "topic": topic,
                        "state_def": s_def,
                        "turn_id": turn_ids[i],
                        "warp_vector": warp.tolist(),
                        "warp_norm": warp_norm,
                        "warp_cos_prev": warp_cos_prev
                    })
                    
                    prev_warp = warp
                else:
                    # First item in conversation, no warp (or warp from 0?)
                    # MVP spec says Delta_t = s_t - s_{t-1}.
                    # We skip the first item as it has no predecessor in this sequence definition.
                    pass
                
                prev_emb = curr_emb
                
    # Save Warps
    warps_df = pd.DataFrame(all_warps)
    warps_df.to_parquet(OUTPUT_FILE)
    print(f"Saved {len(warps_df)} warps to {OUTPUT_FILE}")
    
    # Plotting
    if len(warps_df) > 0:
        plt.figure(figsize=(12, 6))
        
        for s_def in state_defs:
            s_data = warps_df[warps_df['state_def'] == s_def]
            # Average norm per turn index across topics? 
            # Or just scatter plot?
            # "warp_norm over turns".
            # Let's plot average warp norm per relative turn index in the conversation (0..99).
            
            # We need to compute relative index if we want to average.
            # But turn_id is global.
            # Let's re-calculate relative index 1..99
            
            # Or just plot all points.
            plt.plot(s_data['warp_norm'].values, label=f"State {s_def}", alpha=0.5)
            
        plt.title("Warp Norm over Turns (All Topics Concatenated)")
        plt.xlabel("Turn Sequence (Topic Aggregated)")
        plt.ylabel("Norm ||Delta||")
        plt.legend()
        plt.grid(True)
        plt.savefig(PLOT_FILE)
        print(f"Saved plot to {PLOT_FILE}")

if __name__ == "__main__":
    main()
