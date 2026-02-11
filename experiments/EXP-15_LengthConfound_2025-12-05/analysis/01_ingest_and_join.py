"""
Script 01: Ingest and Join
==========================
Loads JSONL + Metrics and produces a unified dataframe.
"""

import os
import sys
import pandas as pd
import numpy as np

# Import utils
sys.path.append(os.path.dirname(__file__))
from exp15_utils import load_data, EXP15_DATA_DIR

def main():
    print("="*60)
    print("01_ingest_and_join.py: Data Ingestion")
    print("="*60)
    
    # Load raw data
    df_metrics, dataset = load_data()
    
    print(f"Metrics shape: {df_metrics.shape}")
    print(f"Dataset size: {len(dataset)}")
    
    # We want to add 'prompt', 'response', 'question', 'answer_truth' to metrics
    # Metric rows are identified by problem_id and condition
    
    print("Enriching metrics with text data...")
    
    # Pre-compute lookup dicts to avoid O(N*M)
    # Map (problem_id, condition) -> {prompt, response}
    # Map (problem_id) -> {question, truth}
    
    text_map = {}
    problem_map = {}
    
    for i, rec in enumerate(dataset):
        problem_map[i] = {
            'question': rec['question'],
            'truth': rec['truth']
        }
        
        for cond in ['direct', 'cot']:
            text_map[(i, cond)] = {
                'prompt': rec[cond]['prompt'],
                'response': rec[cond]['response'],
                'parsed_response': rec[cond].get('parsed_response', ''),
                # Recalculate length here to be safe/consistent
                'response_length_chars': len(rec[cond]['response'])
            }
            
    # Apply to dataframe
    # Iterate is slow, but explicit. 
    # Faster: create a temp DF from dicts and merge.
    
    print("Creating temporary metadata dataframe...")
    meta_records = []
    for (pid, cond), data in text_map.items():
        p_data = problem_map[pid]
        meta_records.append({
            'problem_id': pid,
            'condition': cond,
            'question': p_data['question'],
            'truth': p_data['truth'],
            'prompt': data['prompt'],
            'response': data['response'],
            'response_length_chars': data['response_length_chars']
        })
        
    df_meta = pd.DataFrame(meta_records)
    
    print("Merging...")
    # metrics has problem_id, condition. 
    # Ensure types match
    df_metrics['problem_id'] = df_metrics['problem_id'].astype(int)
    df_meta['problem_id'] = df_meta['problem_id'].astype(int)
    
    df_unified = pd.merge(df_metrics, df_meta, on=['problem_id', 'condition'], how='left')
    
    print(f"Unified shape: {df_unified.shape}")
    
    # Save
    out_file = os.path.join(EXP15_DATA_DIR, "unified_metrics.csv")
    # Use pickle for speed/fidelity if large, but CSV requested for interop?
    # CSV is specified in artifact locator for compatibility.
    # But for internal passing, pickle/parquet is better.
    # I'll save CSV as primary artifact.
    
    df_unified.to_csv(out_file, index=False)
    print(f"Saved unified dataframe to {out_file}")

if __name__ == "__main__":
    main()
