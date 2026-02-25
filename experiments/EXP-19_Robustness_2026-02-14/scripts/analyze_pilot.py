import os
import pandas as pd
import argparse

def analyze_pilot(ssd_root):
    data_dir = os.path.join(ssd_root, "data")
    model_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d != "pilot"]
    
    all_summaries = []
    
    print("\n" + "="*50)
    print("EXPERIMENT 19 PILOT ACCURACY ANALYSIS")
    print("="*50)
    
    for model_key in model_dirs:
        metadata_path = os.path.join(data_dir, model_key, "metadata.csv")
        if not os.path.exists(metadata_path):
            continue
            
        df = pd.read_csv(metadata_path)
        
        # Accuracy by Bin
        acc_by_bin = df.groupby(['bin', 'condition'])['correct'].mean().unstack()
        
        print(f"\nModel: {model_key}")
        print("-" * 20)
        print(acc_by_bin)
        
        # Overall Accuracy
        overall_acc = df.groupby('condition')['correct'].mean()
        print(f"\nOverall: Direct: {overall_acc.get('direct', 0):.2%}, CoT: {overall_acc.get('cot', 0):.2%}")
        
        # Check for hallucinations/contamination
        contamination_rate = df['contaminated'].mean()
        print(f"Contamination/Hallucination Rate: {contamination_rate:.2%}")

    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssd_root", type=str, default="experiments/EXP-19_Robustness_2026-02-14")
    args = parser.parse_args()
    analyze_pilot(args.ssd_root)
