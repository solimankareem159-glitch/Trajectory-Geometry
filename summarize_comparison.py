
import os
import pandas as pd

DATA_PATHS = {
    'Qwen-0.5B (EXP-14)': r"experiments/EXP-14_UniversalSignature_2025-12-03/data/exp14_metrics_full.csv",
    'Qwen-1.5B (EXP-16B)': r"experiments/EXP-16B_Qwen15B_2025-12-08/data/exp16b_metrics_clean.csv",
    'Pythia-70m (EXP-16)': r"experiments/EXP-16_Pythia70m_2025-12-07/data/exp16_metrics_salvaged.csv" 
}

def get_group(row):
    # Logic for Pythia
    if 'group' in row: return row['group']
    cond = row['condition'].lower() # 'cot' or 'direct'
    corr = bool(row['correct']) # True or False
    if 'direct' in cond:
        return 'G2' if corr else 'G1'
    elif 'cot' in cond:
        return 'G4' if corr else 'G3'
    return None

def analyze(name, path):
    if not os.path.exists(path): return
    df = pd.read_csv(path)
    
    # Ensure group column
    if 'group' not in df.columns:
        df['group'] = df.apply(get_group, axis=1)
        
    # Get last layer index
    max_layer = df['layer'].max()
    mid_layer = max_layer // 2
    
    print(f"--- {name} ---")
    print(f"Max Layer: {max_layer}")
    
    # Compare G4 vs G1 at Max Layer for Rg and Deff
    subset = df[df['layer'] == max_layer]
    
    for metric in ['radius_of_gyration', 'effective_dim']:
        if metric not in subset.columns: continue
        
        g4_mean = subset[subset['group']=='G4'][metric].mean()
        g1_mean = subset[subset['group']=='G1'][metric].mean()
        
        diff = g4_mean - g1_mean
        print(f"Layer {max_layer} {metric}: G4={g4_mean:.2f}, G1={g1_mean:.2f}, Diff={diff:.2f}")

def main():
    for name, path in DATA_PATHS.items():
        analyze(name, path)

if __name__ == "__main__":
    main()
