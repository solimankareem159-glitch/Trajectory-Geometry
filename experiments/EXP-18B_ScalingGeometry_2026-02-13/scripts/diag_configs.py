
import os
import pandas as pd
import json

ROOT_DIR = r"c:\Dev\Projects\Trajectory Geometry"
CONFIGS = [
    {
        "name": "qwen_0_5b",
        "model_id": "Qwen/Qwen2.5-0.5B",
        "hidden_dir": os.path.join(ROOT_DIR, "experiments", "EXP-14_UniversalSignature_2025-12-03", "data", "hidden_states_full"),
        "metadata_path": os.path.join(ROOT_DIR, "experiments", "EXP-14_UniversalSignature_2025-12-03", "data", "metadata_full.csv"),
    },
    {
        "name": "qwen_1_5b",
        "model_id": "Qwen/Qwen2.5-1.5B",
        "hidden_dir": os.path.join(ROOT_DIR, "experiments", "EXP-16B_Qwen15B_2025-12-08", "data", "hidden_states_clean"), 
        "metadata_path": os.path.join(ROOT_DIR, "experiments", "EXP-16B_Qwen15B_2025-12-08", "data", "metadata_reparsed.csv"), 
    },
    {
        "name": "pythia_70m",
        "model_id": "EleutherAI/pythia-70m",
        "hidden_dir": os.path.join(ROOT_DIR, "experiments", "EXP-16_Pythia70m_2025-12-07", "data", "hidden_states_clean"),
        "metadata_path": os.path.join(ROOT_DIR, "experiments", "EXP-16_Pythia70m_2025-12-07", "data", "metadata.csv"),
    }
]

for cfg in CONFIGS:
    print(f"\nChecking Model: {cfg['name']}")
    print(f"  Hidden Dir: {cfg['hidden_dir']} - exists: {os.path.exists(cfg['hidden_dir'])}")
    print(f"  Metadata: {cfg['metadata_path']} - exists: {os.path.exists(cfg['metadata_path'])}")
    
    if os.path.exists(cfg['hidden_dir']):
        files = [f for f in os.listdir(cfg['hidden_dir']) if f.endswith('.npy')]
        print(f"  File count: {len(files)}")
    
    if os.path.exists(cfg['metadata_path']):
        try:
            df = pd.read_csv(cfg['metadata_path'])
            print(f"  Metadata rows: {len(df)}")
            print(f"  Columns: {list(df.columns)}")
        except Exception as e:
            print(f"  Error reading metadata: {e}")

exp9_path = os.path.join(ROOT_DIR, "experiments", "EXP-18_ConsolidatedMetricSuite_2026-02-13", "Data", "exp9_dataset.jsonl")
print(f"\nChecking Exp 9 Truth: {exp9_path} - exists: {os.path.exists(exp9_path)}")
