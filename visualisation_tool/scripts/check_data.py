import json
import os

data_path = 'public/trajectory_data.json'
if not os.path.exists(data_path):
    print(f"ERROR: {data_path} not found.")
    exit(1)

try:
    with open(data_path, 'r') as f:
        d = json.load(f)
    print(f"SUCCESS: Loaded {data_path}")
    print(f"Trajectories count: {len(d.get('trajectories', []))}")
    if 'trajectories' not in d:
        print("ERROR: Missing 'trajectories' key in data.")
    else:
        # Check first trajectory
        t = d['trajectories'][0]
        print(f"First trajectory ID: {t.get('id')}")
        print(f"Layers count: {len(t.get('layers', []))}")
except Exception as e:
    print(f"ERROR: Failed to parse JSON: {e}")
