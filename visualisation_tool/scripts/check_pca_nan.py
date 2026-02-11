import json
import math

def is_bad(x):
    return x is None or (isinstance(x, float) and math.isnan(x))

d = json.load(open(r'visualisation_tool/public/trajectory_data.json'))
print(f"Checking {len(d['trajectories'])} trajectories...")

bad_pca_count = 0
for t in d['trajectories']:
    for l in t['layers']:
        if 'pca_path' not in l:
            continue
        for pt in l['pca_path']:
            if any(is_bad(c) for c in pt):
                bad_pca_count += 1
                break

print(f"Layers with NaN/None in PCA: {bad_pca_count}")

# Check first few points
t0 = d['trajectories'][0]
l0 = t0['layers'][0]
print(f"Sample PCA Point: {l0['pca_path'][0]}")
