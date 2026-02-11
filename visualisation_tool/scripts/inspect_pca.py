import json

data_path = 'public/trajectory_data.json'
try:
    with open(data_path, 'r') as f:
        d = json.load(f)
    
    for t_idx, t in enumerate(d.get('trajectories', [])[:5]):
        print(f"Trajectory {t_idx} (ID: {t.get('id')}):")
        layers = t.get('layers', [])
        print(f"  Layers: {len(layers)}")
        for l_idx, l in enumerate(layers[:3]):
            pca = l.get('pca_path', [])
            print(f"    Layer {l_idx}: pca_path length = {len(pca)}")
            if len(pca) > 0:
                print(f"      First PCA point: {pca[0]}")
except Exception as e:
    print(f"ERROR: {e}")
