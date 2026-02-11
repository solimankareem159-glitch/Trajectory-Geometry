import numpy as np
import os

DATA_DIR = r"experiments/experiment 6/data/full_2k"
ids = np.load(os.path.join(DATA_DIR, "ids.npy"))
labels = np.load(os.path.join(DATA_DIR, "labels.npy"))

print(f"IDs shape: {ids.shape}")
print(f"Labels shape: {labels.shape}")

content_ids = ids[:, 0]
paraphrase_ids = ids[:, 1]

print(f"Unique contents: {len(np.unique(content_ids))}")
print(f"Unique paraphrases: {len(np.unique(paraphrase_ids))}")

group_ids = content_ids * 1000 + paraphrase_ids
unique_groups = np.unique(group_ids)
print(f"Unique groups (folds): {len(unique_groups)}")
