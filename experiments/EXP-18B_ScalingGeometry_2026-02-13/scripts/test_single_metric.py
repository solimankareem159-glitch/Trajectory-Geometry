
import os
import sys
import numpy as np

# Add local paths
ROOT_DIR = r"c:\Dev\Projects\Trajectory Geometry"
SCRIPTS_DIR = os.path.join(ROOT_DIR, "experiments", "EXP-18B_ScalingGeometry_2026-02-13", "scripts")
sys.path.append(SCRIPTS_DIR)

from metric_suite import TrajectoryMetrics, compute_all_metrics

# Load a sample file
hidden_dir = os.path.join(ROOT_DIR, "experiments", "EXP-14_UniversalSignature_2025-12-03", "data", "hidden_states_full")
sample_file = "0_cot.npy"
h_stack = np.load(os.path.join(hidden_dir, sample_file))
print(f"Loaded {sample_file} - shape: {h_stack.shape}")

# Setup TM
tm = TrajectoryMetrics()

# Context
ctx = {
    'truth_id': 342,
    'operand_ids': [],
    'centroids': {},
    'correct_vec': np.random.randn(896).astype(np.float32), # Placeholder
    'embed_vec': np.random.randn(896).astype(np.float32)   # Placeholder
}

print("Computing metrics...")
try:
    results = compute_all_metrics(h_stack, tm, ctx)
    print(f"Success! Generated {len(results)} rows.")
    print("Sample metrics keys:", list(results[0].keys()))
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Error: {e}")
