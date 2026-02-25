import json
import os

SSD_ROOT = "experiments/EXP-19_Robustness_2026-02-14"
PROBLEMS_PATH = os.path.join(SSD_ROOT, "data", "problems.json")
PILOT_OUTPUT = os.path.join(SSD_ROOT, "data", "pilot_problems.json")

with open(PROBLEMS_PATH, 'r') as f:
    problems = json.load(f)

# Select 4 from each bin
# Bins are: UltraSmall (0-79), Small (80-159), Medium (160-239), Large (240-319), Negative (320-399)
pilot_probs = []
for i in range(5):
    start_idx = i * 80
    pilot_probs.extend(problems[start_idx : start_idx + 4])

with open(PILOT_OUTPUT, 'w') as f:
    json.dump(pilot_probs, f, indent=2)

print(f"Created pilot dataset at {PILOT_OUTPUT} with {len(pilot_probs)} problems.")
