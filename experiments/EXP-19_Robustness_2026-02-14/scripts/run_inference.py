import os
import sys
import json
import time
import re
import argparse
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm

# Hybrid Storage Configuration
SSD_ROOT = "experiments/EXP-19_Robustness_2026-02-14"
HDD_ROOT = r"D:\Dev\Projects Cold Storage\Trajectory Geometry\EXP-19_Robustness_2026-02-14"

# Models to process
MODELS = {
    "qwen05b": "Qwen/Qwen2.5-0.5B",
    "qwen15b": "Qwen/Qwen2.5-1.5B",
    "pythia410m": "EleutherAI/pythia-410m"
}

# Generation Configs
DIRECT_CONFIG = {
    'max_new_tokens': 15,
    'do_sample': False,
    'temperature': 1.0,
    'repetition_penalty': 1.0,
}

COT_CONFIG = {
    'max_new_tokens': 200,
    'do_sample': False,
    'temperature': 1.0,
    'repetition_penalty': 1.0,
}

# Stop Sequences
STOP_STRINGS_DIRECT = ["\n", "Question", "Calculate", "Let", "Step", "The ", "Because"]
STOP_STRINGS_COT = ["Question:", "Calculate:", "Problem:", "\n\nQ:", "\n\nCalculate", "Let me verify", "Alternatively", "Double check"]

def run_worker_process(model_key, model_name, problems_path, pilot=False):
    """Launch a separate process for model inference to ensure clean state."""
    # Use the venv on C drive
    python_exe = os.path.abspath(os.path.join(os.getcwd(), "exp19_venv", "Scripts", "python.exe"))
    script_path = os.path.join(SSD_ROOT, "scripts", "inference_worker.py")
    
    cmd = [
        python_exe, script_path,
        "--model_key", model_key,
        "--model_name", model_name,
        "--problems", problems_path,
        "--ssd_root", SSD_ROOT,
        "--hdd_root", HDD_ROOT
    ]
    if pilot:
        cmd.append("--pilot")
    
    print(f"\n>>> Starting Inference Worker for {model_key} ({model_name})")
    print(f">>> Command: {' '.join(cmd)}")
    
    process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    process.wait()
    
    if process.returncode != 0:
        print(f"!!! Worker for {model_key} failed with return code {process.returncode}")
    else:
        print(f">>> Worker for {model_key} completed successfully.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="Run a pilot test on 5 problems per model")
    parser.add_argument("--model", type=str, choices=list(MODELS.keys()) + ["all"], default="all", help="Specific model to run")
    args = parser.parse_args()

    # Ensure directories exist
    os.makedirs(os.path.join(SSD_ROOT, "data"), exist_ok=True)
    os.makedirs(os.path.join(HDD_ROOT, "data", "hidden_states"), exist_ok=True)
    
    if args.pilot:
        problems_path = os.path.join(SSD_ROOT, "data", "pilot_problems.json")
    else:
        problems_path = os.path.join(SSD_ROOT, "data", "problems.json")
        
    if not os.path.exists(problems_path):
        print(f"Error: Problems file not found at {problems_path}")
        return

    models_to_run = MODELS.keys() if args.model == "all" else [args.model]

    for m_key in models_to_run:
        run_worker_process(m_key, MODELS[m_key], problems_path, pilot=args.pilot)

if __name__ == "__main__":
    print(f"PID: {os.getpid()}", flush=True)
    main()
