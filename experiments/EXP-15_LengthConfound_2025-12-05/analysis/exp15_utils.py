"""
Experiment 15 Utilities
=======================
Shared functions for input discovery, data loading, and directml support.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import torch
try:
    import torch_directml
    HAS_DML = True
except ImportError:
    HAS_DML = False

# --- Path Configuration ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
EXP14_DATA_DIR = os.path.join(ROOT_DIR, "experiments", "EXP-14_UniversalSignature_2025-12-03", "data")
EXP9_DATA_DIR = os.path.join(ROOT_DIR, "experiments", "EXP-09_GeometryCapability_2025-11-22", "data")
EXP15_DATA_DIR = os.path.join(ROOT_DIR, "experiments", "EXP-15_LengthConfound_2025-12-05", "data")
EXP15_FIGURES_DIR = os.path.join(ROOT_DIR, "experiments", "EXP-15_LengthConfound_2025-12-05", "figures")
EXP15_REPORTS_DIR = os.path.join(ROOT_DIR, "experiments", "EXP-15_LengthConfound_2025-12-05", "reports")

def get_device(use_directml=True):
    """Returns the best available device."""
    if use_directml and HAS_DML and torch_directml.is_available():
        return torch_directml.device()
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def locate_inputs():
    """Locates required input files from previous experiments."""
    metrics_file = os.path.join(EXP14_DATA_DIR, "exp14_metrics_full.csv")
    dataset_file = os.path.join(EXP9_DATA_DIR, "exp9_dataset.jsonl")
    
    if not os.path.exists(metrics_file):
        # Try local search if paths shifted
        print(f"Warning: {metrics_file} not found. Searching...")
        # Placeholder for deeper search if needed
        raise FileNotFoundError(f"Could not find Exp 14 metrics at {metrics_file}")
        
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Could not find Exp 9 dataset at {dataset_file}")
        
    return metrics_file, dataset_file

def load_data():
    """Loads and joins Exp 14 metrics with Exp 9 dataset context."""
    metrics_path, dataset_path = locate_inputs()
    
    # Load metrics
    print(f"Loading metrics from {metrics_path}...")
    df_metrics = pd.read_csv(metrics_path)
    
    # Load dataset to get text context
    print(f"Loading dataset from {dataset_path}...")
    dataset = []
    with open(dataset_path, 'r') as f:
        for line in f:
            dataset.append(json.loads(line))
            
    # We need to map metrics back to dataset. 
    # Metrics df has 'problem_id', 'condition'
    
    # Enrich df with dataset fields
    # Create lookup
    lookup = {}
    for i, rec in enumerate(dataset):
        lookup[(i, 'direct')] = rec['direct']
        lookup[(i, 'cot')] = rec['cot']
        # Also store ground truth and question 
        lookup[(i, 'meta')] = {'question': rec['question'], 'answer': rec['truth']}

    # Add text columns
    text_data = []
    
    # This might be slow for big DFs? 15k rows is fine.
    # Group by problem_id/condition to be safe
    # Actually df_metrics has one row per layer. 
    # We want sample-level metadata attached to every row, or just a sample-level df?
    # The utils should probably return the merged structure or helper to get it.
    
    return df_metrics, dataset

def ensure_dirs():
    os.makedirs(EXP15_DATA_DIR, exist_ok=True)
    os.makedirs(EXP15_FIGURES_DIR, exist_ok=True)
    os.makedirs(EXP15_REPORTS_DIR, exist_ok=True)

print(f"Utils initialized. Root: {ROOT_DIR}")
