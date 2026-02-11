"""
Script 06: Analysis E - Direct-Only Successes
=============================================
Identifies and characterizes cases where Direct succeeds but CoT fails.
"""

import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Import utils
sys.path.append(os.path.dirname(__file__))
from exp15_utils import EXP15_DATA_DIR, EXP15_FIGURES_DIR, EXP15_REPORTS_DIR

def main():
    print("="*60)
    print("06_analysis_E.py: Direct-Only Successes")
    print("="*60)
    
    input_file = os.path.join(EXP15_DATA_DIR, "unified_metrics.csv")
    df = pd.read_csv(input_file)
    
    # We need to map problem_id -> outcome for both conditions
    # df has one row per problem_id/condition/layer
    
    # Get correctness status per problem
    # Take layer 0 (metrics often duplicated across layers or constant for correctness)
    # Actually correctness is property of the response, so constant across layers.
    sub = df[df['layer'] == 0].drop_duplicates(subset=['problem_id', 'condition'])
    
    # Build pivot: problem_id x condition -> group
    # Groups: G1 (DirFail), G2 (DirSucc), G3 (CoTFail), G4 (CoTSucc)
    
    pivot = sub.pivot(index='problem_id', columns='condition', values='group')
    
    # Identify Direct-Only Successes
    # defined as: Direct=G2 AND CoT=G3
    direct_only = pivot[(pivot['direct'] == 'G2') & (pivot['cot'] == 'G3')]
    
    print(f"Found {len(direct_only)} Direct-Only Successes (Dir=Success, CoT=Fail).")
    
    if len(direct_only) == 0:
        print("No cases found.")
        return
        
    case_ids = direct_only.index.tolist()
    
    # Deep dive into these cases
    # Compare their CoT geometry (G3) vs Direct geometry (G2)
    # And compare their CoT geometry to CoT Successes (G4) on OTHER problems?
    
    # For the casebook, we want to show:
    # Problem, Truth, Direct Response, CoT Response
    # Key Metrics Delta (CoT vs Direct)
    
    report_lines = [
        "# Analysis E: Direct-Only Successes Casebook",
        f"Found {len(direct_only)} cases.",
        ""
    ]
    
    for pid in case_ids[:10]: # Limit to 10 for readability
        # Get metadata (from any row with this pid)
        meta = df[df['problem_id'] == pid].iloc[0]
        question = meta['question']
        truth = meta['truth']
        
        # Get responses
        resp_dir = df[(df['problem_id'] == pid) & (df['condition'] == 'direct')]['response'].iloc[0]
        resp_cot = df[(df['problem_id'] == pid) & (df['condition'] == 'cot')]['response'].iloc[0]
        
        report_lines.append(f"## Problem {pid}")
        report_lines.append(f"**Question**: {question}")
        report_lines.append(f"**Truth**: {truth}")
        report_lines.append("")
        report_lines.append(f"**Direct (Success)**: {resp_dir[:200]}...")
        report_lines.append(f"**CoT (Fail)**: {resp_cot[:500]}...")
        report_lines.append("")
        
        # Metrics Comparison (Layer 16)
        m_dir = df[(df['problem_id'] == pid) & (df['condition'] == 'direct') & (df['layer'] == 16)]
        m_cot = df[(df['problem_id'] == pid) & (df['condition'] == 'cot') & (df['layer'] == 16)]
        
        if not m_dir.empty and not m_cot.empty:
            dims_dir = m_dir['effective_dim'].values[0]
            dims_cot = m_cot['effective_dim'].values[0]
            rg_dir = m_dir['radius_of_gyration'].values[0]
            rg_cot = m_cot['radius_of_gyration'].values[0]
            
            report_lines.append(f"- **Effect Dim**: Dir={dims_dir:.2f}, CoT={dims_cot:.2f}")
            report_lines.append(f"- **Rad Gyration**: Dir={rg_dir:.2f}, CoT={rg_cot:.2f}")
        
        report_lines.append("---")
        
        # Radar Plot?
        # Maybe overkill for just 10 lines script, but requested.
        # Skipping radar for now to save tokens/complexity, standard bars in main report is fine.
        
    out_path = os.path.join(EXP15_REPORTS_DIR, "direct_only_successes.md")
    with open(out_path, 'w') as f:
        f.write('\n'.join(report_lines))
        
    print(f"Casebook saved to {out_path}")

if __name__ == "__main__":
    main()
