"""
Experiment 15 Runner
====================
Runs all scripts in sequence cross-platform.
"""

import os
import sys
import subprocess
import time

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = os.path.join(root_dir, "scripts")
    
    scripts = [
        "00_env_check.py",
        "01_ingest_and_join.py",
        "02_analysis_A_difficulty_x_geometry.py",
        "03_analysis_B_failure_subtyping_with_jsonl_context.py",
        "04_analysis_C_token_level_dynamics_sliding_windows.py",
        "05_analysis_D_response_length_anomaly.py",
        "06_analysis_E_direct_only_successes_deep_dive.py",
        "07_new_signals_extract.py",
        "08_report_compile.py"
    ]
    
    print(f"Starting Experiment 15 Pipeline from {root_dir}")
    t0_total = time.time()
    
    for script in scripts:
        print(f"\n{'='*60}\nRunning {script}...\n{'='*60}")
        t0 = time.time()
        
        script_path = os.path.join(scripts_dir, script)
        
        # Determine python executable (sys.executable to use current venv)
        cmd = [sys.executable, script_path]
        
        try:
            subprocess.run(cmd, check=True)
            dt = time.time() - t0
            print(f"-> Finished {script} in {dt:.2f}s")
            
        except subprocess.CalledProcessError as e:
            print(f"!!! Error running {script}: Exit code {e.returncode}")
            sys.exit(e.returncode)
            
    dt_total = time.time() - t0_total
    print(f"\nPipeline Completed Successfully in {dt_total:.2f}s")

if __name__ == "__main__":
    main()
