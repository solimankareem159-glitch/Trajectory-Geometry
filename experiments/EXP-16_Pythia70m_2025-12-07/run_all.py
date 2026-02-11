"""
Runner: Execute all Experiment 16 scripts sequentially
======================================================
Cross-platform Python runner with resume capability.
"""

import os
import sys
import subprocess
import time

SCRIPTS = [
    "00_verify_inputs.py",
    "01_env_and_device.py",
    "02_preflight_20q.py",
    "03_run_inference_dump_hidden.py",
    "04_compute_metrics.py",
    "05_stats_and_tests.py",
    "06_make_figures.py",
    "07_write_report.py"
]

def run_script(script_name):
    """Run a single script and return status."""
    print("\n" + "="*60)
    print(f"Running {script_name}...")
    print("="*60)
    
    python_exe = sys.executable
    script_path = os.path.join("experiments", "Experiment 16", "scripts", script_name)
    
    start_time = time.time()
    result = subprocess.run([python_exe, script_path], cwd=".")
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"\n[FAIL] {script_name} FAILED (exit code {result.returncode})")
        return False
    
    print(f"\n[OK] {script_name} completed in {elapsed:.2f}s")
    return True

def main():
    print("="*60)
    print("Experiment 16 Pipeline Runner")
    print("="*60)
    
    overall_start = time.time()
    
    for script in SCRIPTS:
        if not run_script(script):
            print("\nPipeline FAILED. Fix errors and re-run.")
            sys.exit(1)
    
    overall_elapsed = time.time() - overall_start
    
    print("\n" + "="*60)
    print(f"Pipeline Completed Successfully in {overall_elapsed:.2f}s")
    print("="*60)

if __name__ == "__main__":
    main()
