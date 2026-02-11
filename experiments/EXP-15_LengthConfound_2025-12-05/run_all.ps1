$ErrorActionPreference = "Stop"

Write-Host "Starting Experiment 15 Pipeline..." -ForegroundColor Green

$scripts = @(
    "00_env_check.py",
    "01_ingest_and_join.py",
    "02_analysis_A_difficulty_x_geometry.py",
    "03_analysis_B_failure_subtyping_with_jsonl_context.py",
    "04_analysis_C_token_level_dynamics_sliding_windows.py",
    "05_analysis_D_response_length_anomaly.py",
    "06_analysis_E_direct_only_successes_deep_dive.py",
    "07_new_signals_extract.py",
    "08_report_compile.py"
)

foreach ($script in $scripts) {
    Write-Host "Running $script..." -ForegroundColor Cyan
    $start = Get-Date
    
    # Run simple python command - modify if using specific venv or path is tricky
    # Assuming run from root experiments dir or Exp 15 dir?
    # Relative path from here
    $scriptPath = Join-Path "scripts" $script
    
    python $scriptPath
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Script $script failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
    
    $end = Get-Date
    $duration = $end - $start
    Write-Host "Finished $script in $($duration.TotalSeconds)s" -ForegroundColor Green
}

Write-Host "Experiment 15 Pipeline Completed Successfully!" -ForegroundColor Green
