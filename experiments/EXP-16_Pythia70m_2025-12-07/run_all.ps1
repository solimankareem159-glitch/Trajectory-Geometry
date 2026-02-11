# Experiment 16 Pipeline Runner (PowerShell)
# ==========================================

$ErrorActionPreference = "Stop"

$scripts = @(
    "00_verify_inputs.py",
    "01_env_and_device.py",
    "02_preflight_20q.py",
    "03_run_inference_dump_hidden.py",
    "04_compute_metrics.py",
    "05_stats_and_tests.py",
    "06_make_figures.py",
    "07_write_report.py"
)

Write-Host "="*60
Write-Host "Experiment 16 Pipeline Runner"
Write-Host "="*60

$startTime = Get-Date

foreach ($script in $scripts) {
    Write-Host "`n$('='*60)"
    Write-Host "Running $script..."
    Write-Host "="*60
    
    $scriptPath = "experiments\Experiment 16\scripts\$script"
    $scriptStart = Get-Date
    
    & python $scriptPath
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n✗ $script FAILED"
        exit 1
    }
    
    $scriptElapsed = (Get-Date) - $scriptStart
    Write-Host "`n✓ $script completed in $($scriptElapsed.TotalSeconds.ToString('F2'))s"
}

$totalElapsed = (Get-Date) - $startTime

Write-Host "`n$('='*60)"
Write-Host "Pipeline Completed Successfully in $($totalElapsed.TotalSeconds.ToString('F2'))s"
Write-Host "="*60
