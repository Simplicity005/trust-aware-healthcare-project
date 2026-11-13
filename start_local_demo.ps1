# ==============================================================================
# Windows PowerShell Local Demo Launcher
# ==============================================================================
Set-Location $PSScriptRoot

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  Trust-Aware Multi-Patient ICU Healthcare Demo (Windows)" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Start Bedside Monitor on port 8080 in background
Write-Host "[1/2] Starting Jetson Bedside Monitor on http://localhost:8080..." -ForegroundColor Yellow
$bedsideProc = Start-Process python -ArgumentList "jetson_bedside/jetson_edge_brain.py --web-port 8080" -PassThru -NoNewWindow

Start-Sleep -Seconds 2

# 2. Start Central Nurse Station Streamlit Dashboard on port 8501
Write-Host "[2/2] Launching Central Nurse Station on http://localhost:8501..." -ForegroundColor Green
Write-Host "Press Ctrl+C to stop both dashboards." -ForegroundColor Gray
Write-Host ""

try {
    streamlit run nurse_station/dashboard.py
} finally {
    if ($bedsideProc -and -not $bedsideProc.HasExited) {
        Write-Host "Stopping background Bedside Monitor process..." -ForegroundColor Gray
        Stop-Process -Id $bedsideProc.Id -Force -ErrorAction SilentlyContinue
    }
}

