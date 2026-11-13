@echo off
title Trust-Aware Healthcare Local Demo
cd /d "%~dp0"
echo ======================================================================
echo   Trust-Aware Multi-Patient ICU Healthcare Demo (Windows)
echo ======================================================================

echo [1/2] Launching Bedside ICU Monitor in background on http://localhost:8080...
start "Bedside ICU Monitor" /min python jetson_bedside\jetson_edge_brain.py --web-port 8080

timeout /t 2 /nobreak >nul

echo [2/2] Launching Central Nurse Station Dashboard on http://localhost:8501...
echo.
streamlit run nurse_station\dashboard.py

