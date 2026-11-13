@echo off
title Central Nurse Station Dashboard
cd /d "%~dp0"
echo ======================================================================
echo   Launching Central Nurse Station Dashboard
echo   URL: http://localhost:8501
echo ======================================================================
streamlit run nurse_station\dashboard.py
pause

