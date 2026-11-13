@echo off
title Jetson Bedside ICU Monitor
cd /d "%~dp0"
echo ======================================================================
echo   Launching Jetson Bedside ICU Monitor & Vitals Manipulator
echo   URL: http://localhost:8080
echo ======================================================================
python jetson_bedside\jetson_edge_brain.py --web-port 8080
pause

