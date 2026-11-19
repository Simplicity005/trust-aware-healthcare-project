@echo off
python "%~dp0nurse_station\launch.py" --bedside %*
exit /b %errorlevel%
