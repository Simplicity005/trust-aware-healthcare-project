@echo off
python "%~dp0nurse_station\launch.py" --demo %*
exit /b %errorlevel%
