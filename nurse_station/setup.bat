@echo off
python "%~dp0setup.py" %*
exit /b %errorlevel%
