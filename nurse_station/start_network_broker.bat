@echo off
python "%~dp0network_broker.py" %*
exit /b %errorlevel%
