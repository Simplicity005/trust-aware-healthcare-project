@echo off
python "%~dp0nurse_station\network_broker.py" %*
exit /b %errorlevel%
