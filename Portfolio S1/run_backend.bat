@echo off
cd /d "%~dp0"

:: Start the server and log output
python main.py >> backend.log 2>&1
