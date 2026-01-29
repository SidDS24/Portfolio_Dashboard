@echo off
setlocal
echo ===========================================
echo   PRO-TRADER DASHBOARD: BACKEND LAUNCHER
echo ===========================================
echo.

echo [1/3] Checking Python installation...
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Python is not installed or not in your PATH.
    echo Please install Python from https://python.org
    pause
    exit /b
)

echo [2/3] Ensuring backend dependencies are installed...
echo (This may take a moment on first run)
pip install fastapi uvicorn pandas numpy yfinance scipy PyPortfolioOpt openpyxl python-multipart --quiet

echo [3/3] Starting Backend Server...
echo.
echo !!! KEEP THIS WINDOW OPEN WHILE USING THE DASHBOARD !!!
echo.

python main.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: The backend failed to start.
    echo Common fixes:
    echo 1. Close any other terminal windows running main.py
    echo 2. Try running: pip install fastapi uvicorn
    pause
)

pause
