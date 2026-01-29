@echo off
:: LAUNCH_DASHBOARD.bat
:: Starts the backend silently and opens the dashboard

echo ===========================================
echo   PRO-TRADER DASHBOARD: LAUNCHER
echo ===========================================
echo.

:: 1. Start backend silently
echo [1/2] Starting backend in background...
wscript.exe "START_BACKEND_SILENT.vbs"

:: 2. Wait for initialization
echo [2/2] Opening dashboard...
timeout /t 3 /nobreak >nul

:: 3. Open the site
start "" "index.html"

echo.
echo All components launched!
timeout /t 2 >nul
exit
