@echo off
setlocal

echo ===========================================
echo   PRO-TRADER DASHBOARD: AUTO-START SETUP
echo ===========================================
echo.
echo This will configure Windows to automatically start 
echo the backend server when you log in.
echo.

:: Get the current directory
set "SCRIPT_DIR=%~dp0"
set "VBS_PATH=%SCRIPT_DIR%START_BACKEND_SILENT.vbs"

echo [1/3] Removing any existing task...
schtasks /delete /tn "ProTraderBackend" /f >nul 2>&1

echo [2/3] Creating scheduled task to run at login...
schtasks /create /tn "ProTraderBackend" /tr "wscript.exe \"%VBS_PATH%\"" /sc onlogon /rl highest /f

if %ERRORLEVEL% EQU 0 (
    echo.
    echo =========================================
    echo   SUCCESS! Auto-start is now configured.
    echo =========================================
    echo.
    echo The backend will start automatically when you log in.
    echo.
    echo [3/3] Starting backend now for immediate use...
    wscript.exe "%VBS_PATH%"
    
    :: Wait a moment for server to start
    timeout /t 3 /nobreak >nul
    
    echo.
    echo Backend should now be running!
    echo You can verify by visiting: http://127.0.0.1:8000/health
    echo.
    echo USEFUL COMMANDS:
    echo   - To stop the backend:  Run STOP_BACKEND.bat
    echo   - To disable auto-start: Run DISABLE_AUTOSTART.bat
    echo.
) else (
    echo.
    echo ERROR: Failed to create scheduled task.
    echo Please run this script as Administrator.
    echo.
)

pause
