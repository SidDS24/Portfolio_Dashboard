@echo off
echo ===========================================
echo   PRO-TRADER DASHBOARD: STOPPING BACKEND
echo ===========================================
echo.

echo Stopping any running Python backend processes...

:: Kill pythonw processes (silent Python)
taskkill /f /im pythonw.exe >nul 2>&1

:: Kill python processes running on port 8000
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do (
    echo Killing process %%a on port 8000...
    taskkill /f /pid %%a >nul 2>&1
)

echo.
echo Backend has been stopped.
echo.
pause
