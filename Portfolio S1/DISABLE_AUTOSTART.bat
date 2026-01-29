@echo off
echo ===========================================
echo   PRO-TRADER DASHBOARD: DISABLE AUTO-START
echo ===========================================
echo.

echo Removing scheduled task...
schtasks /delete /tn "ProTraderBackend" /f

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Auto-start has been disabled.
    echo The backend will no longer start automatically at login.
) else (
    echo.
    echo Note: Task may not exist or already removed.
)

echo.
pause
