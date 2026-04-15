@echo off
:: ============================================================
:: QC Monitor — Windows Service Uninstaller (requires Admin)
:: ============================================================
setlocal

set SERVICE_NAME=QCMonitor
set NSSM_EXE=D:\code\qc_monitor\tools\nssm\nssm.exe

net session >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: This script must be run as Administrator.
    pause
    exit /b 1
)

if not exist "%NSSM_EXE%" (
    echo ERROR: NSSM not found at %NSSM_EXE%.
    pause
    exit /b 1
)

echo Stopping %SERVICE_NAME%...
"%NSSM_EXE%" stop %SERVICE_NAME% >nul 2>&1

echo Removing %SERVICE_NAME%...
"%NSSM_EXE%" remove %SERVICE_NAME% confirm
if %errorlevel% equ 0 (
    echo Service removed successfully.
) else (
    echo Failed to remove service (may not be installed).
)

pause
