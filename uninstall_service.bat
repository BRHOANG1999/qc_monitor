@echo off
:: ============================================================
:: QC Monitor - Windows Service Uninstaller (requires Admin)
:: Removes both split services (and the legacy single service).
:: ============================================================
setlocal

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

call :remove_svc QCMonitorDashboard
call :remove_svc QCMonitorDaemon
call :remove_svc QCMonitor

echo Done.
pause
exit /b 0

:remove_svc
"%NSSM_EXE%" status %~1 >nul 2>&1
if %errorlevel% equ 0 (
    echo Stopping and removing %~1...
    "%NSSM_EXE%" stop %~1 >nul 2>&1
    "%NSSM_EXE%" remove %~1 confirm
) else (
    echo %~1 not installed - skipping.
)
exit /b 0
