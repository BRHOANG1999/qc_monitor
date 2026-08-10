@echo off
:: ============================================================
:: QC Monitor - Windows Service Installer (requires Admin)
:: Installs TWO services so the dashboard runs in its OWN process,
:: isolated from the daemon's GIL-holding .mat reads:
::   QCMonitorDaemon    = main.py --no-dashboard  (ingest/MATLAB/workers)
::   QCMonitorDashboard = dashboard_main.py       (Dash UI under waitress)
:: They share only the SQLite WAL DB. Run from an elevated Command Prompt.
:: ============================================================
setlocal

set PYTHON_EXE=C:\Python313\python.exe
set WORKING_DIR=D:\code\qc_monitor
set DAEMON_SVC=QCMonitorDaemon
set DASH_SVC=QCMonitorDashboard
set NSSM_DIR=D:\code\qc_monitor\tools\nssm
set NSSM_EXE=%NSSM_DIR%\nssm.exe
set NSSM_URL=https://nssm.cc/release/nssm-2.24.zip
set NSSM_ZIP=%NSSM_DIR%\nssm-2.24.zip

:: --- Check for admin privileges ---
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: This script must be run as Administrator.
    echo Right-click Command Prompt and select "Run as administrator".
    pause
    exit /b 1
)

:: --- Download and extract NSSM if not present ---
if not exist "%NSSM_EXE%" (
    echo [1/5] Downloading NSSM...
    if not exist "%NSSM_DIR%" mkdir "%NSSM_DIR%"
    powershell -Command "Invoke-WebRequest -Uri '%NSSM_URL%' -OutFile '%NSSM_ZIP%'"
    if %errorlevel% neq 0 (
        echo ERROR: Failed to download NSSM.
        pause
        exit /b 1
    )
    echo [2/5] Extracting NSSM...
    powershell -Command "Expand-Archive -Path '%NSSM_ZIP%' -DestinationPath '%NSSM_DIR%' -Force"
    copy "%NSSM_DIR%\nssm-2.24\win64\nssm.exe" "%NSSM_EXE%" >nul
    rmdir /s /q "%NSSM_DIR%\nssm-2.24" 2>nul
    del "%NSSM_ZIP%" 2>nul
    echo NSSM downloaded and extracted.
) else (
    echo [1/5] NSSM already present at %NSSM_EXE%.
)

:: --- Remove the OLD single service and any prior split services (re-install) ---
echo [3/5] Removing any existing QC Monitor services...
call :remove_svc QCMonitor
call :remove_svc %DAEMON_SVC%
call :remove_svc %DASH_SVC%

:: --- Install both services ---
echo [4/5] Installing services...
call :install_svc %DAEMON_SVC% "%WORKING_DIR%\main.py --no-dashboard" "QC Monitor Daemon" "24/7 QC ingest + MATLAB + workers (headless)" service_daemon
if %errorlevel% neq 0 goto :fail
call :install_svc %DASH_SVC% "%WORKING_DIR%\dashboard_main.py" "QC Monitor Dashboard" "QC Monitor Dash web UI (own process, waitress)" service_dash
if %errorlevel% neq 0 goto :fail

:: --- Start both services ---
echo [5/5] Starting services...
"%NSSM_EXE%" start %DAEMON_SVC%
"%NSSM_EXE%" start %DASH_SVC%

echo.
echo ======================================================
echo  SUCCESS: %DAEMON_SVC% + %DASH_SVC% installed.
echo ======================================================
echo  Daemon logs : logs\service_daemon_std*.log
echo  Dash logs   : logs\service_dash_std*.log
echo  Manage: nssm ^<start^|stop^|restart^|status^|remove^> %DAEMON_SVC% ^| %DASH_SVC%
echo  Or use services.msc.
echo ======================================================
pause
exit /b 0

:: ------------------------------------------------------------
:: :install_svc  <name>  <"program args">  <displayName>  <desc>  <logPrefix>
:: ------------------------------------------------------------
:install_svc
set _SVC=%~1
set _PARAMS=%~2
set _DISP=%~3
set _DESC=%~4
set _LOGP=%~5
"%NSSM_EXE%" install %_SVC% "%PYTHON_EXE%"
if %errorlevel% neq 0 (
    echo ERROR: Failed to install %_SVC%.
    exit /b 1
)
"%NSSM_EXE%" set %_SVC% AppParameters "%_PARAMS%"
"%NSSM_EXE%" set %_SVC% AppDirectory "%WORKING_DIR%"
"%NSSM_EXE%" set %_SVC% DisplayName "%_DISP%"
"%NSSM_EXE%" set %_SVC% Description "%_DESC%"
"%NSSM_EXE%" set %_SVC% Start SERVICE_AUTO_START
"%NSSM_EXE%" set %_SVC% AppStdout "%WORKING_DIR%\logs\%_LOGP%_stdout.log"
"%NSSM_EXE%" set %_SVC% AppStderr "%WORKING_DIR%\logs\%_LOGP%_stderr.log"
"%NSSM_EXE%" set %_SVC% AppStdoutCreationDisposition 4
"%NSSM_EXE%" set %_SVC% AppStderrCreationDisposition 4
"%NSSM_EXE%" set %_SVC% AppRotateFiles 1
"%NSSM_EXE%" set %_SVC% AppRotateBytes 10485760
"%NSSM_EXE%" set %_SVC% AppRestartDelay 10000
"%NSSM_EXE%" set %_SVC% AppStopMethodSkip 0
"%NSSM_EXE%" set %_SVC% AppStopMethodConsole 30000
echo   installed %_SVC%.
exit /b 0

:: ------------------------------------------------------------
:: :remove_svc  <name>   (no-op if absent)
:: ------------------------------------------------------------
:remove_svc
"%NSSM_EXE%" status %~1 >nul 2>&1
if %errorlevel% equ 0 (
    echo   removing existing %~1...
    "%NSSM_EXE%" stop %~1 >nul 2>&1
    "%NSSM_EXE%" remove %~1 confirm >nul 2>&1
)
exit /b 0

:fail
echo ERROR: service install failed. See messages above.
pause
exit /b 1
