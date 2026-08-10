@echo off
:: ============================================================
:: QC Monitor - Windows Service Installer (requires Admin)
:: Installs TWO services so the dashboard runs in its OWN process,
:: isolated from the daemon's GIL-holding .mat reads:
::   QCMonitorDaemon    = main.py --no-dashboard  (ingest/MATLAB/workers)
::   QCMonitorDashboard = dashboard_main.py       (Dash UI under waitress)
:: They share only the SQLite WAL DB. Run from an elevated Command Prompt.
::
:: IMPORTANT - LOG-ON ACCOUNT: the app's Python packages live in the USER
:: site-packages and the recording share is reached over per-user Tailscale, so
:: the services MUST log on as your Windows account (NOT LocalSystem, which
:: cannot see either). You will be prompted for the account + password below.
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

:: --- Log-on account (required so the services can see user-site packages +
::     the Tailscale share). Leave blank ONLY if you will set it later via
::     services.msc. Password is typed in plain text at this local console. ---
set SVC_ACCOUNT=
set SVC_PASSWORD=
set /p SVC_ACCOUNT=Run services as which account? (e.g. .\%USERNAME% or DOMAIN\%USERNAME%):
if defined SVC_ACCOUNT set /p SVC_PASSWORD=Password for %SVC_ACCOUNT%:

:: --- Download and extract NSSM if not present ---
if not exist "%NSSM_EXE%" (
    echo [1/5] Downloading NSSM...
    if not exist "%NSSM_DIR%" mkdir "%NSSM_DIR%"
    powershell -Command "Invoke-WebRequest -Uri '%NSSM_URL%' -OutFile '%NSSM_ZIP%'"
    if %errorlevel% neq 0 ( echo ERROR: Failed to download NSSM. & pause & exit /b 1 )
    echo [2/5] Extracting NSSM...
    powershell -Command "Expand-Archive -Path '%NSSM_ZIP%' -DestinationPath '%NSSM_DIR%' -Force"
    copy "%NSSM_DIR%\nssm-2.24\win64\nssm.exe" "%NSSM_EXE%" >nul
    rmdir /s /q "%NSSM_DIR%\nssm-2.24" 2>nul
    del "%NSSM_ZIP%" 2>nul
) else (
    echo [1/5] NSSM already present at %NSSM_EXE%.
)

:: --- Remove the OLD single service and any prior split services ---
echo [3/5] Removing any existing QC Monitor services...
call :remove_svc QCMonitor
call :remove_svc %DAEMON_SVC%
call :remove_svc %DASH_SVC%

:: --- Install DAEMON (main.py --no-dashboard) ---
echo [4/5] Installing %DAEMON_SVC%...
"%NSSM_EXE%" install %DAEMON_SVC% "%PYTHON_EXE%"
"%NSSM_EXE%" set %DAEMON_SVC% AppParameters "%WORKING_DIR%\main.py --no-dashboard"
"%NSSM_EXE%" set %DAEMON_SVC% AppDirectory "%WORKING_DIR%"
"%NSSM_EXE%" set %DAEMON_SVC% DisplayName "QC Monitor Daemon"
"%NSSM_EXE%" set %DAEMON_SVC% Description "QC ingest MATLAB and workers, headless"
"%NSSM_EXE%" set %DAEMON_SVC% Start SERVICE_AUTO_START
"%NSSM_EXE%" set %DAEMON_SVC% AppStdout "%WORKING_DIR%\logs\service_daemon_stdout.log"
"%NSSM_EXE%" set %DAEMON_SVC% AppStderr "%WORKING_DIR%\logs\service_daemon_stderr.log"
"%NSSM_EXE%" set %DAEMON_SVC% AppRotateFiles 1
"%NSSM_EXE%" set %DAEMON_SVC% AppRotateBytes 10485760
"%NSSM_EXE%" set %DAEMON_SVC% AppRestartDelay 10000
"%NSSM_EXE%" set %DAEMON_SVC% AppStopMethodSkip 0
"%NSSM_EXE%" set %DAEMON_SVC% AppStopMethodConsole 30000
if defined SVC_ACCOUNT "%NSSM_EXE%" set %DAEMON_SVC% ObjectName "%SVC_ACCOUNT%" "%SVC_PASSWORD%"

:: --- Install DASHBOARD (dashboard_main.py) ---
echo Installing %DASH_SVC%...
"%NSSM_EXE%" install %DASH_SVC% "%PYTHON_EXE%"
"%NSSM_EXE%" set %DASH_SVC% AppParameters "%WORKING_DIR%\dashboard_main.py"
"%NSSM_EXE%" set %DASH_SVC% AppDirectory "%WORKING_DIR%"
"%NSSM_EXE%" set %DASH_SVC% DisplayName "QC Monitor Dashboard"
"%NSSM_EXE%" set %DASH_SVC% Description "QC Monitor Dash web UI, own process, waitress"
"%NSSM_EXE%" set %DASH_SVC% Start SERVICE_AUTO_START
"%NSSM_EXE%" set %DASH_SVC% AppStdout "%WORKING_DIR%\logs\service_dash_stdout.log"
"%NSSM_EXE%" set %DASH_SVC% AppStderr "%WORKING_DIR%\logs\service_dash_stderr.log"
"%NSSM_EXE%" set %DASH_SVC% AppRotateFiles 1
"%NSSM_EXE%" set %DASH_SVC% AppRotateBytes 10485760
"%NSSM_EXE%" set %DASH_SVC% AppRestartDelay 10000
"%NSSM_EXE%" set %DASH_SVC% AppStopMethodSkip 0
"%NSSM_EXE%" set %DASH_SVC% AppStopMethodConsole 30000
if defined SVC_ACCOUNT "%NSSM_EXE%" set %DASH_SVC% ObjectName "%SVC_ACCOUNT%" "%SVC_PASSWORD%"

:: --- Start both ---
echo [5/5] Starting services...
"%NSSM_EXE%" start %DAEMON_SVC%
"%NSSM_EXE%" start %DASH_SVC%

echo.
echo ======================================================
echo  Installed %DAEMON_SVC% + %DASH_SVC%.
if not defined SVC_ACCOUNT echo  WARNING: no account set - services run as LocalSystem and WILL fail
if not defined SVC_ACCOUNT echo  ^(no user packages / no Tailscale share^). Set Log On in services.msc.
echo  Daemon logs : logs\service_daemon_std*.log
echo  Dash logs   : logs\service_dash_std*.log
echo  Manage: nssm ^<start^|stop^|restart^|status^> %DAEMON_SVC% ^| %DASH_SVC%
echo ======================================================
pause
exit /b 0

:remove_svc
"%NSSM_EXE%" status %~1 >nul 2>&1
if %errorlevel% equ 0 (
    echo   removing existing %~1...
    "%NSSM_EXE%" stop %~1 >nul 2>&1
    "%NSSM_EXE%" remove %~1 confirm >nul 2>&1
)
exit /b 0
