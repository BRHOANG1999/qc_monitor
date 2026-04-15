@echo off
:: ============================================================
:: QC Monitor — Windows Service Installer (requires Admin)
:: Run this script from an elevated Command Prompt.
:: ============================================================
setlocal

set SERVICE_NAME=QCMonitor
set PYTHON_EXE=C:\Python313\python.exe
set SCRIPT_PATH=D:\code\qc_monitor\main.py
set WORKING_DIR=D:\code\qc_monitor
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
    echo [1/4] Downloading NSSM...
    if not exist "%NSSM_DIR%" mkdir "%NSSM_DIR%"
    powershell -Command "Invoke-WebRequest -Uri '%NSSM_URL%' -OutFile '%NSSM_ZIP%'"
    if %errorlevel% neq 0 (
        echo ERROR: Failed to download NSSM.
        pause
        exit /b 1
    )
    echo [2/4] Extracting NSSM...
    powershell -Command "Expand-Archive -Path '%NSSM_ZIP%' -DestinationPath '%NSSM_DIR%' -Force"
    :: Copy the 64-bit exe to our tools dir
    copy "%NSSM_DIR%\nssm-2.24\win64\nssm.exe" "%NSSM_EXE%" >nul
    :: Clean up
    rmdir /s /q "%NSSM_DIR%\nssm-2.24" 2>nul
    del "%NSSM_ZIP%" 2>nul
    echo NSSM downloaded and extracted.
) else (
    echo [1/4] NSSM already present at %NSSM_EXE%.
)

:: --- Remove existing service if present (for re-installs) ---
"%NSSM_EXE%" status %SERVICE_NAME% >nul 2>&1
if %errorlevel% equ 0 (
    echo Removing existing %SERVICE_NAME% service...
    "%NSSM_EXE%" stop %SERVICE_NAME% >nul 2>&1
    "%NSSM_EXE%" remove %SERVICE_NAME% confirm
)

:: --- Install the service ---
echo [3/4] Installing service "%SERVICE_NAME%"...
"%NSSM_EXE%" install %SERVICE_NAME% "%PYTHON_EXE%" "%SCRIPT_PATH%"
if %errorlevel% neq 0 (
    echo ERROR: Failed to install service.
    pause
    exit /b 1
)

:: --- Configure service parameters ---
"%NSSM_EXE%" set %SERVICE_NAME% AppDirectory "%WORKING_DIR%"
"%NSSM_EXE%" set %SERVICE_NAME% DisplayName "QC Monitor Daemon"
"%NSSM_EXE%" set %SERVICE_NAME% Description "24/7 offline QC analysis daemon for KMrecorder neurophysiology data"
"%NSSM_EXE%" set %SERVICE_NAME% Start SERVICE_AUTO_START
"%NSSM_EXE%" set %SERVICE_NAME% AppStdout "%WORKING_DIR%\logs\service_stdout.log"
"%NSSM_EXE%" set %SERVICE_NAME% AppStderr "%WORKING_DIR%\logs\service_stderr.log"
"%NSSM_EXE%" set %SERVICE_NAME% AppStdoutCreationDisposition 4
"%NSSM_EXE%" set %SERVICE_NAME% AppStderrCreationDisposition 4
"%NSSM_EXE%" set %SERVICE_NAME% AppRotateFiles 1
"%NSSM_EXE%" set %SERVICE_NAME% AppRotateBytes 10485760
:: Auto-restart on failure with 10s delay
"%NSSM_EXE%" set %SERVICE_NAME% AppRestartDelay 10000
:: Graceful shutdown: send Ctrl+C first, wait 30s before killing
"%NSSM_EXE%" set %SERVICE_NAME% AppStopMethodSkip 0
"%NSSM_EXE%" set %SERVICE_NAME% AppStopMethodConsole 30000

:: --- Start the service ---
echo [4/4] Starting service...
"%NSSM_EXE%" start %SERVICE_NAME%
if %errorlevel% equ 0 (
    echo.
    echo ======================================================
    echo  SUCCESS: QCMonitor service is installed and running!
    echo ======================================================
    echo.
    echo  The service will:
    echo    - Start automatically at Windows boot
    echo    - Auto-restart on crash (10s delay)
    echo    - Log stdout/stderr to logs\service_*.log
    echo.
    echo  Manage it with:
    echo    nssm start QCMonitor
    echo    nssm stop QCMonitor
    echo    nssm restart QCMonitor
    echo    nssm status QCMonitor
    echo    nssm remove QCMonitor confirm   (uninstall)
    echo.
    echo  Or use Windows Services panel (services.msc).
    echo ======================================================
) else (
    echo WARNING: Service installed but failed to start.
    echo Check logs\service_stderr.log for details.
)

pause
