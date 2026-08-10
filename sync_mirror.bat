@echo off
:: ============================================================
:: QC Monitor - rolling local mirror of recent recordings.
:: Copies the last MAXAGE days of .mat recordings from the Tailscale share to
:: fast local disk (MIRROR), then prunes local files older than PRUNE days.
:: The QC daemon + dashboard then read those files locally (mirror.local_first)
:: instead of over the slow VPN.
::
:: Install as a Scheduled Task running AS THE USER ACCOUNT (Tailscale is per-user)
:: every ~30 min -- INDEPENDENT of the QC services so the mirror stays fresh even
:: if a service is down:
::   schtasks /create /tn QCMirrorSync /tr "D:\code\qc_monitor\sync_mirror.bat" ^
::            /sc minute /mo 30 /ru .\%USERNAME% /rp * /rl LIMITED /f
:: (it will prompt for the account password). Verify: schtasks /run /tn QCMirrorSync
:: ============================================================
setlocal
set MIRROR=D:\qc_mirror
set MAXAGE=60
set PRUNE=70
set LOG=D:\code\qc_monitor\logs\mirror_sync.log

if not exist "%MIRROR%" mkdir "%MIRROR%"

echo [%date% %time%] mirror sync start >> "%LOG%"

:: /E subdirs, /MAXAGE only recent files, /XO skip files already current locally,
:: /R:1 /W:5 minimal retries (never hang on a bad file), /MT:8 parallel,
:: /NFL /NDL /NJH /NJS /NP quiet. Robocopy exit codes 0-7 are SUCCESS.
robocopy \\100.106.104.22\database "%MIRROR%\database" /E /MAXAGE:%MAXAGE% /XO /R:1 /W:5 /MT:8 /NFL /NDL /NJH /NJS /NP >> "%LOG%" 2>&1
if %errorlevel% GEQ 8 echo [%date% %time%] WARN robocopy database rc=%errorlevel% >> "%LOG%"

robocopy \\100.106.104.22\bhz "%MIRROR%\bhz" /E /MAXAGE:%MAXAGE% /XO /R:1 /W:5 /MT:8 /NFL /NDL /NJH /NJS /NP >> "%LOG%" 2>&1
if %errorlevel% GEQ 8 echo [%date% %time%] WARN robocopy bhz rc=%errorlevel% >> "%LOG%"

:: Prune local files older than PRUNE days (buffer beyond the copy window) so the
:: mirror stays bounded. Empty dirs are left; harmless.
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-ChildItem -LiteralPath '%MIRROR%' -Recurse -File -ErrorAction SilentlyContinue | Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-%PRUNE%) } | Remove-Item -Force -ErrorAction SilentlyContinue"

echo [%date% %time%] mirror sync done >> "%LOG%"
endlocal
exit /b 0
