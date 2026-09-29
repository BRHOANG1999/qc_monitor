# Install the QC Monitor health watchdog as a Windows Scheduled Task.
#
# The watchdog must run independently of QCMonitorDaemon (so it can restart the daemon
# when THAT is down) and with elevation (so it can restart services). This registers a
# task that runs every 10 minutes as SYSTEM with highest privileges.
#
# Run this ONCE from an ADMINISTRATOR PowerShell:
#   powershell -ExecutionPolicy Bypass -File tools\install_health_watchdog.ps1
#
# Uninstall:  Unregister-ScheduledTask -TaskName 'QCMonitorHealthWatchdog' -Confirm:$false

param(
    [string]$RepoRoot = "D:\code\qc_monitor",
    [string]$Python   = "",                       # auto-detected if empty
    [int]$IntervalMinutes = 10,
    [string]$TaskName = "QCMonitorHealthWatchdog"
)

$ErrorActionPreference = "Stop"

if (-not $Python) {
    $Python = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $Python) { throw "python not found on PATH; pass -Python <path>" }
}
if (-not (Test-Path $RepoRoot)) { throw "RepoRoot not found: $RepoRoot" }

Write-Host "Repo   : $RepoRoot"
Write-Host "Python : $Python"
Write-Host "Every  : $IntervalMinutes min"

# -WorkingDirectory makes the module's relative paths (config/, data/, logs/) resolve.
$action = New-ScheduledTaskAction -Execute $Python `
    -Argument "-m src.health.watchdog" -WorkingDirectory $RepoRoot

# Repeat forever, starting a minute from now; the task is short-lived each run.
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes $IntervalMinutes)

# SYSTEM + highest privileges so it can Restart-Service without a prompt.
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" `
    -LogonType ServiceAccount -RunLevel Highest

$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 9) `
    -RestartCount 2 -RestartInterval (New-TimeSpan -Minutes 1)

if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Write-Host "Replacing existing task '$TaskName'..."
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings `
    -Description "QC Monitor 24/7 health watchdog: checks services, email jobs, pipeline, disk/log/memory; bounded auto-restart + email report." | Out-Null

Write-Host "Installed '$TaskName'. Running one check now to verify..."
Start-ScheduledTask -TaskName $TaskName
Start-Sleep -Seconds 3
Get-ScheduledTask -TaskName $TaskName |
    Select-Object TaskName, State |
    Format-Table -AutoSize
Write-Host "Done. Logs: watch data\health_state.json and the watchdog email."
