# Registers the Stock Ledger morning scan in Windows Task Scheduler (Monday to Friday).
#   powershell -ExecutionPolicy Bypass -File install_daily_task.ps1            # 08:00
#   powershell -ExecutionPolicy Bypass -File install_daily_task.ps1 -Time 07:30
#   powershell -ExecutionPolicy Bypass -File install_daily_task.ps1 -Remove
param([string]$Time = "08:00", [switch]$Remove)

$Name = "StockLedgerDailyScan"
if ($Remove) {
    Unregister-ScheduledTask -TaskName $Name -Confirm:$false -ErrorAction SilentlyContinue
    Write-Output "Removed task $Name."
    exit 0
}

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Here ".venv\Scripts\pythonw.exe"      # pythonw: no console window
if (-not (Test-Path $Python)) { throw "Missing $Python. Run run.bat once first to create the environment." }

$Action = New-ScheduledTaskAction -Execute $Python -Argument "`"$Here\daily_scan.py`"" -WorkingDirectory $Here
$Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At $Time
# StartWhenAvailable: if the PC was off or asleep at that time, run as soon as it is back.
# RunOnlyIfNetworkAvailable: after waking from sleep, wait for the network before starting.
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RunOnlyIfNetworkAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30) -MultipleInstances IgnoreNew
# Runs as you, only while you are signed in (needed for the notification).
$Principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $Name -Action $Action -Trigger $Trigger -Settings $Settings -Principal $Principal `
    -Description "Stock Ledger: morning trend scan and daily ideas" -Force | Out-Null
Write-Output "Registered ${Name}: Monday-Friday at $Time."
