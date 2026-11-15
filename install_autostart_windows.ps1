$ErrorActionPreference = "Stop"
$dir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$bat  = Join-Path $dir "start_darts.bat"
$port = 8000
if (-not (Test-Path $bat)) { throw "Nem talalom: $bat" }

$action    = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c start `"`" /min `"$bat`"" -WorkingDirectory $dir
$trigger   = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings  = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
             -RestartCount 5 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero)
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName "BullsEye-CV" -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null
Write-Host "Feladat letrehozva: BullsEye-CV (bejelentkezeskor indul)"

Get-NetFirewallRule -DisplayName "Darts web" -ErrorAction SilentlyContinue | Remove-NetFirewallRule
New-NetFirewallRule -DisplayName "Darts web" -Direction Inbound -Protocol TCP -LocalPort $port -Action Allow -RemoteAddress LocalSubnet | Out-Null
Write-Host "Tuzfal-szabaly letrehozva (TCP $port, csak helyi halozat)"

powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /change monitor-timeout-ac 0
Write-Host "Alvas/hibernalas kikapcsolva (halozati taplalas)"

$ip = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike "169.*" -and $_.IPAddress -ne "127.0.0.1" } | Select-Object -First 1).IPAddress
Write-Host ""
Write-Host "Kesz. Telefonrol:  http://${ip}:$port"
Write-Host "Teszt: inditsd ujra a gepet, vagy:  Start-ScheduledTask -TaskName BullsEye-CV"
Write-Host "FONTOS: a gepnek automatikusan be kell jelentkeznie (netplwiz), kulonben a feladat nem indul."
