param(
    [string] $AutoRunTaskName = "LotoSystem-AutoRun",
    [string] $WebTaskName = "LotoSystem-Web",
    [int] $AutoRunIntervalHours = 3
)

. "$PSScriptRoot\common.ps1"

$repoRoot = Get-LotoRepoRoot
$python = Get-LotoPython
$currentUser = "$env:USERDOMAIN\$env:USERNAME"
$autoScript = Join-Path $repoRoot "scripts\windows\run-auto-cycle.ps1"
$webScript = Join-Path $repoRoot "scripts\windows\start-server.ps1"

if (-not (Test-Path $autoScript)) {
    throw "Missing automation script: $autoScript"
}
if (-not (Test-Path $webScript)) {
    throw "Missing server script: $webScript"
}

Push-Location $repoRoot
try {
    & $python -m backend.app.research.cli automation-status | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "CLI verification failed."
    }
    & $python -c "import backend.app.main" | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "FastAPI import verification failed."
    }
}
finally {
    Pop-Location
}

Write-Host "Installing scheduled tasks:"
Write-Host "  ${AutoRunTaskName}: every $AutoRunIntervalHours hours, runs auto-run --lottery ALL"
Write-Host "  ${WebTaskName}: at user logon, starts local FastAPI/frontend server"

$autoAction = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$autoScript`""
$autoTrigger = New-ScheduledTaskTrigger `
    -Once `
    -At (Get-Date).Date `
    -RepetitionInterval (New-TimeSpan -Hours $AutoRunIntervalHours)
$autoSettings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 2)

$webAction = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$webScript`""
$webTrigger = New-ScheduledTaskTrigger -AtLogOn -User $currentUser
$webSettings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew

try {
    Register-ScheduledTask -TaskName $AutoRunTaskName -Action $autoAction -Trigger $autoTrigger -Settings $autoSettings -Description "LotoSystem one-shot automation wake-up." -Force | Out-Null
    Register-ScheduledTask -TaskName $WebTaskName -Action $webAction -Trigger $webTrigger -Settings $webSettings -Description "LotoSystem local FastAPI/frontend server." -Force | Out-Null
}
catch {
    Write-Error "Failed to install LotoSystem scheduled tasks: $($_.Exception.Message)"
    throw
}

Write-Host "Scheduled tasks installed or updated."
