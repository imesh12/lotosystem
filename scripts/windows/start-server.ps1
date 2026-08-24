param(
    [string] $HostName = "127.0.0.1",
    [int] $Port = 8000
)

. "$PSScriptRoot\common.ps1"

$repoRoot = Get-LotoRepoRoot
$python = Get-LotoPython

if (Test-LotoApiHealth -HostName $HostName -Port $Port) {
    Write-Host "LotoSystem API is already reachable at http://${HostName}:${Port}"
    exit 0
}

$logPath = New-LotoLogPath -Prefix "server"
$errorLogPath = New-LotoLogPath -Prefix "server-error"
Remove-OldLotoLogs -Pattern "server-*.log" -Retain 30
Remove-OldLotoLogs -Pattern "server-error-*.log" -Retain 30

Write-Host "Starting LotoSystem API at http://${HostName}:${Port}"
Write-Host "Log: $logPath"
Write-Host "Error log: $errorLogPath"

$process = Start-Process `
    -FilePath $python `
    -ArgumentList @("-m", "uvicorn", "backend.app.main:app", "--host", $HostName, "--port", "$Port") `
    -WorkingDirectory $repoRoot `
    -RedirectStandardOutput $logPath `
    -RedirectStandardError $errorLogPath `
    -WindowStyle Hidden `
    -PassThru

for ($attempt = 1; $attempt -le 15; $attempt++) {
    Start-Sleep -Seconds 1
    if (Test-LotoApiHealth -HostName $HostName -Port $Port) {
        Write-Host "LotoSystem API started with process id $($process.Id)"
        exit 0
    }
}

if (-not $process.HasExited) {
    Stop-Process -Id $process.Id -Force
}
throw "LotoSystem API did not become healthy at http://${HostName}:${Port}/api/health"
