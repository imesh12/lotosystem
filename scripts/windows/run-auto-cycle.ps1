param(
    [ValidateSet("ALL", "LOTO6", "MINI_LOTO")]
    [string] $Lottery = "ALL",
    [int] $Seed = 123456,
    [int] $TicketsPerDraw = 3,
    [switch] $Headed
)

. "$PSScriptRoot\common.ps1"

$logPath = New-LotoLogPath -Prefix "automation"
Remove-OldLotoLogs -Pattern "automation-*.log" -Retain 30

$arguments = @(
    "--lottery", $Lottery,
    "--seed", "$Seed",
    "--tickets-per-draw", "$TicketsPerDraw"
)
if ($Headed) {
    $arguments += "--headed"
}
$arguments += "auto-run"

Write-Host "Running LotoSystem auto-run for $Lottery"
Write-Host "Log: $logPath"

$repoRoot = Get-LotoRepoRoot
$python = Get-LotoPython
Push-Location $repoRoot
try {
    & $python -m backend.app.research.cli @arguments *>&1 |
        Tee-Object -FilePath $logPath
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
