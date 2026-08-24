param(
    [string] $HostName = "127.0.0.1",
    [int] $Port = 8000
)

. "$PSScriptRoot\common.ps1"

$repoRoot = Get-LotoRepoRoot
$pythonAvailable = $false
try {
    $python = Get-LotoPython
    $pythonAvailable = Test-Path $python
}
catch {
    $python = $null
}

$apiReachable = Test-LotoApiHealth -HostName $HostName -Port $Port
$automationStatus = $null
$notificationStatus = $null
if ($pythonAvailable) {
    $automationStatus = (& $python -m backend.app.research.cli --lottery ALL automation-status 2>&1) -join "`n"
    $notificationStatus = (& $python -m backend.app.research.cli notification-status 2>&1) -join "`n"
}

$tasks = @()
foreach ($taskName in @("LotoSystem-AutoRun", "LotoSystem-Web")) {
    $task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    if ($null -ne $task) {
        $tasks += @{
            name = $task.TaskName
            state = "$($task.State)"
        }
    }
}

$latestRun = $null
$runDirectory = Join-Path $repoRoot "data\automation\runs"
if (Test-Path $runDirectory) {
    $latest = Get-ChildItem -Path $runDirectory -Filter "*.json" -File |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -ne $latest) {
        $latestRun = $latest.Name
    }
}

[ordered]@{
    repository_path = $repoRoot
    python_venv_available = $pythonAvailable
    api_server_reachable = $apiReachable
    api_url = "http://${HostName}:${Port}"
    scheduled_tasks = $tasks
    latest_automation_run = $latestRun
    automation_status = $automationStatus
    notification_status = $notificationStatus
} | ConvertTo-Json -Depth 8
