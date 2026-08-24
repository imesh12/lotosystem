param(
    [string] $AutoRunTaskName = "LotoSystem-AutoRun",
    [string] $WebTaskName = "LotoSystem-Web"
)

$ErrorActionPreference = "Stop"

foreach ($taskName in @($AutoRunTaskName, $WebTaskName)) {
    $task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    if ($null -eq $task) {
        Write-Host "Scheduled task not found: $taskName"
        continue
    }
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Host "Removed scheduled task: $taskName"
}
