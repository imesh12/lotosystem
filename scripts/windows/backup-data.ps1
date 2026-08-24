param(
    [string] $BackupDirectory = ""
)

. "$PSScriptRoot\common.ps1"

$repoRoot = Get-LotoRepoRoot
if (-not $BackupDirectory) {
    $BackupDirectory = Join-Path $repoRoot "backups"
}
New-Item -ItemType Directory -Force -Path $BackupDirectory | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$staging = Join-Path ([System.IO.Path]::GetTempPath()) "lotosystem-backup-$timestamp"
$archivePath = Join-Path $BackupDirectory "lotosystem-data-$timestamp.zip"

New-Item -ItemType Directory -Force -Path $staging | Out-Null
try {
    $items = @(
        "config\operational_settings.json",
        "data\processed",
        "data\predictions",
        "data\settlements",
        "data\notifications",
        "data\automation"
    )
    foreach ($item in $items) {
        $source = Join-Path $repoRoot $item
        if (-not (Test-Path $source)) {
            continue
        }
        $destination = Join-Path $staging $item
        $destinationParent = Split-Path $destination -Parent
        New-Item -ItemType Directory -Force -Path $destinationParent | Out-Null
        Copy-Item -Path $source -Destination $destination -Recurse -Force
    }
    Compress-Archive -Path (Join-Path $staging "*") -DestinationPath $archivePath -Force
    Write-Host "Backup created: $archivePath"
}
finally {
    Remove-Item -Path $staging -Recurse -Force -ErrorAction SilentlyContinue
}
