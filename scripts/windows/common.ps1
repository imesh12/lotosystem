Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-LotoRepoRoot {
    return (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}

function Get-LotoPython {
    $repoRoot = Get-LotoRepoRoot
    $python = Join-Path $repoRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path $python)) {
        throw "Missing virtual environment Python: $python. Run the project setup first."
    }
    return $python
}

function Initialize-LotoLogDirectory {
    $repoRoot = Get-LotoRepoRoot
    $logDirectory = Join-Path $repoRoot "data\logs"
    New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
    return $logDirectory
}

function New-LotoLogPath {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Prefix
    )
    $logDirectory = Initialize-LotoLogDirectory
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    return Join-Path $logDirectory "$Prefix-$timestamp.log"
}

function Remove-OldLotoLogs {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Pattern,
        [int] $Retain = 30
    )
    $logDirectory = Initialize-LotoLogDirectory
    Get-ChildItem -Path $logDirectory -Filter $Pattern -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -Skip $Retain |
        Remove-Item -Force
}

function Invoke-LotoCli {
    param(
        [Parameter(Mandatory = $true)]
        [string[]] $Arguments
    )
    $repoRoot = Get-LotoRepoRoot
    $python = Get-LotoPython
    Push-Location $repoRoot
    try {
        & $python -m backend.app.research.cli @Arguments
        return $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
}

function Test-LotoApiHealth {
    param(
        [string] $HostName = "127.0.0.1",
        [int] $Port = 8000
    )
    $uri = "http://${HostName}:${Port}/api/health"
    try {
        $response = Invoke-WebRequest -Uri $uri -UseBasicParsing -TimeoutSec 5
        return ($response.StatusCode -eq 200)
    }
    catch {
        return $false
    }
}
