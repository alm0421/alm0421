<#
.SYNOPSIS
    Launch the Streamlit operator dashboard.

.DESCRIPTION
    PowerShell equivalent of run_dashboard.bat, for environments where .bat
    files are restricted or where richer error reporting is wanted.

.EXAMPLE
    .\run_dashboard.ps1
    .\run_dashboard.ps1 -Port 8502

.NOTES
    If script execution is blocked, run PowerShell as the current user with:
        Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
#>
[CmdletBinding()]
param(
    [int]$Port = 8501
)

$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot

try {
    $VenvPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path $VenvPython)) {
        Write-Error "Virtual environment not found at $VenvPython. Run launchers\setup.bat first."
        exit 1
    }

    foreach ($dir in @('logs', 'outputs')) {
        $path = Join-Path $ProjectRoot $dir
        if (-not (Test-Path $path)) {
            New-Item -ItemType Directory -Path $path -Force | Out-Null
        }
    }

    Write-Host "Starting dashboard on http://localhost:$Port (Ctrl+C to stop)..." -ForegroundColor Cyan
    & $VenvPython (Join-Path $ProjectRoot 'scripts\run_dashboard.py') '--server.port' $Port
    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0) {
        Write-Host "The dashboard exited with code $exitCode. Check logs\platform.log" -ForegroundColor Red
    }
    exit $exitCode
}
finally {
    Pop-Location
}
