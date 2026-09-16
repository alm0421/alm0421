<#
.SYNOPSIS
    Run the non-destructive preflight check. Places no orders.

.EXAMPLE
    .\preflight.ps1
    .\preflight.ps1 -SkipNetwork
    .\preflight.ps1 -VerifySymbols
#>
[CmdletBinding()]
param(
    [switch]$SkipNetwork,
    [switch]$VerifySymbols
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

    $arguments = @((Join-Path $ProjectRoot 'scripts\preflight.py'))
    if ($SkipNetwork)   { $arguments += '--skip-network' }
    if ($VerifySymbols) { $arguments += '--verify-symbols' }

    & $VenvPython @arguments
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
