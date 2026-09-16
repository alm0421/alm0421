@echo off
REM ===========================================================================
REM  One-time setup: create a virtual environment and install dependencies.
REM  Run this first, from anywhere. Paths are resolved relative to this file.
REM ===========================================================================
setlocal EnableDelayedExpansion

set "PROJECT_ROOT=%~dp0.."
pushd "%PROJECT_ROOT%" || (echo [ERROR] Could not enter project directory. & exit /b 1)

echo ============================================================
echo   Trading Platform - Setup
echo   Project: %CD%
echo ============================================================
echo.

where py >nul 2>&1
if errorlevel 1 (
    where python >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Python was not found on PATH.
        echo         Install Python 3.11 or newer from https://www.python.org/downloads/
        echo         and tick "Add python.exe to PATH" during installation.
        popd & exit /b 1
    )
    set "PY_CMD=python"
) else (
    set "PY_CMD=py -3"
)

echo [1/4] Using: !PY_CMD!
!PY_CMD! --version

if not exist ".venv\Scripts\python.exe" (
    echo [2/4] Creating virtual environment in .venv ...
    !PY_CMD! -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create the virtual environment.
        popd & exit /b 1
    )
) else (
    echo [2/4] Virtual environment already exists - reusing it.
)

echo [3/4] Installing dependencies ...
call ".venv\Scripts\python.exe" -m pip install --upgrade pip
call ".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Dependency installation failed. See the output above.
    popd & exit /b 1
)

echo [4/4] Creating runtime directories ...
if not exist "logs"    mkdir "logs"
if not exist "outputs" mkdir "outputs"

if not exist ".env" (
    if exist ".env.example" (
        copy /Y ".env.example" ".env" >nul
        echo.
        echo [ACTION REQUIRED] A .env file was created from .env.example.
        echo                   Edit it and add your Alpaca PAPER API keys.
        echo                   .env is gitignored and must never be committed.
    )
)

echo.
echo ============================================================
echo   Setup complete.
echo.
echo   Next:  launchers\preflight.bat        (validate everything)
echo          launchers\run_scanner.bat      (generate signals)
echo          launchers\run_dashboard.bat    (open the dashboard)
echo ============================================================
popd
endlocal
exit /b 0
