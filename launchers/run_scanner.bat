@echo off
REM ===========================================================================
REM  Run the scan -> signal pipeline.
REM
REM    run_scanner.bat                 one cycle, signals only
REM    run_scanner.bat --loop          run continuously
REM    run_scanner.bat --execute       permit order placement (mode still decides)
REM ===========================================================================
setlocal

set "PROJECT_ROOT=%~dp0.."
pushd "%PROJECT_ROOT%" || (echo [ERROR] Could not enter project directory. & exit /b 1)

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found.
    echo         Run launchers\setup.bat first.
    popd & exit /b 1
)

if not exist "logs"    mkdir "logs"
if not exist "outputs" mkdir "outputs"

echo Starting scanner ^(Ctrl+C to stop^)...
call ".venv\Scripts\python.exe" "scripts\run_scanner.py" %*
set "EXIT_CODE=%ERRORLEVEL%"

if %EXIT_CODE% NEQ 0 (
    echo.
    echo [FAILED] The scanner exited with code %EXIT_CODE%. Check logs\platform.log
)

popd
endlocal & exit /b %EXIT_CODE%
