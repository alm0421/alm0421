@echo off
REM ===========================================================================
REM  Non-destructive preflight check. Places no orders.
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

call ".venv\Scripts\python.exe" "scripts\preflight.py" %*
set "EXIT_CODE=%ERRORLEVEL%"

if %EXIT_CODE% NEQ 0 (
    echo.
    echo [FAILED] Preflight reported problems ^(exit code %EXIT_CODE%^).
) 

popd
endlocal & exit /b %EXIT_CODE%
