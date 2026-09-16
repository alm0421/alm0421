@echo off
REM ===========================================================================
REM  Launch the Streamlit operator dashboard (opens on http://localhost:8501).
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

echo Starting dashboard on http://localhost:8501 ^(Ctrl+C to stop^)...
call ".venv\Scripts\python.exe" "scripts\run_dashboard.py" %*
set "EXIT_CODE=%ERRORLEVEL%"

if %EXIT_CODE% NEQ 0 (
    echo.
    echo [FAILED] The dashboard exited with code %EXIT_CODE%. Check logs\platform.log
)

popd
endlocal & exit /b %EXIT_CODE%
