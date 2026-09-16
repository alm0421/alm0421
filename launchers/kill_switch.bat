@echo off
REM ===========================================================================
REM  Engage or release the kill switch.
REM
REM  While engaged, EVERY order is rejected by the risk engine before it can
REM  reach a broker. This works without restarting anything.
REM
REM    kill_switch.bat on     engage  (halt all order submission)
REM    kill_switch.bat off    release
REM    kill_switch.bat        show current state
REM ===========================================================================
setlocal

set "PROJECT_ROOT=%~dp0.."
pushd "%PROJECT_ROOT%" || (echo [ERROR] Could not enter project directory. & exit /b 1)

set "SWITCH=%CD%\logs\KILL_SWITCH"
if not exist "logs" mkdir "logs"

if /I "%~1"=="on" goto :engage
if /I "%~1"=="off" goto :release
goto :status

:engage
echo Engaged by %USERNAME% at %DATE% %TIME% > "%SWITCH%"
echo [KILL SWITCH ENGAGED]
echo All order submission is now halted.
echo File: %SWITCH%
goto :end

:release
if exist "%SWITCH%" del /Q "%SWITCH%"
echo [KILL SWITCH RELEASED]
echo Order submission may resume, subject to all other risk checks.
goto :end

:status
if exist "%SWITCH%" (
    echo [KILL SWITCH IS ENGAGED] - all orders are being rejected.
    echo File: %SWITCH%
) else (
    echo [KILL SWITCH IS OFF]
)
echo.
echo Usage: kill_switch.bat [on^|off]

:end
popd
endlocal
exit /b 0
