@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if not exist "RUN_PIPELINE.ps1" (
    echo ERROR: RUN_PIPELINE.ps1 is missing from the project root.
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0RUN_PIPELINE.ps1" -Stage test
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
    echo.
    echo FINAL TEST RUN FAILED.
)

exit /b %RC%
