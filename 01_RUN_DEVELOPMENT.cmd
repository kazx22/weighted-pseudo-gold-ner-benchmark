@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if not exist "RUN_PIPELINE.ps1" (
    echo ERROR: RUN_PIPELINE.ps1 is missing from the project root.
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0RUN_PIPELINE.ps1" -Stage development
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
    echo.
    echo DEVELOPMENT RUN FAILED.
)

exit /b %RC%
