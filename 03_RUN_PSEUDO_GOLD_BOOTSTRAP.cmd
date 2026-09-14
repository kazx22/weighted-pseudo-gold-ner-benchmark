@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if not exist "RUN_PSEUDO_GOLD_BOOTSTRAP.ps1" (
    echo ERROR: RUN_PSEUDO_GOLD_BOOTSTRAP.ps1 is missing from the project root.
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0RUN_PSEUDO_GOLD_BOOTSTRAP.ps1"
exit /b %ERRORLEVEL%
