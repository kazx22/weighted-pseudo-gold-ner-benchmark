@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if not exist "RUN_MEDGEMMA_BC5CDR.ps1" (
    echo ERROR: RUN_MEDGEMMA_BC5CDR.ps1 is missing.
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0RUN_MEDGEMMA_BC5CDR.ps1"
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
    echo.
    echo MEDGEMMA BC5CDR RUN FAILED.
    echo Run this CMD again after fixing the reported problem. Completed documents resume from cache.
)

exit /b %RC%
