@echo off
setlocal EnableExtensions
cd /d "%~dp0"

python -m src.medgemma_smoke_test
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo MEDGEMMA SMOKE TEST FAILED.
)
exit /b %RC%
