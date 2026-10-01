@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv\Scripts\python.exe was not found.
    echo Create/restore the project virtual environment first.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
python -m src.run_medgemma
set EXITCODE=%ERRORLEVEL%

if not "%EXITCODE%"=="0" (
    echo.
    echo RUN_MEDGEMMA failed. Read the newest log in logs\.
    pause
    exit /b %EXITCODE%
)

echo.
echo RUN_MEDGEMMA completed successfully.
pause
