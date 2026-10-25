@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if /I "%~1"=="--dry-run" goto :dry
call SETUP_ENVIRONMENT.cmd
if errorlevel 1 goto :fail
:dry
if not exist ".venv\Scripts\python.exe" (
  echo Missing environment. Run SETUP_ENVIRONMENT.cmd first.
  exit /b 1
)
".venv\Scripts\python.exe" -m src.run_all %*
if errorlevel 1 goto :fail
 echo.
 echo SUCCESS. Check results\publication and results\run_control\last_run.json
exit /b 0
:fail
echo.
echo FAILED. Inspect results\run_control\last_run.json and logs\
exit /b 1
