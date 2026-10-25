@echo off
setlocal EnableExtensions
cd /d "%~dp0"

call SETUP_ENVIRONMENT.cmd
if errorlevel 1 goto :fail


echo ==============================================================
echo BIORED OFFICIAL + OVERLAP-EXCLUDED SENSITIVITY
echo ==============================================================
".venv\Scripts\python.exe" -m src.run_biored %*
if errorlevel 1 goto :fail

echo.
echo ==============================================================
echo COMPLETE
echo BIORED OFFICIAL + OVERLAP-EXCLUDED SENSITIVITY
echo ==============================================================
exit /b 0

:fail
echo.
echo ==============================================================
echo FAILED
if exist "logs" echo Check the newest log in logs\
echo ==============================================================
exit /b 1
