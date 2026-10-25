@echo off
setlocal EnableExtensions
cd /d "%~dp0"

call SETUP_ENVIRONMENT.cmd
if errorlevel 1 goto :fail

echo ==============================================================
echo SYMMETRIC SUPERVISED CONTROLS: BC5CDR + BIORED
echo ==============================================================
".venv\Scripts\python.exe" -m src.finetuned_models_symmetric %*
if errorlevel 1 goto :fail

echo.
echo ==============================================================
echo REFRESHING RESOURCE / PROVENANCE / MANUSCRIPT SUMMARIES
echo ==============================================================
".venv\Scripts\python.exe" -m src.resource_summary
if errorlevel 1 goto :fail
".venv\Scripts\python.exe" -m src.reproducibility_snapshot
if errorlevel 1 goto :fail
".venv\Scripts\python.exe" -m src.manuscript_results_log
if errorlevel 1 goto :fail

echo.
echo ==============================================================
echo COMPLETE
echo SYMMETRIC SUPERVISED CONTROLS: BC5CDR + BIORED
echo ==============================================================
exit /b 0

:fail
echo.
echo ==============================================================
echo FAILED
if exist "logs" echo Check the newest log in logs\
echo ==============================================================
exit /b 1
