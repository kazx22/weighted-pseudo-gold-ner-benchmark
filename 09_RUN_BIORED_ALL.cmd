@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo BioRED DISEASE + CHEMICAL: COMPLETE PIPELINE
echo ============================================================
echo This runs the five conventional models, weighted pseudo-gold,
echo held-out evaluation, then the local MedGemma experiment.
echo Existing conventional prediction files are skipped automatically.
echo MedGemma uses durable caches and resumes completed documents.
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0RUN_BIORED.ps1" -Mode All
if errorlevel 1 (
  echo.
  echo BIORED PIPELINE FAILED. Read the newest logs\biored_disease_chemical_*.log
  pause
  exit /b 1
)
echo.
echo BIORED COMPLETE PIPELINE FINISHED.
pause
