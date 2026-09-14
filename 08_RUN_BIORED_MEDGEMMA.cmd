@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo BioRED DISEASE + CHEMICAL: MEDGEMMA EXTENSION
echo ============================================================
echo Make sure Ollama is running and medgemma1.5:4b-it-q4_K_M is installed.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0RUN_BIORED.ps1" -Mode MedGemma
if errorlevel 1 (
  echo.
  echo BIORED MEDGEMMA FAILED. Read the newest logs\biored_disease_chemical_*.log
  pause
  exit /b 1
)
echo.
echo BIORED MEDGEMMA COMPLETED.
pause
