@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo MEDGEMMA BC5CDR POST-PROCESSING REPORT
echo No Ollama calls. No MedGemma inference. No six-hour rerun.
echo ============================================================

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Could not find .venv\Scripts\python.exe
    echo Run this file from the project root after copying the patch there.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"
python -m src.medgemma_report --split test

if errorlevel 1 (
    echo.
    echo REPORT GENERATION FAILED.
    echo Read the Python error above. The LLM outputs have not been changed.
    pause
    exit /b 1
)

echo.
echo REPORT GENERATION COMPLETED.
echo Open: results\test\medgemma_report\MEDGEMMA_RESULTS_REPORT.md
echo Figures: results\test\medgemma_report\figures
pause
endlocal
