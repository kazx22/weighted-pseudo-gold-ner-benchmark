@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

echo ==============================================================
echo CONTENT-AWARE WEIGHTED VOTING - BC5CDR ONLY
echo MV and WV are NOT changed by this runner.
echo ==============================================================

echo.
echo [1/3] Fit CWA reliability, alpha, and threshold on DEV only...
"%PYTHON%" -m src.content_aware_voting --split dev
if errorlevel 1 goto :fail

echo.
echo [2/3] Apply frozen CWA settings to TEST without test gold...
"%PYTHON%" -m src.content_aware_voting --split test
if errorlevel 1 goto :fail

echo.
echo [3/3] Evaluate MV -^> WV -^> CWA and run paired bootstrap on held-out TEST...
"%PYTHON%" -m src.cwa_evaluation --split test --resamples 1000 --seed 42
if errorlevel 1 goto :fail

echo.
echo ==============================================================
echo CWA COMPLETE
echo ==============================================================
exit /b 0

:fail
echo.
echo CWA RUN FAILED. See the error above.
exit /b 1
