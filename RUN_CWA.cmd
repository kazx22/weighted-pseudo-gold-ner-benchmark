@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

echo ==============================================================
echo CONTENT-AWARE WEIGHTED VOTING - BC5CDR + BIORED
echo MV and WV are NOT changed by this runner.
echo ==============================================================

echo.
echo [1/6] BC5CDR: fit CWA on DEV...
"%PYTHON%" -m src.content_aware_voting --split dev
if errorlevel 1 goto :fail

echo.
echo [2/6] BC5CDR: apply frozen CWA to TEST...
"%PYTHON%" -m src.content_aware_voting --split test
if errorlevel 1 goto :fail

echo.
echo [3/6] BC5CDR: evaluate MV -^> WV -^> CWA + paired bootstrap...
"%PYTHON%" -m src.cwa_evaluation --split test --resamples 1000 --seed 42
if errorlevel 1 goto :fail

set "BIORED_VARIANT=official"

echo.
echo [4/6] BioRED: fit CWA on official DEV...
"%PYTHON%" -m src.biored_content_aware_voting --split dev
if errorlevel 1 goto :fail

echo.
echo [5/6] BioRED: apply frozen CWA to official TEST...
"%PYTHON%" -m src.biored_content_aware_voting --split test
if errorlevel 1 goto :fail

echo.
echo [6/6] BioRED: evaluate MV -^> WV -^> CWA + paired bootstrap...
"%PYTHON%" -m src.biored_cwa_evaluation --split test --resamples 1000 --seed 42
if errorlevel 1 goto :fail

echo.
echo ==============================================================
echo CWA COMPLETE - BC5CDR + BIORED
echo ==============================================================
exit /b 0

:fail
echo.
echo CWA RUN FAILED. See the error above.
exit /b 1
