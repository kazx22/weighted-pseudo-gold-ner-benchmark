@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

echo ==============================================================
echo PREDICTION SPAN INTEGRITY AUDIT - SAVED PREDICTIONS ONLY
echo No NER models will be rerun.
echo ==============================================================

echo.
echo [1/6] BC5CDR DEV...
"%PYTHON%" -m src.span_integrity_audit --dataset bc5cdr --split dev --top 50 --fail-on-integrity-error
if errorlevel 1 goto :fail

echo.
echo [2/6] BC5CDR TEST...
"%PYTHON%" -m src.span_integrity_audit --dataset bc5cdr --split test --top 50 --fail-on-integrity-error
if errorlevel 1 goto :fail

set "BIORED_VARIANT=official"
echo.
echo [3/6] BioRED official DEV...
"%PYTHON%" -m src.span_integrity_audit --dataset biored --split dev --top 50 --fail-on-integrity-error
if errorlevel 1 goto :fail

echo.
echo [4/6] BioRED official TEST...
"%PYTHON%" -m src.span_integrity_audit --dataset biored --split test --top 50 --fail-on-integrity-error
if errorlevel 1 goto :fail

set "BIORED_VARIANT=overlap_excluded"
echo.
echo [5/6] BioRED overlap-excluded DEV...
"%PYTHON%" -m src.span_integrity_audit --dataset biored --split dev --top 50 --fail-on-integrity-error
if errorlevel 1 goto :fail

echo.
echo [6/6] BioRED overlap-excluded TEST...
"%PYTHON%" -m src.span_integrity_audit --dataset biored --split test --top 50 --fail-on-integrity-error
if errorlevel 1 goto :fail

echo.
echo ==============================================================
echo SPAN AUDIT COMPLETE

echo PASS = no hard offset/text corruption.
echo WARN = inspect longest_predictions.csv for suspicious long spans.
echo ==============================================================
exit /b 0

:fail
echo.
echo SPAN AUDIT FAILED. A hard prediction integrity problem was found.
echo Check the span_audit folder printed above before rerunning evaluation.
exit /b 1
