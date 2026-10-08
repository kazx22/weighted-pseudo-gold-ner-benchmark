@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

echo ==============================================================
echo BIORED KAPPA + TAXONOMY SANITY CHECK - SAVED PREDICTIONS ONLY
echo No NER models will be rerun.
echo ==============================================================
echo.

set "BIORED_VARIANT=official"

echo [1/4] Audit BioRED TEST prediction spans...
"%PYTHON%" -m src.span_integrity_audit --dataset biored --split test --top 50 --fail-on-integrity-error
if errorlevel 1 goto :fail

echo.
echo [2/4] Regenerate BioRED token-level kappa...
"%PYTHON%" -m src.biored_kappa --split test
if errorlevel 1 goto :fail

echo.
echo [3/4] Run detailed kappa diagnostics...
"%PYTHON%" -m src.biored_kappa_diagnostic --split test --sample-docs 5
if errorlevel 1 goto :fail

echo.
echo [4/4] Regenerate taxonomy for all five models...
"%PYTHON%" -m src.biored_error_taxonomy --split test --models scispacy biobert pubmedbert clinicalbert d4data
if errorlevel 1 goto :fail

echo.
echo ==============================================================
echo CHECK COMPLETE

echo Span audit: results\biored\test\span_audit\
echo Kappa:      results\biored\test\kappa_diagnostic\
echo Taxonomy:   results\biored\test\error_taxonomy\
echo ==============================================================
exit /b 0

:fail
echo.
echo CHECK FAILED. Read the error above.
exit /b 1
