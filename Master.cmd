@echo off
setlocal EnableExtensions
cd /d "%~dp0"

title BC5CDR Complete Pipeline

echo ============================================================
echo   BC5CDR COMPLETE PIPELINE
echo ============================================================
echo.
echo This will run, in order:
echo   1. Clean/preparation
echo   2. Development pipeline
echo   3. Final held-out test pipeline
echo   4. Pseudo-gold bootstrap analysis
echo.
echo IMPORTANT:
echo   - The clean step runs only once, before development.
echo   - It is NOT run between development and test.
echo   - The five NER models will be run again.
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Python virtual environment was not found:
    echo         %CD%\.venv\Scripts\python.exe
    echo.
    echo Create or restore the .venv folder before running this file.
    pause
    exit /b 1
)

for %%F in (
    "00_PREPARE_CLEAN_RUN.cmd"
    "01_RUN_DEVELOPMENT.cmd"
    "02_RUN_FINAL_TEST.cmd"
    "03_RUN_PSEUDO_GOLD_BOOTSTRAP.cmd"
) do (
    if not exist "%%~F" (
        echo [ERROR] Required runner is missing: %%~F
        pause
        exit /b 1
    )
)

echo Press CTRL+C now to cancel, or
pause

echo.
echo ============================================================
echo [1/4] PREPARING CLEAN RUN
echo ============================================================
call "00_PREPARE_CLEAN_RUN.cmd"
if errorlevel 1 goto :failed

echo.
echo ============================================================
echo [2/4] RUNNING DEVELOPMENT
echo ============================================================
call "01_RUN_DEVELOPMENT.cmd"
if errorlevel 1 goto :failed

echo.
echo ============================================================
echo [3/4] RUNNING FINAL HELD-OUT TEST
echo ============================================================
call "02_RUN_FINAL_TEST.cmd"
if errorlevel 1 goto :failed

echo.
echo ============================================================
echo [4/4] RUNNING PSEUDO-GOLD BOOTSTRAP
echo ============================================================
call "03_RUN_PSEUDO_GOLD_BOOTSTRAP.cmd"
if errorlevel 1 goto :failed

echo.
echo ============================================================
echo   COMPLETE PIPELINE FINISHED SUCCESSFULLY
echo ============================================================
echo.
echo Check:
echo   results\dev\
echo   results\test\
echo   logs\
echo.
pause
exit /b 0

:failed
echo.
echo ============================================================
echo   PIPELINE STOPPED BECAUSE A STEP FAILED
echo ============================================================
echo.
echo Review the most recent file in the logs folder.
echo No later stage was run after the failure.
echo.
pause
exit /b 1
