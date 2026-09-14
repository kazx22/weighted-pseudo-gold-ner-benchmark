@echo off
setlocal EnableExtensions
cd /d "%~dp0"

for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "RUNSTAMP=%%I"
set "BACKUP_DIR=backups\before_dev_test_%RUNSTAMP%"

mkdir "%BACKUP_DIR%" >nul 2>&1

echo This keeps raw datasets and source code untouched.
echo Existing generated outputs will be MOVED to:
echo %BACKUP_DIR%
echo.

call :move_if_exists "data\processed\bc5cdr" "data_processed_bc5cdr"
call :move_if_exists "data\gold" "data_gold"
call :move_if_exists "results" "results"
call :move_if_exists "figure" "figure"
call :move_if_exists "data\analysis\error_taxonomy" "old_error_taxonomy"

echo.
echo Clean-run preparation completed.
echo Raw files remain in data\raw\bc5cdr\
echo Backup created at %BACKUP_DIR%
exit /b 0

:move_if_exists
if exist "%~1" (
    echo Moving %~1
    move "%~1" "%BACKUP_DIR%\%~2" >nul
) else (
    echo Not found, skipping: %~1
)
exit /b 0
