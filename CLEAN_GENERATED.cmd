@echo off
setlocal
cd /d "%~dp0"
echo This removes generated results, checkpoints, logs, processed data and extracted BioRED files.
echo Raw BC5CDR and BIORED.zip are preserved.
set /p CONFIRM=Type YES to continue: 
if /I not "%CONFIRM%"=="YES" exit /b 0
if exist results rmdir /s /q results
if exist checkpoints rmdir /s /q checkpoints
if exist logs rmdir /s /q logs
if exist data\processed rmdir /s /q data\processed
if exist data\gold rmdir /s /q data\gold
if exist data\raw\biored rmdir /s /q data\raw\biored
if exist data\raw\_extracted rmdir /s /q data\raw\_extracted
echo Clean complete. Raw source datasets were preserved.
