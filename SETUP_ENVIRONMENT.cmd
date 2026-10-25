@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PYBASE="
where py >nul 2>&1
if not errorlevel 1 (
    py -3.11 -c "import sys; assert sys.version_info[:2]==(3,11)" >nul 2>&1
    if not errorlevel 1 set "PYBASE=py -3.11"
)
if not defined PYBASE (
    where python >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Python 3.11 was not found.
        echo Install Python 3.11 x64 and rerun RUN_ALL.cmd.
        exit /b 1
    )
    python -c "import sys; assert sys.version_info[:2]==(3,11)" >nul 2>&1
    if errorlevel 1 (
        echo ERROR: This repository is locked to Python 3.11.
        python --version
        exit /b 1
    )
    set "PYBASE=python"
)

if not exist ".venv\Scripts\python.exe" (
    echo ==============================================================
    echo CREATING PROJECT VIRTUAL ENVIRONMENT
    echo ==============================================================
    %PYBASE% -m venv .venv
    if errorlevel 1 exit /b 1
)

set "PY=.venv\Scripts\python.exe"
"%PY%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 exit /b 1

call src\ensure_torch_cuda.cmd "%PY%"
if errorlevel 1 exit /b 1

"%PY%" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

call src\ensure_tokenizer_backends.cmd "%PY%"
if errorlevel 1 exit /b 1

call src\ensure_scispacy_model.cmd "%PY%"
if errorlevel 1 exit /b 1

"%PY%" -c "import numpy, matplotlib, sklearn, spacy, scispacy, transformers, psutil, torch; print('Environment ready. Python import checks passed.'); print('torch', torch.__version__, '| transformers', transformers.__version__, '| spacy', spacy.__version__)"
if errorlevel 1 exit /b 1

exit /b 0
