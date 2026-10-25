@echo off
setlocal
set "PY=%~1"
if "%PY%"=="" set "PY=python"

"%PY%" -c "import spacy; spacy.load('en_ner_bc5cdr_md')" >nul 2>&1
if not errorlevel 1 (
    echo scispaCy model preflight OK: en_ner_bc5cdr_md
    exit /b 0
)

echo.
echo ==============================================================
echo SCISPACY MODEL SETUP
echo ==============================================================
echo Installing en_ner_bc5cdr_md 0.5.4 used by this project.
echo ==============================================================
"%PY%" -m pip install "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz"
if errorlevel 1 exit /b 1
"%PY%" -c "import spacy; nlp=spacy.load('en_ner_bc5cdr_md'); print('scispaCy model ready:', nlp.meta.get('name'), nlp.meta.get('version'))"
if errorlevel 1 exit /b 1
exit /b 0
