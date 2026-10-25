@echo off
setlocal

set "PY=%~1"
if "%PY%"=="" set "PY=python"

"%PY%" -c "import sentencepiece, tiktoken, tokenizers" >nul 2>&1
if not errorlevel 1 exit /b 0

echo.
echo ==============================================================
echo TOKENIZER BACKEND PREFLIGHT
echo ==============================================================
echo Installing tokenizer dependencies required by current
echo Transformers releases for legacy WordPiece checkpoints:
echo   sentencepiece
echo   tiktoken
echo   tokenizers
echo ==============================================================

"%PY%" -m pip install --upgrade sentencepiece tiktoken tokenizers
if errorlevel 1 (
    echo ERROR: tokenizer dependency installation failed.
    exit /b 1
)

"%PY%" -c "import sentencepiece, tiktoken, tokenizers; print('Tokenizer backend preflight OK')"
if errorlevel 1 exit /b 1

exit /b 0
