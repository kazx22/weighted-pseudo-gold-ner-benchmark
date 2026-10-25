@echo off
setlocal

set "PY=%~1"
if "%PY%"=="" set "PY=python"

"%PY%" -c "import torch,sys; v=tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]); sys.exit(0 if v >= (2,6) and torch.cuda.is_available() else 1)" >nul 2>&1
if not errorlevel 1 (
    "%PY%" -c "import torch; print('PyTorch preflight OK:', torch.__version__, '| CUDA runtime:', torch.version.cuda, '| GPU:', torch.cuda.get_device_name(0))"
    exit /b 0
)

echo.
echo ==============================================================
echo PYTORCH/CUDA PREFLIGHT
echo ==============================================================
echo The public BioBERT checkpoints use pytorch_model.bin.
echo Transformers requires PyTorch 2.6 or newer to load that format safely.
echo Installing the pinned CUDA build used by this repository:
echo   torch 2.6.0 + CUDA 12.4
echo   torchvision 0.21.0
echo   torchaudio 2.6.0
echo This is a one-time environment update.
echo ==============================================================

"%PY%" -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
if errorlevel 1 (
    echo.
    echo ERROR: PyTorch 2.6 CUDA installation failed.
    exit /b 1
)

"%PY%" -c "import torch,sys; v=tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]); print('Installed PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); sys.exit(0 if v >= (2,6) and torch.cuda.is_available() else 1)"
if errorlevel 1 (
    echo.
    echo ERROR: PyTorch 2.6 was installed but CUDA is not available.
    echo Update the NVIDIA driver, then rerun the same command.
    exit /b 1
)

exit /b 0
