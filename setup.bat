@echo off
rem Creates .venv with CUDA PyTorch and the training requirements.
rem cu126 supports GTX 10xx up to RTX 40xx. For RTX 50xx run: set TORCH_INDEX=https://download.pytorch.org/whl/cu128 before setup.bat
if "%TORCH_INDEX%"=="" set TORCH_INDEX=https://download.pytorch.org/whl/cu126

python -m venv .venv || exit /b 1
.venv\Scripts\python -m pip install --upgrade pip || exit /b 1
.venv\Scripts\pip install torch torchvision --index-url %TORCH_INDEX% || exit /b 1
.venv\Scripts\pip install -r requirements.txt || exit /b 1
.venv\Scripts\python -c "import torch; print('CUDA available:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
