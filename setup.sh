#!/usr/bin/env bash
# Creates .venv with CUDA PyTorch and the training requirements.
# cu126 supports GTX 10xx up to RTX 40xx. For RTX 50xx run: TORCH_INDEX=https://download.pytorch.org/whl/cu128 ./setup.sh
set -e
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch torchvision --index-url "$TORCH_INDEX"
.venv/bin/pip install -r requirements.txt
.venv/bin/python -c "import torch; print('CUDA available:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
