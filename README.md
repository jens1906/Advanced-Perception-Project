# Advanced-Perception-Project

YOLO training branch: fine-tunes YOLO26 on SoccerNet tracking (ball, player, goalkeeper, referee).

## Setup
Linux: `./setup.sh`. Windows: `setup.bat`. Both create `.venv` with CUDA PyTorch (cu126: GTX 10xx up to RTX 40xx).
For an RTX 50xx GPU, set `TORCH_INDEX=https://download.pytorch.org/whl/cu128` first.
The script should end with `CUDA available: True`.

## Data
With `.venv` active, download and unzip SoccerNet tracking train and test (about 18 GB) into `Data/SoccerNet/tracking/`:
```
python DownloadData.py
```
Zips or folders that are already there are not downloaded or extracted again.

## Train
Activate `.venv` (`source .venv/bin/activate` or `.venv\Scripts\activate`), then run:
```
python ConvertToYolo.py
python TrainYolo.py
```
The result is `runs/detect/soccernet/weights/best.pt`.
