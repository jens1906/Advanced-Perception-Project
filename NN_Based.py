import random
from pathlib import Path
import sys

import numpy as np
from ultralytics import YOLO

from PreliminaryFunctions import TRAIN_DIR, FindSequences, PlaySequence

# YOLO model
model = YOLO("yolo26n.pt")


def ResetTracker():
    if model.predictor is not None and hasattr(model.predictor, "trackers"):
        for tracker in model.predictor.trackers:
            tracker.reset()


def ProcessFrame(frame):
    result = model.track(frame, persist=True, tracker="botsort.yaml", imgsz=1280, conf=0.1, classes=[32], verbose=False)[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    candidates = np.column_stack([boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]])
    ballPosition = None
    if len(boxes):
        x1, y1, x2, y2 = boxes[result.boxes.conf.cpu().numpy().argmax()]
        ballPosition = ((x1 + x2) / 2, (y1 + y2) / 2)
    return [result.plot()], ballPosition, candidates


if __name__ == "__main__":
    sequences = FindSequences(TRAIN_DIR)
    if not sequences:
        raise SystemExit(f"No sequences found under {TRAIN_DIR}")

    seqDir = Path(sys.argv[1]) if len(sys.argv) > 1 else random.choice(sequences)
    print(f"Playing {seqDir}")
    PlaySequence(seqDir, ProcessFrame, ResetTracker)
