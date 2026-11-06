import random

import numpy as np
from ultralytics import YOLO

from PreliminaryFunctions import TRAIN_DIR, FindSequences, PlaySequence
from Tuning import YOLO_BALL_CLASS, YOLO_CONFIDENCE, YOLO_IMAGE_SIZE, YOLO_TRACKER, YOLO_WEIGHTS

# YOLO model
model = YOLO(YOLO_WEIGHTS)


def ResetTracker():
    if model.predictor is not None and hasattr(model.predictor, "trackers"):
        for tracker in model.predictor.trackers:
            tracker.reset()


def ProcessFrame(frame):
    result = model.track(frame, persist=True, tracker=YOLO_TRACKER, imgsz=YOLO_IMAGE_SIZE, conf=YOLO_CONFIDENCE, classes=[YOLO_BALL_CLASS], verbose=False)[0]
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

    seqDir = random.choice(sequences)
    print(f"Playing {seqDir}")
    PlaySequence(seqDir, ProcessFrame, ResetTracker)
