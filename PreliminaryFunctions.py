import configparser
from pathlib import Path

import cv2
import numpy as np

from Tuning import VERIFY_MARGIN

# Dataset constants
TRAIN_DIR = Path("Data/SoccerNet/tracking/train")


def FindSequences(root):
    return sorted(p.parent for p in root.rglob("seqinfo.ini"))


def LoadSequenceInfo(seqDir):
    config = configparser.ConfigParser()
    config.read(seqDir / "seqinfo.ini")
    info = config["Sequence"]
    return {
        "name": info["name"],
        "im_dir": seqDir / info["imDir"],
        "fps": int(info["frameRate"]),
        "length": int(info["seqLength"]),
        "ext": info["imExt"],
    }


def LoadBallGroundTruth(seqDir):
    config = configparser.ConfigParser()
    config.read(seqDir / "gameinfo.ini")
    ballIds = []
    for key, value in config["Sequence"].items():
        if value.strip().startswith("ball"):
            ballIds.append(int(key.split("_")[1]))
    if not ballIds:
        return {}
    rows = np.loadtxt(seqDir / "gt" / "gt.txt", delimiter=",", ndmin=2)
    return {int(row[0]): row[2:6] for row in rows if row[1] == ballIds[0]}


def PlaySequence(seqDir, ProcessFrame, Reset=None, margin=VERIFY_MARGIN):
    if Reset is not None:
        Reset()
    info = LoadSequenceInfo(seqDir)
    groundTruth = LoadBallGroundTruth(seqDir)
    framePaths = sorted(info["im_dir"].glob(f"*{info['ext']}"))
    hits, candidateHits, errors, frameCount = 0, 0, [], 0
    delayMs = max(1, int(100 / info["fps"]))

    paused = False
    frameIndex = 0
    while frameIndex < len(framePaths):
        if not paused:
            frames, ballPosition, candidates = ProcessFrame(cv2.imread(str(framePaths[frameIndex])))
            windows = [f"{info['name']} {windowIndex}" for windowIndex in range(len(frames))]
            if frameIndex == 0:
                for window in windows:
                    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window, 960, 540)
            box = groundTruth.get(int(framePaths[frameIndex].stem))
            if box is not None:
                hit, candidateHit, error = ScoreFrame(ballPosition, candidates, box, margin)
                hits += hit
                candidateHits += candidateHit
                errors += error
                frameCount += 1
            cv2.putText(frames[0], f"{info['name']}  frame {frameIndex + 1}/{len(framePaths)}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            for window, frame in zip(windows, frames):
                cv2.imshow(window, frame)
            frameIndex += 1

        key = cv2.waitKey(delayMs) & 0xFF
        if key == ord("q") or any(cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1 for window in windows):
            break
        if key == ord(" "):
            paused = not paused

    cv2.destroyAllWindows()
    PrintVerificationResult(info["name"], hits, candidateHits, errors, frameCount)


def IsInsideBox(point, box, margin):
    x, y, w, h = box
    return x - margin <= point[0] <= x + w + margin and y - margin <= point[1] <= y + h + margin


def ScoreFrame(ballPosition, candidates, box, margin):
    boxCenter = box[:2] + box[2:] / 2
    hit, errors = False, []
    if ballPosition is not None:
        errors.append(np.linalg.norm(np.asarray(ballPosition) - boxCenter))
        hit = IsInsideBox(ballPosition, box, margin)
    candidateCenters = candidates[:, :2] + candidates[:, 2:4] / 2
    candidateHit = any(IsInsideBox(center, box, margin) for center in candidateCenters)
    return hit, candidateHit, errors


def PrintVerificationResult(name, hits, candidateHits, errors, frameCount):
    if frameCount == 0:
        print(f"{name}  no ground truth ball")
        return
    meanError = f"{np.mean(errors):.1f}px" if errors else "-"
    print(f"{name}  hit {100 * hits / frameCount:.1f}%  candidate recall {100 * candidateHits / frameCount:.1f}%  "
          f"error {meanError}  ({frameCount} GT frames)")


def VerifySequences(seqDirs, ProcessFrame, Reset=None, margin=VERIFY_MARGIN):
    totalHits, totalCandidateHits, totalErrors, totalFrames = 0, 0, [], 0
    for seqDir in seqDirs:
        if Reset is not None:
            Reset()
        info = LoadSequenceInfo(seqDir)
        groundTruth = LoadBallGroundTruth(seqDir)
        hits, candidateHits, errors, frameCount = 0, 0, [], 0

        for framePath in sorted(info["im_dir"].glob(f"*{info['ext']}")):
            _, ballPosition, candidates = ProcessFrame(cv2.imread(str(framePath)))
            box = groundTruth.get(int(framePath.stem))
            if box is None:
                continue
            hit, candidateHit, error = ScoreFrame(ballPosition, candidates, box, margin)
            hits += hit
            candidateHits += candidateHit
            errors += error
            frameCount += 1

        PrintVerificationResult(info["name"], hits, candidateHits, errors, frameCount)
        totalHits += hits
        totalCandidateHits += candidateHits
        totalErrors += errors
        totalFrames += frameCount

    PrintVerificationResult("TOTAL", totalHits, totalCandidateHits, totalErrors, totalFrames)
