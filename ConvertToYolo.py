import configparser
import shutil
from pathlib import Path

import numpy as np

from PreliminaryFunctions import TRAIN_DIR, FindSequences, LoadSequenceInfo

# Conversion parameters
VAL_DIR = Path("Data/SoccerNet/tracking/test")
OUTPUT_DIR = Path("Data/yolo")
FRAME_STEP = 5
CLASS_NAMES = ["ball", "player", "goalkeeper", "referee"]


def LoadTrackletClasses(seqDir):
    config = configparser.ConfigParser()
    config.read(seqDir / "gameinfo.ini")
    classes = {}
    for key, value in config["Sequence"].items():
        if not key.startswith("trackletid_"):
            continue
        for classIndex, name in enumerate(CLASS_NAMES):
            if value.strip().startswith(name):
                classes[int(key.split("_")[1])] = classIndex
    return classes


def ConvertSequence(seqDir, split):
    info = LoadSequenceInfo(seqDir)
    config = configparser.ConfigParser()
    config.read(seqDir / "seqinfo.ini")
    width, height = int(config["Sequence"]["imWidth"]), int(config["Sequence"]["imHeight"])
    classes = LoadTrackletClasses(seqDir)
    rows = np.loadtxt(seqDir / "gt" / "gt.txt", delimiter=",", ndmin=2)

    framePaths = sorted(info["im_dir"].glob(f"*{info['ext']}"))[::FRAME_STEP]
    for framePath in framePaths:
        lines = []
        for row in rows[rows[:, 0] == int(framePath.stem)]:
            classIndex = classes.get(int(row[1]))
            if classIndex is None:
                continue
            x1, y1 = max(row[2], 0), max(row[3], 0)
            x2, y2 = min(row[2] + row[4], width), min(row[3] + row[5], height)
            if x2 <= x1 or y2 <= y1:
                continue
            lines.append(f"{classIndex} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
                         f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")

        name = f"{info['name']}_{framePath.stem}"
        (OUTPUT_DIR / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
        imagePath = OUTPUT_DIR / "images" / split / f"{name}{info['ext']}"
        try:
            imagePath.symlink_to(framePath.resolve())
        except OSError:
            shutil.copy(framePath, imagePath)
    return len(framePaths)


if __name__ == "__main__":
    splits = {"train": FindSequences(TRAIN_DIR), "val": FindSequences(VAL_DIR)}
    if not splits["val"]:
        raise SystemExit(f"No sequences found under {VAL_DIR}, unzip test.zip there first")

    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    for split, sequences in splits.items():
        (OUTPUT_DIR / "images" / split).mkdir(parents=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True)
        imageCount = sum(ConvertSequence(seqDir, split) for seqDir in sequences)
        print(f"{split}  {len(sequences)} sequences  {imageCount} images")

    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    (OUTPUT_DIR / "soccernet.yaml").write_text(
        f"path: {OUTPUT_DIR.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
