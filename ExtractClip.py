from pathlib import Path

import cv2
import numpy as np

# Clip parameters
CLIP_PATH = Path("Data/denmark_wales_clip_1min.mp4")
CLIP_DIR = Path("Data/denmark_wales")
CVAT_GT_PATH = Path("Data/AnnotatedFootballClip/gt/gt.txt")  # MOT 1.1 export from CVAT
BALL_TRACK_ID = 1

if __name__ == "__main__":
    imageDir = CLIP_DIR / "img1"
    imageDir.mkdir(parents=True, exist_ok=True)
    (CLIP_DIR / "gt").mkdir(exist_ok=True)

    capture = cv2.VideoCapture(str(CLIP_PATH))
    fps = round(capture.get(cv2.CAP_PROP_FPS))
    frameCount, height, width = 0, 0, 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frameCount += 1
        height, width = frame.shape[:2]
        cv2.imwrite(str(imageDir / f"{frameCount:06d}.jpg"), frame)
    capture.release()

    (CLIP_DIR / "seqinfo.ini").write_text(
        f"[Sequence]\nname={CLIP_DIR.name}\nimDir=img1\nframeRate={fps}\nseqLength={frameCount}\n"
        f"imWidth={width}\nimHeight={height}\nimExt=.jpg\n")
    (CLIP_DIR / "gameinfo.ini").write_text(f"[Sequence]\nname={CLIP_DIR.name}\ntrackletID_{BALL_TRACK_ID}= ball;1\n")
    print(f"Wrote {frameCount} frames ({width}x{height}) to {imageDir}")

    # CVAT exports boxes drawn in Shape mode with track ID -1, so give every row the ball ID
    rows = np.loadtxt(CVAT_GT_PATH, delimiter=",", ndmin=2)
    rows[:, 1] = BALL_TRACK_ID
    np.savetxt(CLIP_DIR / "gt" / "gt.txt", rows, delimiter=",", fmt="%g")
    print(f"Imported {len(rows)} ball boxes from {CVAT_GT_PATH} (frames {int(rows[:, 0].min())}-{int(rows[:, 0].max())})")
