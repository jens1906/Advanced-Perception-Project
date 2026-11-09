from pathlib import Path

import cv2

# Clip parameters
CLIP_PATH = Path("Data/denmark_wales_clip_1min.mp4")
CLIP_DIR = Path("Data/denmark_wales")
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
    print(f"Wrote {frameCount} frames ({width}x{height}) to {imageDir}, put the CVAT MOT gt.txt in {CLIP_DIR / 'gt'}")
