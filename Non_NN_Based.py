import configparser
import random
from pathlib import Path

import cv2

TRAIN_DIR = Path("Data/SoccerNet/tracking/train")


def find_sequences(root):
    # A sequence is any folder containing seqinfo.ini (robust to nested train/train)
    return sorted(p.parent for p in root.rglob("seqinfo.ini"))


def load_seqinfo(seq_dir):
    config = configparser.ConfigParser()
    config.read(seq_dir / "seqinfo.ini")
    info = config["Sequence"]
    return {
        "name": info["name"],
        "im_dir": seq_dir / info["imDir"],
        "fps": int(info["frameRate"]),
        "length": int(info["seqLength"]),
        "ext": info["imExt"],
    }

PreFrame = None  # Global variable to store the previous frame

def process_frame(frame):
    global PreFrame
    if PreFrame is None:
        PreFrame = frame.copy()
        return frame  # Return the first frame as is
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Apply a threshold to the HSV frame to isolate the green color
    hsv_thresh = cv2.inRange(hsv_frame, (0, 30, 0), (255, 130, 255))  # Example threshold for green color
    dialted_frame = cv2.dilate(hsv_thresh, None, iterations=2)
    eroded_frame = cv2.erode(dialted_frame, None, iterations=1)
    #OutputFrame = cv2.absdiff(frame, PreFrame)
    #PreFrame = frame.copy()
    '''
    gray_frame = cv2.cvtColor(OutputFrame, cv2.COLOR_BGR2GRAY)
    dialted_frame = cv2.dilate(gray_frame, None, iterations=2)
    eroded_frame = cv2.erode(dialted_frame, None, iterations=1)
    '''

    # Placeholder for the non-NN ball tracker; returns the frame unchanged for now
    return eroded_frame  # Return the processed frame (currently just the eroded difference)


def play_sequence(seq_dir):
    info = load_seqinfo(seq_dir)
    frame_paths = sorted(info["im_dir"].glob(f"*{info['ext']}"))
    delay_ms = max(1, int(1000 / info["fps"]))

    window = f"Non-NN - {info['name']}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 720)

    paused = False
    i = 0
    while i < len(frame_paths):
        if not paused:
            frame = cv2.imread(str(frame_paths[i]))
            frame = process_frame(frame)
            cv2.putText(frame, f"{info['name']}  frame {i + 1}/{len(frame_paths)}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.imshow(window, frame)
            i += 1

        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord("q") or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord(" "):
            paused = not paused

    cv2.destroyAllWindows()


if __name__ == "__main__":
    sequences = find_sequences(TRAIN_DIR)
    if not sequences:
        raise SystemExit(f"No sequences found under {TRAIN_DIR}")

    seq_dir = random.choice(sequences)
    print(f"Playing {seq_dir}")
    play_sequence(seq_dir)
