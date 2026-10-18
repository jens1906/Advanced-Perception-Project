import configparser
import random
from pathlib import Path
from unittest import result

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

'''
Load the ground truth. Look up the ball's track ID in gameinfo.ini (e.g. ball;1 = ID 18). Filter gt.txt by that ID and draw the box on each frame.
Field mask. HSV green threshold → dilate and erode → fill the largest contour. Only search inside this area.
Candidates. Grayscale → white top-hat (kernel ~25 px) → threshold → connected components → loose size filter.
Check candidate recall. Is the true ball among the candidates in >85% of the frames where it's visible? Tune step 3 until it is.
Score each candidate by hand-chosen rules: size vs. y-position, circularity, brightness/color, local contrast, not on a field line. Pick the best one.
Kalman filter. Predict → choose the candidate near the prediction → update. Keep predicting when the ball is hidden, and reset after ~10 missed frames.
Tune parameters on a few train clips, then freeze them.
Evaluate on the test clips and your own annotated 1-minute clip: precision/recall by center distance, position error, FPS.
Report: pipeline, failed early attempts, failure cases, comparison with YOLO.
'''


def process_frame(frame):
    global PreFrame
    if PreFrame is None:
        PreFrame = frame.copy()
        return frame  # Return the first frame as is
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Apply a threshold to the HSV frame to isolate the green color
    subtracted_frame = cv2.absdiff(frame, PreFrame)
    PreFrame = frame.copy()    
    hsv_thresh = cv2.inRange(hsv_frame, (0, 30, 0), (255, 130, 255))  # Example threshold for green color
    gray_frame = cv2.cvtColor(subtracted_frame, cv2.COLOR_BGR2GRAY)
    median_frame = cv2.medianBlur(gray_frame, 5)
    dialted_frame = cv2.dilate(median_frame, None, iterations=4)
    eroded_frame = cv2.erode(dialted_frame, None, iterations=1)

    hough_frame = cv2.HoughCircles(gray_frame, cv2.HOUGH_GRADIENT, dp=0.2, minDist=10,
                                   param1=50, param2=30, minRadius=1, maxRadius=10)

    if hough_frame is not None:
        for x, y, r in hough_frame[0]:
            cv2.circle(frame, (int(x), int(y)), int(r), (0, 255, 0), 2)
    '''
    template_football = cv2.imread("Football.png", cv2.IMREAD_GRAYSCALE)
    # Perform template matching to find the ball in the frame
    result = cv2.matchTemplate(hough_frame, template_football, cv2.TM_CCOEFF_NORMED)
    threshold = 0.5  # Adjust this threshold based on your needs
    loc = cv2.minMaxLoc(result)
    if loc[1] >= threshold:
        # Ball detected, draw a rectangle around it
        top_left = loc[3]
        h, w = template_football.shape
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(hough_frame, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(hough_frame, "Ball Detected", (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    result_vis = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    '''


    # Placeholder for the non-NN ball tracker; returns the frame unchanged for now
    return frame  # Return the processed frame (currently just the eroded difference)


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
