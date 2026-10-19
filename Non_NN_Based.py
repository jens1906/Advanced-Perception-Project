import configparser
import random
from pathlib import Path
from unittest import result

import cv2
import numpy as np

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
kernel_size = 5
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
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

# ---------------- Kalman filter (added) ----------------
# State = (x, y, vx, vy), measurement = (x, y), one time step = one frame
kf = cv2.KalmanFilter(4, 2)
kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)
missed_frames = 11  # Start "lost" so the first frame starts a track


def kalman_filter_predict_and_update(kf, candidates, best_candidate):
    global missed_frames
    centers = (candidates[:, :2] + candidates[:, 2:4] / 2).astype(np.float32)

    # Predict the next state
    prediction = kf.predict()[:2, 0]

    # Update with the nearest candidate within 40 px, otherwise just keep the prediction
    distances = np.linalg.norm(centers - prediction, axis=1)
    if len(centers) and distances.min() < 40:
        kf.correct(centers[distances.argmin()].reshape(2, 1))
        missed_frames = 0
    else:
        missed_frames += 1
        # Lost for >10 frames: restart the track on the best-scoring candidate
        if missed_frames > 10 and best_candidate is not None:
            x, y, w, h = best_candidate
            kf.statePost = np.float32([[x + w / 2], [y + h / 2], [0], [0]])
            missed_frames = 0

    return kf.statePost[:2, 0].astype(int)
# -------------------------------------------------------


def process_frame(frame):
    # White pixels -> top-hat keeps only small objects -> open removes 1-2 px noise (erode deleted small balls)
    white_frame = cv2.inRange(frame, (125, 165, 175), (255, 255, 255))
    white_top_hat_frame = cv2.morphologyEx(white_frame, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    opened_frame = cv2.morphologyEx(white_top_hat_frame, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # Each blob becomes a candidate (stats row = x, y, w, h, area), keep only ball-sized ones
    _, _, stats, _ = cv2.connectedComponentsWithStats(opened_frame, connectivity=8)
    candidates = stats[1:][(stats[1:, 4] > 8) & (stats[1:, 4] < 600)]  # Row 0 is the background

    # Score: the ball is a lone white blob, so pick the one with the least other white around it
    best_candidate, best_score = None, -1
    for x, y, w, h, area in candidates:
        pad = max(w, h)
        around = white_frame[max(y - pad, 0):y + h + pad, max(x - pad, 0):x + w + pad]
        score = 1 - (np.count_nonzero(around) - area) / around.size
        if score > best_score:
            best_candidate, best_score = (x, y, w, h), score

    for x, y, w, h, area in candidates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
    if best_candidate is not None:
        x, y, w, h = best_candidate
        cv2.rectangle(frame, (x - 4, y - 4), (x + w + 4, y + h + 4), (0, 0, 255), 3)

    # ---------------- Kalman filter (added) ----------------
    # Yellow circle = tracked ball position
    ball_position = kalman_filter_predict_and_update(kf, candidates, best_candidate)
    cv2.circle(frame, (int(ball_position[0]), int(ball_position[1])), 12, (0, 255, 255), 3)
    # -------------------------------------------------------
    return frame

    '''
    # Template matching attempt
    greenmask_RGB_frame = cv2.inRange(frame, (125, 165, 175), (255, 255, 255))
    white_top_hat_frame = cv2.morphologyEx(greenmask_RGB_frame, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    morph_open_frame = cv2.morphologyEx(white_top_hat_frame, cv2.MORPH_OPEN, kernel)
    template = cv2.morphologyEx(cv2.imread("Football.png", cv2.IMREAD_GRAYSCALE), cv2.MORPH_CLOSE, kernel)
    result = cv2.matchTemplate(morph_open_frame, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.5
    locations = cv2.minMaxLoc(result)
    if locations[1] > threshold:
        best_match_location = locations[3]
        top_left = best_match_location
        bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
    return frame
    '''


def play_sequence(seq_dir):
    info = load_seqinfo(seq_dir)
    frame_paths = sorted(info["im_dir"].glob(f"*{info['ext']}"))
    delay_ms = max(1, int(100 / info["fps"]))

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
