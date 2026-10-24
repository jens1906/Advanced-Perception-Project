import random

import cv2
import numpy as np

from PreliminaryFunctions import TRAIN_DIR, FindSequences, PlaySequence
from Tuning import (BALL_FULL_AREA, FIELD_CLOSE_SIZE, FIELD_MARGIN, FIELD_MIN_PART, FIELD_SCALE, GATE_RADIUS,
                    GRASS_HUE_RANGE, GRASS_MIN_SATURATION, MAX_ELONGATION, MAX_MISSED_FRAMES, TOP_CANDIDATES)

PreFrame = None  # Global variable to store the previous frame
kernel_size = 5
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

# Tracking variables
lastBallPosition = None
missedFrames = 0

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


def ResetTracker():
    global lastBallPosition, missedFrames
    lastBallPosition, missedFrames = None, 0


def ProcessFrame(frame):
    global lastBallPosition, missedFrames
    # White pixels -> top-hat keeps only small objects -> open removes 1-2 px noise (erode deleted small balls)
    white_frame = cv2.inRange(frame, (125, 165, 175), (255, 255, 255))
    white_top_hat_frame = cv2.morphologyEx(white_frame, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40)))
    opened_frame = cv2.morphologyEx(white_top_hat_frame, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # Each blob becomes a candidate (stats row = x, y, w, h, area), keep only ball-sized ones
    _, _, stats, _ = cv2.connectedComponentsWithStats(opened_frame, connectivity=8)
    candidates = stats[1:][(stats[1:, 4] > 8) & (stats[1:, 4] < 600)]  # Row 0 is the background

    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    grassHue = int(np.argmax(cv2.calcHist([hsvFrame], [0], None, [180], [0, 180])))
    greenFrame = cv2.inRange(hsvFrame, (grassHue - GRASS_HUE_RANGE, GRASS_MIN_SATURATION, 0), (grassHue + GRASS_HUE_RANGE, 255, 255))

    smallMask = cv2.resize(greenFrame, None, fx=1 / FIELD_SCALE, fy=1 / FIELD_SCALE, interpolation=cv2.INTER_AREA)
    smallMask = cv2.morphologyEx(smallMask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (FIELD_CLOSE_SIZE, FIELD_CLOSE_SIZE)))
    contours, _ = cv2.findContours(smallMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fieldMask = np.zeros_like(smallMask)
    if contours:
        largestArea = max(cv2.contourArea(contour) for contour in contours)
        fieldContours = [contour for contour in contours if cv2.contourArea(contour) > FIELD_MIN_PART * largestArea]
        cv2.drawContours(fieldMask, [cv2.convexHull(np.vstack(fieldContours))], -1, 255, -1)
        fieldMask = cv2.dilate(fieldMask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (FIELD_MARGIN, FIELD_MARGIN)))
    fieldMask = cv2.resize(fieldMask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    centersX = candidates[:, 0] + candidates[:, 2] // 2
    centersY = candidates[:, 1] + candidates[:, 3] // 2
    candidates = candidates[fieldMask[centersY, centersX] > 0]

    # Score: the ball is a lone white blob, so pick the one with the least other white around it
    scores = []
    for x, y, w, h, area in candidates:
        pad = max(w, h)
        top, left = max(y - pad, 0), max(x - pad, 0)
        around = white_frame[top:y + h + pad, left:x + w + pad]
        isolation = 1 - (np.count_nonzero(around) - area) / around.size
        green = min(np.count_nonzero(greenFrame[top:y + h + pad, left:x + w + pad]) / max(around.size - w * h, 1), 1)
        moments = cv2.moments(opened_frame[y:y + h, x:x + w], True)
        spread = np.linalg.eigvalsh([[moments["mu20"], moments["mu11"]], [moments["mu11"], moments["mu02"]]])
        elongation = np.sqrt(spread[1] / max(spread[0], 1e-6))
        scores.append(isolation * green * min(area / BALL_FULL_AREA, 1) * (elongation < MAX_ELONGATION))

    for x, y, w, h, area in candidates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
    ball_position = None
    if scores:
        topIndices = np.argsort(scores)[::-1][:TOP_CANDIDATES]
        topCenters = candidates[topIndices, :2] + candidates[topIndices, 2:4] / 2
        chosen = topIndices[0]
        if lastBallPosition is not None:
            distances = np.linalg.norm(topCenters - lastBallPosition, axis=1)
            if distances.min() < GATE_RADIUS:
                chosen = topIndices[distances.argmin()]
                missedFrames = 0
            else:
                missedFrames += 1
        if lastBallPosition is None or missedFrames == 0 or missedFrames > MAX_MISSED_FRAMES:
            lastBallPosition = candidates[chosen, :2] + candidates[chosen, 2:4] / 2
            missedFrames = 0
        for x, y, w, h, area in candidates[topIndices]:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        x, y, w, h, area = candidates[chosen]
        cv2.rectangle(frame, (x - 4, y - 4), (x + w + 4, y + h + 4), (0, 0, 255), 3)
        ball_position = (x + w / 2, y + h / 2)

    return [frame, opened_frame, fieldMask], ball_position, candidates[:, :4]

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


'''
# Kalman filter attempt: locked onto noise because a candidate was almost always within 40 px of the prediction
kf = cv2.KalmanFilter(4, 2)
kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)
missed_frames = 11

def KalmanFilterPredictAndUpdate(kf, candidates, best_candidate):
    global missed_frames
    centers = (candidates[:, :2] + candidates[:, 2:4] / 2).astype(np.float32)
    prediction = kf.predict()[:2, 0]
    distances = np.linalg.norm(centers - prediction, axis=1)
    if len(centers) and distances.min() < 40:
        kf.correct(centers[distances.argmin()].reshape(2, 1))
        missed_frames = 0
    else:
        missed_frames += 1
        if missed_frames > 10 and best_candidate is not None:
            x, y, w, h = best_candidate
            kf.statePost = np.float32([[x + w / 2], [y + h / 2], [0], [0]])
            missed_frames = 0
    return kf.statePost[:2, 0].astype(int)
'''


if __name__ == "__main__":
    sequences = FindSequences(TRAIN_DIR)
    if not sequences:
        raise SystemExit(f"No sequences found under {TRAIN_DIR}")

    seq_dir = random.choice(sequences)
    print(f"Playing {seq_dir}")
    PlaySequence(seq_dir, ProcessFrame, ResetTracker)
