import random
from pathlib import Path
import sys

import cv2
import numpy as np

from PreliminaryFunctions import TRAIN_DIR, FindSequences, PlaySequence
from Tuning import (ACCEPT_SCORE, BALL_FULL_AREA, BODY_FRACTION, BODY_WEIGHT, CIRCLE_EDGE_THRESHOLD, CIRCLE_MIN_SHAPE,
                    CIRCLE_VOTES, CIRCLE_WEIGHT, FIELD_CLOSE_SIZE, FIELD_MARGIN, FIELD_MIN_PART, FIELD_SCALE,
                    GATE_GROWTH, GATE_RADIUS, GRASS_HUE_RANGE, GRASS_MIN_SATURATION, LINE_ANGLES, LINE_LENGTH,
                    LINE_SCALE, MAX_ELONGATION, MAX_MISSED_FRAMES, PLAYER_MAX_HEIGHT, PLAYER_MIN_AREA,
                    PLAYER_MIN_ASPECT, PLAYER_MIN_HEIGHT, START_SCORE, VELOCITY_DAMPING, VELOCITY_GAIN)

PreFrame = None  # Global variable to store the previous frame
kernel_size = 5
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

# Line removal kernels
LINE_KERNELS = []
for angle in np.arange(0, 180, 180 / LINE_ANGLES):
    lineKernel = np.zeros((LINE_LENGTH, LINE_LENGTH), np.uint8)
    offsetX, offsetY = np.cos(np.radians(angle)) * LINE_LENGTH / 2, np.sin(np.radians(angle)) * LINE_LENGTH / 2
    center = LINE_LENGTH // 2
    cv2.line(lineKernel, (int(center - offsetX), int(center - offsetY)), (int(center + offsetX), int(center + offsetY)), 1, 1)
    LINE_KERNELS.append(lineKernel)

# Tracking variables
lastBallPosition = None
ballVelocity = np.zeros(2)
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
    global lastBallPosition, ballVelocity, missedFrames
    lastBallPosition, ballVelocity, missedFrames = None, np.zeros(2), 0


def ProcessFrame(frame):
    global lastBallPosition, ballVelocity, missedFrames
    # White pixels -> top-hat keeps only small objects -> open removes 1-2 px noise (erode deleted small balls)
    white_frame = cv2.inRange(frame, (125, 165, 175), (255, 255, 255))
    smallWhite = cv2.resize(white_frame, None, fx=1 / LINE_SCALE, fy=1 / LINE_SCALE, interpolation=cv2.INTER_NEAREST)
    lineMask = np.zeros_like(smallWhite)
    for lineKernel in LINE_KERNELS:
        lineMask |= cv2.morphologyEx(smallWhite, cv2.MORPH_OPEN, lineKernel)
    lineMask = cv2.dilate(cv2.resize(lineMask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST), np.ones((3, 3), np.uint8))
    white_frame = cv2.bitwise_and(white_frame, cv2.bitwise_not(lineMask))
    white_top_hat_frame = cv2.morphologyEx(white_frame, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40)))
    opened_frame = cv2.morphologyEx(white_top_hat_frame, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

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
    inField = fieldMask[centersY, centersX] > 0

    peopleFrame = cv2.bitwise_and(cv2.bitwise_and(cv2.bitwise_not(greenFrame), fieldMask), cv2.bitwise_not(lineMask))
    peopleFrame = cv2.morphologyEx(cv2.morphologyEx(peopleFrame, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)), cv2.MORPH_CLOSE, np.ones((9, 5), np.uint8))
    _, _, playerStats, _ = cv2.connectedComponentsWithStats(peopleFrame, connectivity=8)
    bodyMask = np.zeros_like(peopleFrame)
    for playerX, playerY, playerWidth, playerHeight, playerArea in playerStats[1:]:
        if PLAYER_MIN_HEIGHT <= playerHeight <= PLAYER_MAX_HEIGHT and playerHeight >= PLAYER_MIN_ASPECT * playerWidth and playerArea >= PLAYER_MIN_AREA:
            cv2.rectangle(bodyMask, (playerX, playerY), (playerX + playerWidth, playerY + int(BODY_FRACTION * playerHeight)), 255, -1)
    grayFrame = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3), 0)

    # Score: the ball is a lone white blob, so pick the one with the least other white around it
    globalScores, trajectoryScores = [], []
    for (x, y, w, h, area), onField in zip(candidates, inField):
        pad = max(w, h)
        top, left = max(y - pad, 0), max(x - pad, 0)
        around = white_frame[top:y + h + pad, left:x + w + pad]
        isolation = 1 - (np.count_nonzero(around) - area) / around.size
        green = min(np.count_nonzero(greenFrame[top:y + h + pad, left:x + w + pad]) / max(around.size - w * h, 1), 1)
        moments = cv2.moments(opened_frame[y:y + h, x:x + w], True)
        spread = np.linalg.eigvalsh([[moments["mu20"], moments["mu11"]], [moments["mu11"], moments["mu02"]]])
        elongation = np.sqrt(spread[1] / max(spread[0], 1e-6))
        shape = isolation * min(area / BALL_FULL_AREA, 1) * (elongation < MAX_ELONGATION) * (BODY_WEIGHT if bodyMask[y + h // 2, x + w // 2] else 1)
        circleFound = False
        if shape > CIRCLE_MIN_SHAPE:
            radius = pad / 2
            circles = cv2.HoughCircles(grayFrame[top:y + h + pad, left:x + w + pad], cv2.HOUGH_GRADIENT, dp=1, minDist=1000, param1=CIRCLE_EDGE_THRESHOLD, param2=CIRCLE_VOTES, minRadius=max(2, int(0.5 * radius)), maxRadius=int(1.5 * radius) + 2)
            circleFound = circles is not None and np.linalg.norm(circles[0][:, :2] - (x + w / 2 - left, y + h / 2 - top), axis=1).min() <= max(2, 0.6 * radius)
        shape *= CIRCLE_WEIGHT + (1 - CIRCLE_WEIGHT) * circleFound
        trajectoryScores.append(shape)
        globalScores.append(shape * green * onField)
    globalScores, trajectoryScores = np.array(globalScores), np.array(trajectoryScores)

    centers = candidates[:, :2] + candidates[:, 2:4] / 2
    chosen, ball_position = None, None
    if lastBallPosition is not None:
        prediction = lastBallPosition + ballVelocity
        gateRadius = GATE_RADIUS + GATE_GROWTH * missedFrames
        distances = np.linalg.norm(centers - prediction, axis=1)
        inGate = (distances < gateRadius) & (trajectoryScores >= ACCEPT_SCORE)
        cv2.circle(frame, (int(prediction[0]), int(prediction[1])), int(gateRadius), (0, 255, 255), 2)
        if inGate.any():
            chosen = int(np.argmax(np.where(inGate, trajectoryScores * np.exp(-distances ** 2 / (2 * GATE_RADIUS ** 2)), -1)))
            ballVelocity = ballVelocity + VELOCITY_GAIN * (centers[chosen] - prediction)
            lastBallPosition, missedFrames = centers[chosen], 0
        else:
            lastBallPosition, ballVelocity, missedFrames, ball_position = prediction, ballVelocity * VELOCITY_DAMPING, missedFrames + 1, prediction
            if missedFrames > MAX_MISSED_FRAMES:
                lastBallPosition = None
    if len(globalScores) and globalScores.max() > 0:
        best = int(np.argmax(globalScores))
        if (lastBallPosition is None or missedFrames > 0) and globalScores[best] >= START_SCORE:
            chosen, lastBallPosition, ballVelocity, missedFrames = best, centers[best], np.zeros(2), 0
        if chosen is None and ball_position is None:
            ball_position = centers[best]
    if chosen is not None:
        ball_position = centers[chosen]

    for (x, y, w, h, area), onField in zip(candidates, inField):
        if onField:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
    if chosen is not None:
        x, y, w, h, area = candidates[chosen]
        cv2.rectangle(frame, (x - 4, y - 4), (x + w + 4, y + h + 4), (0, 0, 255), 3)
    elif ball_position is not None:
        cv2.circle(frame, (int(ball_position[0]), int(ball_position[1])), 5, (0, 0, 255), -1)

    return [frame, opened_frame], ball_position, candidates[:, :4]

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

    seq_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else random.choice(sequences)
    print(f"Playing {seq_dir}")
    PlaySequence(seq_dir, ProcessFrame, ResetTracker)
