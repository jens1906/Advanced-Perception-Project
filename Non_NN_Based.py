import random
from pathlib import Path
import sys

import cv2
import numpy as np

from PreliminaryFunctions import TRAIN_DIR, FindSequences, PlaySequence

# Parameters
WHITE_LOW = (125, 165, 175)  # BGR lower bound for "white"
TOPHAT_SIZE = 40  # White things smaller than this survive the top-hat
MIN_AREA, MAX_AREA = 8, 600  # Ball-sized blobs in pixels
BALL_FULL_AREA = 15  # Blobs smaller than this get a lower score
MAX_ELONGATION = 1.8  # Ball is round, leftover line pieces are long
GRASS_HUE_RANGE = 15  # Hue +- around the most common hue counts as grass
GRASS_MIN_SATURATION = 60
PLAYER_MIN_HEIGHT, PLAYER_MAX_HEIGHT = 25, 350  # Non-grass blobs this tall and narrow are players
PLAYER_WEIGHT = 0.3  # Score multiplier for blobs on a player (white shirts, socks)
NO_CIRCLE_WEIGHT = 0.2  # Score multiplier for blobs where no circle edge is found
MIN_SCORE = 0.05  # Below this nothing counts as the ball


def ResetTracker():
    pass  # No tracking, every frame is detected on its own


def ProcessFrame(frame):
    # White pixels -> top-hat keeps only small white objects -> open removes 1-2 px noise
    white_frame = cv2.inRange(frame, WHITE_LOW, (255, 255, 255))
    top_hat_frame = cv2.morphologyEx(white_frame, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TOPHAT_SIZE, TOPHAT_SIZE)))
    opened_frame = cv2.morphologyEx(top_hat_frame, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # Each blob becomes a candidate (stats row = x, y, w, h, area), keep only ball-sized ones
    _, _, stats, _ = cv2.connectedComponentsWithStats(opened_frame, connectivity=8)
    candidates = stats[1:][(stats[1:, 4] > MIN_AREA) & (stats[1:, 4] < MAX_AREA)]  # Row 0 is the background

    # Grass = the most common hue in the frame
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    grass_hue = int(np.argmax(cv2.calcHist([hsv_frame], [0], None, [180], [0, 180])))
    green_frame = cv2.inRange(hsv_frame, (grass_hue - GRASS_HUE_RANGE, GRASS_MIN_SATURATION, 0), (grass_hue + GRASS_HUE_RANGE, 255, 255))

    # Players = tall, narrow blobs that are not grass
    not_green = cv2.morphologyEx(cv2.bitwise_not(green_frame), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    _, _, player_stats, _ = cv2.connectedComponentsWithStats(not_green, connectivity=8)
    player_frame = np.zeros_like(green_frame)
    for x, y, w, h, area in player_stats[1:]:
        if PLAYER_MIN_HEIGHT <= h <= PLAYER_MAX_HEIGHT and h >= 1.2 * w:
            cv2.rectangle(player_frame, (x, y), (x + w, y + int(0.8 * h)), 255, -1)  # Skip the feet, the ball is often there
    gray_frame = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3), 0)

    # Score: the ball is a lone, round white blob with grass around it, not on a player, with a circular edge
    ball_position, best_score = None, MIN_SCORE
    for x, y, w, h, area in candidates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        pad = max(w, h)
        window = (slice(max(y - pad, 0), y + h + pad), slice(max(x - pad, 0), x + w + pad))
        green = min(np.count_nonzero(green_frame[window]) / max(green_frame[window].size - w * h, 1), 1)
        isolation = 1 - (np.count_nonzero(white_frame[window]) - area) / white_frame[window].size
        moments = cv2.moments(opened_frame[y:y + h, x:x + w], True)
        spread = np.linalg.eigvalsh([[moments["mu20"], moments["mu11"]], [moments["mu11"], moments["mu02"]]])
        round_enough = np.sqrt(spread[1] / max(spread[0], 1e-6)) < MAX_ELONGATION
        score = green * isolation * min(area / BALL_FULL_AREA, 1) * round_enough
        if player_frame[y + h // 2, x + w // 2]:
            score *= PLAYER_WEIGHT
        if score > best_score:  # Only run the slow circle check when the blob could still win
            circles = cv2.HoughCircles(gray_frame[window], cv2.HOUGH_GRADIENT, dp=1, minDist=1000, param1=60, param2=12,
                                       minRadius=max(2, pad // 4), maxRadius=int(0.75 * pad) + 2)
            if circles is None:
                score *= NO_CIRCLE_WEIGHT
        if score > best_score:
            ball_position, best_score = (x + w / 2, y + h / 2), score

    if ball_position is not None:
        cv2.circle(frame, (int(ball_position[0]), int(ball_position[1])), 12, (0, 0, 255), 3)

    return [frame, opened_frame], ball_position, candidates[:, :4]


if __name__ == "__main__":
    sequences = FindSequences(TRAIN_DIR)
    if not sequences:
        raise SystemExit(f"No sequences found under {TRAIN_DIR}")

    seq_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else random.choice(sequences)
    print(f"Playing {seq_dir}")
    PlaySequence(seq_dir, ProcessFrame, ResetTracker)
