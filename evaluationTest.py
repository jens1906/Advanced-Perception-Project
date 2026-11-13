import csv
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from PreliminaryFunctions import VERIFY_MARGIN, FindSequences, LoadBallGroundTruth, LoadSequenceInfo, PrintVerificationResult, ScoreFrame

# Evaluation settings
USE_NN = True  # True = NN_Based (YOLO), False = Non_NN_Based
PLOT_ONLY = False  # True = skip the evaluation and redraw the plot from the saved CSV
TEST_DIR = Path("Data/SoccerNet/tracking/test")
TEST_ZIP = Path("Data/SoccerNet/tracking/test.zip")
PROGRESS_EVERY = 10
OUTPUT_NAME = f"evaluation_{'NN' if USE_NN else 'NonNN'}"

# Only import the chosen detector, NN_Based loads YOLO on import
if not PLOT_ONLY:
    if USE_NN:
        from NN_Based import ProcessFrame, ResetTracker
    else:
        from Non_NN_Based import ProcessFrame, ResetTracker


def FormatDuration(seconds):
    return str(timedelta(seconds=int(seconds)))


def Rate(count, frameCount):
    return 100 * count / frameCount if frameCount else 0.0


def PrintProgress(seqIndex, seqCount, name, done, total, start):
    elapsed = time.time() - start
    fps = done / elapsed if elapsed > 0 else 0.0
    remaining = (total - done) / fps if fps > 0 else 0.0
    finish = datetime.now() + timedelta(seconds=remaining)
    print(f"\r[seq {seqIndex}/{seqCount} {name}] frames {done}/{total} ({100 * done / total:.1f}%) | {fps:.1f} fps | "
          f"elapsed {FormatDuration(elapsed)} | ETA {FormatDuration(remaining)} | finish ~{finish:%H:%M}   ", end="", flush=True)


def EvaluateSequences(seqDirs):
    totalFrames = sum(LoadSequenceInfo(seqDir)["length"] for seqDir in seqDirs)
    print(f"Evaluating {'NN (YOLO)' if USE_NN else 'Non-NN'} on {len(seqDirs)} sequences, {totalFrames} frames")

    results, done, start = [], 0, time.time()
    for seqIndex, seqDir in enumerate(seqDirs, 1):
        ResetTracker()
        info = LoadSequenceInfo(seqDir)
        groundTruth = LoadBallGroundTruth(seqDir)
        hits, candidateHits, errors, frameCount = 0, 0, [], 0

        for framePath in sorted(info["im_dir"].glob(f"*{info['ext']}")):
            _, ballPosition, candidates = ProcessFrame(cv2.imread(str(framePath)))
            done += 1
            if done % PROGRESS_EVERY == 0:
                PrintProgress(seqIndex, len(seqDirs), info["name"], done, totalFrames, start)
            box = groundTruth.get(int(framePath.stem))
            if box is None:
                continue
            hit, candidateHit, error = ScoreFrame(ballPosition, candidates, box, VERIFY_MARGIN)
            hits += hit
            candidateHits += candidateHit
            errors += error
            frameCount += 1

        print("\r" + " " * 140 + "\r", end="")
        PrintVerificationResult(info["name"], hits, candidateHits, errors, frameCount)
        results.append({"name": info["name"], "hits": hits, "candidateHits": candidateHits, "errors": errors, "frameCount": frameCount})

    runtime = time.time() - start
    print(f"Finished in {FormatDuration(runtime)} ({done / runtime:.1f} fps)")
    return results


def SaveResults(results):
    csvPath = Path(f"{OUTPUT_NAME}.csv")
    with open(csvPath, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["name", "hits", "candidateHits", "frameCount", "meanError"])
        for result in results:
            meanError = f"{np.mean(result['errors']):.1f}" if result["errors"] else ""
            writer.writerow([result["name"], result["hits"], result["candidateHits"], result["frameCount"], meanError])
    print(f"Results saved to {csvPath}")


def LoadResults():
    with open(f"{OUTPUT_NAME}.csv", newline="") as file:
        return [{"name": row["name"], "hits": int(row["hits"]), "candidateHits": int(row["candidateHits"]),
                 "frameCount": int(row["frameCount"])} for row in csv.DictReader(file)]


def PlotResults(results):
    totalFrames = sum(result["frameCount"] for result in results)
    totalHitRate = Rate(sum(result["hits"] for result in results), totalFrames)
    totalRecall = Rate(sum(result["candidateHits"] for result in results), totalFrames)
    scored = [result for result in results if result["frameCount"]]
    names = [result["name"] for result in scored]
    hitRates = [Rate(result["hits"], result["frameCount"]) for result in scored]
    recalls = [Rate(result["candidateHits"], result["frameCount"]) for result in scored]
    label = "NN (YOLO)" if USE_NN else "Non-NN"

    plt.figure(figsize=(max(10, 0.35 * len(names)), 6))
    plt.plot(names, hitRates, "o-", color="blue", label="Hit rate")
    plt.plot(names, recalls, "o-", color="red", label="Candidate recall")
    plt.axhline(totalHitRate, color="blue", linestyle="--", label=f"Combined hit rate {totalHitRate:.1f}%")
    plt.axhline(totalRecall, color="red", linestyle="--", label=f"Combined recall {totalRecall:.1f}%")
    plt.xticks(rotation=90)
    plt.ylim(0, 100)
    plt.xlabel("Sequence")
    plt.ylabel("Rate (%)")
    plt.title(f"{label}  hit {totalHitRate:.1f}%  recall {totalRecall:.1f}%")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plotPath = Path(f"{OUTPUT_NAME}.png")
    plt.savefig(plotPath, dpi=150)
    print(f"Plot saved to {plotPath}")


if __name__ == "__main__":
    if PLOT_ONLY:
        PlotResults(LoadResults())
        raise SystemExit

    if not FindSequences(TEST_DIR):
        print(f"Extracting {TEST_ZIP} ...")
        zipfile.ZipFile(TEST_ZIP).extractall(TEST_DIR.parent)
    sequences = FindSequences(TEST_DIR)
    if not sequences:
        raise SystemExit(f"No sequences found under {TEST_DIR}")

    results = EvaluateSequences(sequences)
    totalHits = sum(result["hits"] for result in results)
    totalCandidateHits = sum(result["candidateHits"] for result in results)
    totalErrors = [error for result in results for error in result["errors"]]
    totalFrames = sum(result["frameCount"] for result in results)
    PrintVerificationResult("TOTAL", totalHits, totalCandidateHits, totalErrors, totalFrames)
    SaveResults(results)
    PlotResults(results)
