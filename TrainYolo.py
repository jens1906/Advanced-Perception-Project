import time
from datetime import datetime, timedelta

from ultralytics import YOLO

from ConvertToYolo import OUTPUT_DIR

# Training parameters
TRAIN_WEIGHTS = "yolo26n.pt"
TRAIN_IMAGE_SIZE = 1280
TRAIN_EPOCHS = 40
TRAIN_PATIENCE = 10


def FormatDuration(seconds):
    return str(timedelta(seconds=int(seconds)))


def AddProgressTracker(model):
    state = {"start": None}

    def OnTrainStart(trainer):
        state["start"] = time.time()
        print(f"\nTraining started {datetime.now():%H:%M:%S}, {trainer.epochs} epochs planned")

    def OnEpochEnd(trainer):
        done = trainer.epoch + 1
        elapsed = time.time() - state["start"]
        perEpoch = elapsed / done
        remaining = perEpoch * (trainer.epochs - done)
        finish = datetime.now() + timedelta(seconds=remaining)
        mAP = trainer.metrics.get("metrics/mAP50-95(B)", float("nan"))
        print(f"\n[Progress] epoch {done}/{trainer.epochs} ({100 * done / trainer.epochs:.0f}%) | "
              f"elapsed {FormatDuration(elapsed)} | {FormatDuration(perEpoch)}/epoch | "
              f"remaining ~{FormatDuration(remaining)} | finish ~{finish:%a %H:%M} | mAP50-95 {mAP:.3f}\n")

    def OnTrainEnd(trainer):
        print(f"\nTraining finished in {FormatDuration(time.time() - state['start'])}, best weights: {trainer.best}")

    model.add_callback("on_train_start", OnTrainStart)
    model.add_callback("on_fit_epoch_end", OnEpochEnd)
    model.add_callback("on_train_end", OnTrainEnd)


if __name__ == "__main__":
    model = YOLO(TRAIN_WEIGHTS)
    AddProgressTracker(model)
    model.train(data=str(OUTPUT_DIR / "soccernet.yaml"), imgsz=TRAIN_IMAGE_SIZE, epochs=TRAIN_EPOCHS,
                patience=TRAIN_PATIENCE, batch=-1, amp=False, workers=4, name="soccernet")
