from ultralytics import YOLO

from ConvertToYolo import OUTPUT_DIR

# Training parameters
TRAIN_WEIGHTS = "yolo26n.pt"
TRAIN_IMAGE_SIZE = 1280
TRAIN_EPOCHS = 40
TRAIN_PATIENCE = 10

if __name__ == "__main__":
    model = YOLO(TRAIN_WEIGHTS)
    model.train(data=str(OUTPUT_DIR / "soccernet.yaml"), imgsz=TRAIN_IMAGE_SIZE, epochs=TRAIN_EPOCHS,
                patience=TRAIN_PATIENCE, batch=-1, amp=False, workers=4, name="soccernet")
