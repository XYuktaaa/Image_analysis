from pathlib import Path
from ultralytics import YOLO

MODEL = Path("models/best-50.pt")
IMAGE_DIR = Path("data/yolo/images/val")

if not MODEL.exists():
    raise FileNotFoundError(f"Model not found: {MODEL}")

images = sorted(IMAGE_DIR.glob("*.png"))

if not images:
    raise FileNotFoundError(f"No PNG images found in {IMAGE_DIR}")

model = YOLO(str(MODEL))

results = model.predict(
    source=str(images[0]),
    imgsz=640,
    conf=0.25,
    device="cpu",
    save=True,
    project="results/predictions",
    name="first_test",
)

print("Image:", images[0])
print("Predictions saved under results/predictions/first_test/")
