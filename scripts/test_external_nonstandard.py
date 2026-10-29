from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent

IMAGE_DIR = BASE_DIR / "data/images/DatasetForFetus/External Test Set/Non-standard"

MODEL_PATH = BASE_DIR / "models/best-expanded-50.pt"

OUTPUT_DIR = BASE_DIR / "results/external_nonstandard"

model = YOLO(str(MODEL_PATH))

image_paths = sorted(
    p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
)

if not image_paths:
    raise RuntimeError(f"No images found in {IMAGE_DIR}")

print(f"Images found: {len(image_paths)}")

total_detections = 0
images_with_detections = 0

for image_path in image_paths:
    result = model.predict(
        source=str(image_path),
        imgsz=640,
        conf=0.25,
        iou=0.4,
        device="cpu",
        verbose=False,
    )[0]

    count = len(result.boxes)
    total_detections += count

    if count:
        images_with_detections += 1

    result.save(filename=str(OUTPUT_DIR / image_path.name))

    print(f"{image_path.name}: {count} detections")

print("\n--- Summary ---")
print(f"Images tested: {len(image_paths)}")
print(f"Images with detections: {images_with_detections}")
print(f"Images with zero detections: {len(image_paths) - images_with_detections}")
print(f"Total detections: {total_detections}")
print(f"Annotated images saved in: {OUTPUT_DIR}")
