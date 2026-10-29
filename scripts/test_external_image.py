from pathlib import Path
import cv2
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent.parent
IMAGE = Path("/home/yukta/devel/wip/python/ultrasound-analysis-old/Untitled.png")
MODEL = BASE / "models" / "best-expanded-50.pt"

image = cv2.imread(str(IMAGE))
if image is None:
    raise FileNotFoundError(f"Cannot read image: {IMAGE}")

# Crop the ultrasound display area, excluding most borders and text.
h, w = image.shape[:2]
cropped = image[int(h * 0.12) : int(h * 0.83), int(w * 0.08) : int(w * 0.94)]

out = BASE / "results" / "comparisons" / "external_cropped.jpg"
out.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out), cropped)

model = YOLO(str(MODEL))

for name, source in [("original", image), ("cropped", cropped)]:
    result = model.predict(
        source=source,
        imgsz=640,
        conf=0.001,
        iou=0.4,
        device="cpu",
        verbose=False,
    )[0]

    print(f"\n{name}: {len(result.boxes)} detections")
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"  {model.names[cls]}: {conf:.4f}")

    if name == "cropped":
        result.save(filename=str(out.with_name("external_predictions.jpg")))
