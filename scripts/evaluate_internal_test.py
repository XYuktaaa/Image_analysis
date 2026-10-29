from pathlib import Path
from ultralytics import YOLO
import csv

DATA = "data/yolo_internal_test/dataset.yaml"

MODELS = {
    "original": "models/best-50.pt",
    "expanded": "models/best-expanded-50.pt",
}

OUTPUT = Path("results/internal_test")
OUTPUT.mkdir(parents=True, exist_ok=True)

rows = []

for name, weights in MODELS.items():
    print(f"\nEvaluating: {name}")

    model = YOLO(weights)

    metrics = model.val(
        data=DATA,
        split="test",
        imgsz=640,
        batch=8,
        conf=0.001,
        iou=0.5,
        device="cpu",
        plots=True,
        project=str(OUTPUT),
        name=name,
        exist_ok=True,
    )

    row = {
        "model": name,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
    }
    rows.append(row)

    print(f"Precision: {row['precision']:.4f}")
    print(f"Recall:    {row['recall']:.4f}")
    print(f"mAP@50:    {row['mAP50']:.4f}")
    print(f"mAP@50-95: {row['mAP50_95']:.4f}")

csv_path = OUTPUT / "comparison.csv"

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"\nComparison saved to {csv_path}")
