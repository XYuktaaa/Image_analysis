from ultralytics import YOLO

DATA = "data/yolo_expanded/dataset.yaml"

for name, weights in [
    ("Original model", "models/best-50.pt"),
    ("Expanded model", "models/best-expanded-50.pt"),
]:
    print(f"\n{'=' * 50}\n{name}\n{'=' * 50}")

    model = YOLO(weights)
    metrics = model.val(
        data=DATA,
        split="val",
        imgsz=640,
        batch=8,
        device="cpu",
        plots=False,
    )

    print(f"Precision:  {metrics.box.mp:.4f}")
    print(f"Recall:     {metrics.box.mr:.4f}")
    print(f"mAP@50:     {metrics.box.map50:.4f}")
    print(f"mAP@50-95:  {metrics.box.map:.4f}")
