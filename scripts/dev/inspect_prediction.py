from ultralytics import YOLO

model = YOLO("models/best-50.pt")

results = model.predict(
    source="data/yolo/images/val/103.png",
    imgsz=640,
    conf=0.25,
    iou=0.5,
    device="cpu",
    verbose=False,
)

for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(float, box.xyxy[0])

        print(
            f"{model.names[class_id]:22s} "
            f"confidence={confidence:.3f} "
            f"xyxy=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
        )
