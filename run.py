from pathlib import Path
from uuid import uuid4

import cv2
from flask import Flask, abort, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best-expanded-50.pt"
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results" / "web"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
CLASS_NAMES = {
    0: "Thalami",
    1: "Midbrain",
    2: "Palate",
    3: "Fourth Ventricle",
    4: "Cisterna Magna",
    5: "Nuchal Translucency",
    6: "Nasal Tip",
    7: "Nasal Skin",
    8: "Nasal Bone",
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024  # 12 MB upload limit
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model weights not found: {MODEL_PATH}\n"
        "Place best-expanded-50.pt inside the project's models/ directory."
    )

model = YOLO(str(MODEL_PATH))


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/analyze")
def analyze():
    uploaded = request.files.get("image")
    if uploaded is None or not uploaded.filename:
        return render_template("index.html", error="Choose an image file first."), 400
    if not allowed_file(uploaded.filename):
        return render_template(
            "index.html", error="Unsupported file type. Upload PNG, JPG, JPEG, or WEBP."
        ), 400

    # Use a generated filename rather than trusting the client-supplied name.
    extension = uploaded.filename.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid4().hex}.{extension}"
    upload_path = UPLOAD_DIR / secure_filename(unique_name)
    uploaded.save(upload_path)

    try:
        image = cv2.imread(str(upload_path))
        if image is None:
            return render_template(
                "index.html", error="The uploaded file could not be read as an image."
            ), 400

        confidence = request.form.get("confidence", "0.25")
        try:
            confidence = float(confidence)
        except ValueError:
            confidence = 0.25
        confidence = min(0.90, max(0.05, confidence))

        result = model.predict(
            source=image,
            imgsz=640,
            conf=confidence,
            iou=0.4,
            device="cpu",
            verbose=False,
        )[0]

        output_name = f"result_{uuid4().hex}.jpg"
        output_path = RESULT_DIR / output_name
        plotted = result.plot()
        if not cv2.imwrite(str(output_path), plotted):
            raise RuntimeError("Could not save the annotated result image.")

        detections = []
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            score = float(box.conf[0].item())
            coordinates = [round(float(v), 1) for v in box.xyxy[0].tolist()]
            detections.append({
                "name": CLASS_NAMES.get(class_id, model.names.get(class_id, str(class_id))),
                "confidence": round(score * 100, 1),
                "coordinates": coordinates,
            })
        detections.sort(key=lambda item: item["confidence"], reverse=True)

        return render_template(
            "index.html",
            result_url=f"/results/{output_name}",
            original_name=secure_filename(uploaded.filename),
            detections=detections,
            confidence=confidence,
        )
    except Exception as exc:
        app.logger.exception("Image inference failed")
        return render_template(
            "index.html", error=f"Could not analyze this image. Check the terminal logs and try another file."
        ), 500
    finally:
        # Do not retain uploaded originals. Annotated outputs are kept for the demo.
        try:
            upload_path.unlink(missing_ok=True)
        except OSError:
            app.logger.warning("Could not remove temporary upload: %s", upload_path)


@app.get("/results/<path:filename>")
def result_file(filename):
    # Only serve generated result files from the dedicated output directory.
    if Path(filename).name != filename:
        abort(404)
    return send_from_directory(RESULT_DIR, filename, as_attachment=False)


@app.errorhandler(413)
def too_large(_error):
    return render_template("index.html", error="File is too large. Maximum upload size is 12 MB."), 413


if __name__ == "__main__":
    print("Starting MediLens at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
