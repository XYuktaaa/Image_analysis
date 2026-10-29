# MediLens — AI-Powered Fetal Ultrasound Structure Detection

MediLens is a research prototype that uses a YOLOv8n object-detection model to draw bounding boxes around nine annotated fetal ultrasound structures. A Flask web interface accepts an image and displays model-generated detections and confidence scores.

>  This project is not a medical device, does not provide a diagnosis, and must not be used to make clinical decisions. Model outputs require qualified clinical review.

## Features

- Upload PNG, JPG, JPEG, or WEBP images through a Flask interface.
- Run inference with the expanded YOLOv8n model on CPU.
- Visualize annotated bounding boxes and confidence scores.
- Adjust the confidence threshold for exploratory inference.
- View a detection list sorted by confidence.
- Basic upload validation, file-size limit, and error handling.

## Detected classes

| ID | Class |
|---:|---|
| 0 | Thalami |
| 1 | Midbrain |
| 2 | Palate |
| 3 | Fourth ventricle |
| 4 | Cisterna magna |
| 5 | Nuchal translucency |
| 6 | Nasal tip |
| 7 | Nasal skin |
| 8 | Nasal bone |

## Model evaluation

Both models were evaluated on the same 72-image internal test set with 624 annotated objects.

| Metric | Original model | Expanded model |
|---|---:|---:|
| Precision | 87.24% | 93.92% |
| Recall | 90.51% | 95.01% |
| mAP@50 | 93.02% | 96.90% |
| mAP@50–95 | 48.59% | 57.48% |

These are the project's recorded internal-test results. No clinical validity though. The external non-standard image experiment was qualitative only: 92 images produced 852 detections at confidence 0.01; at confidence 0.25, 88 images had at least one detection and 4 had none, with 388 detections overall. No accuracy metrics can be calculated for those images without ground-truth labels.

## Project structure

```text
ultrasound-analysis-old/
├── run.py
├── models/
│   └── best-expanded-50.pt       # add your trained weights here
├── templates/
│   └── index.html
├── static/
│   └── css/style.css
├── uploads/                      # temporary originals are deleted after inference
├── results/web/                  # annotated output images
├── data/                          # dataset files (not included in this bundle)
└── README.md
```


Start the application:

```bash
python run.py
```

udes formal error analysis, data provenance checks, calibration/threshold evaluation on validation data, and automated tests for the web application.

## Technology stack

Python, YOLOv8n (Ultralytics), PyTorch, OpenCV, Flask, HTML, CSS, JavaScript.
