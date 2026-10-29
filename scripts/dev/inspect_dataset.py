from pathlib import Path
from collections import Counter
import pandas as pd
from PIL import Image

ROOT = Path("data/images/DatasetForFetus/")
EXCEL = Path("data/annotations/ObjectDetection.xlsx")

df = pd.read_excel(EXCEL)

# Class mapping candidates
class_mapping = {
    "thalami": "thalami",
    "midbrain": "midbrain",
    "palate": "palate",
    "IT": "fourth_ventricle",
    "CM": "cisterna_magna",
    "NT": "nuchal_translucency",
    "nasal tip": "nasal_tip",
    "nasal skin": "nasal_skin",
    "nasal bone": "nasal_bone",
}

print("Annotation row:", len(df))
print("Unique annotated filenames:", df["fname"].nunique())

print("\nStructure names:")
for name, count in df["structure"].value_counts().items():
    print(f"{name:15} {count}")

print("\nUnmapped structure names:")
print(sorted(set(df["structure"]) - set(class_mapping)))

# Check bounding boxes against image dimensions.
errors = []
checked = 0

for _, row in df.iterrows():
    matches = list(ROOT.rglob(str(row["fname"])))
    if not matches:
        errors.append((row["fname"], "image not found"))
        continue

    # The same filename could exist in multiple directories.
    image_path = matches[0]

    try:
        with Image.open(image_path) as img:
            width, height = img.size

        x1, y1 = row["w_min"], row["h_min"]
        x2, y2 = row["w_max"], row["h_max"]

        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            errors.append(
                (row["fname"], f"box {(x1, y1, x2, y2)}; image {(width, height)}")
            )
        checked += 1
    except Exception as exc:
        errors.append((row["fname"], str(exc)))

print(f"\nBounding boxes checked: {checked}")
print(f"Invalid boxes or image errors: {len(errors)}")

for error in errors[:20]:
    print(error)
