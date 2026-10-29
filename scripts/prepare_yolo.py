from pathlib import Path
from collections import defaultdict
import random
import shutil

import pandas as pd
from PIL import Image

ROOT = Path("data/images/DatasetForFetus")
ANNOTATIONS = Path("data/annotations/ObjectDetection.xlsx")
OUTPUT = Path("data/yolo")

SET1 = ROOT / "Set1-Training&Validation Sets CNN" / "Standard"
INTERNAL_TEST = ROOT / "Internal Test Set" / "Standard"

# Preserve the original annotation labels.
CLASSES = [
    "thalami",
    "midbrain",
    "palate",
    "IT",
    "CM",
    "NT",
    "nasal tip",
    "nasal skin",
    "nasal bone",
]

CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}

SEED = 42
TRAIN_RATIO = 0.8


def find_images(folder):
    return {
        p.name: p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    }


def convert_box(row, width, height):
    # Spreadsheet coordinates: h = y, w = x.
    x1 = float(row["w_min"])
    y1 = float(row["h_min"])
    x2 = float(row["w_max"])
    y2 = float(row["h_max"])

    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(
            f"Invalid box for {row['fname']}: "
            f"{(x1, y1, x2, y2)} with image size {(width, height)}"
        )

    x_center = ((x1 + x2) / 2) / width
    y_center = ((y1 + y2) / 2) / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height

    return x_center, y_center, box_width, box_height


def main():
    df = pd.read_excel(ANNOTATIONS)
    df["fname"] = df["fname"].astype(str).str.strip()
    df["structure"] = df["structure"].astype(str).str.strip()

    unknown = set(df["structure"]) - set(CLASS_TO_ID)
    if unknown:
        raise ValueError(f"Unknown classes in spreadsheet: {unknown}")

    set1_images = find_images(SET1)
    internal_images = find_images(INTERNAL_TEST)

    # Only use annotated images actually in Set 1 for training/validation.
    set1_names = sorted(set(df["fname"]) & set(set1_images))
    internal_names = sorted(set(df["fname"]) & set(internal_images))

    if not set1_names:
        raise RuntimeError("No annotated images found in Set 1.")
    if set(set1_names) & set(internal_names):
        raise RuntimeError("An image appears in both Set 1 and internal test.")

    # Group all annotation rows by filename.
    rows_by_name = {name: group for name, group in df.groupby("fname", sort=False)}

    random.Random(SEED).shuffle(set1_names)
    split_index = int(len(set1_names) * TRAIN_RATIO)
    splits = {
        "train": set1_names[:split_index],
        "val": set1_names[split_index:],
    }

    # Refuse to overwrite an existing output dataset.
    if OUTPUT.exists():
        raise FileExistsError(
            f"{OUTPUT} already exists. Rename or remove it deliberately "
            "before regenerating the dataset."
        )

    try:
        for split, names in splits.items():
            image_dir = OUTPUT / "images" / split
            label_dir = OUTPUT / "labels" / split
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)

            for name in names:
                source = set1_images[name]
                with Image.open(source) as img:
                    width, height = img.size

                shutil.copy2(source, image_dir / name)

                label_path = label_dir / f"{Path(name).stem}.txt"
                with label_path.open("w", encoding="utf-8") as f:
                    for _, row in rows_by_name[name].iterrows():
                        class_id = CLASS_TO_ID[row["structure"]]
                        box = convert_box(row, width, height)
                        f.write(
                            f"{class_id} "
                            f"{box[0]:.6f} {box[1]:.6f} "
                            f"{box[2]:.6f} {box[3]:.6f}\n"
                        )

        # Write dataset configuration.
        yaml_path = OUTPUT / "dataset.yaml"
        yaml_path.write_text(
            "path: " + str(OUTPUT.resolve()) + "\n"
            "train: images/train\n"
            "val: images/val\n"
            "names:\n"
            + "".join(f"  {i}: '{name}'\n" for i, name in enumerate(CLASSES)),
            encoding="utf-8",
        )

    except Exception:
        shutil.rmtree(OUTPUT, ignore_errors=True)
        raise

    print("YOLO dataset created.")
    print("Training images:", len(splits["train"]))
    print("Validation images:", len(splits["val"]))
    print("Internal test images with annotations:", len(internal_names))
    print("Dataset config:", yaml_path)
    print("\nClass IDs:")
    for name, class_id in CLASS_TO_ID.items():
        print(f"  {class_id}: {name}")


if __name__ == "__main__":
    main()
