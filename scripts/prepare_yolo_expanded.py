from pathlib import Path
from collections import defaultdict
import hashlib
import random
import shutil

import pandas as pd
from PIL import Image

ROOT = Path("data/images/DatasetForFetus")
ANNOTATIONS = Path("data/annotations/ObjectDetection.xlsx")
OUTPUT = Path("data/yolo_expanded")

FOLDERS = [
    ROOT / "Set1-Training&Validation Sets CNN" / "Standard",
    ROOT / "Set2-Training&Validation Sets ANN Scoring system" / "Standard",
]

TEST_FOLDERS = [
    ROOT / "Internal Test Set" / "Standard",
]

# Keep class IDs compatible with the existing trained model.
CLASSES = [
    "thalami",
    "midbrain",
    "palate",
    "fourth_ventricle",
    "cisterna_magna",
    "nuchal_translucency",
    "nasal_tip",
    "nasal_skin",
    "nasal_bone",
]

ALIASES = {
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

CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}
SEED = 42
TRAIN_RATIO = 0.8
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def image_files(folder):
    if not folder.exists():
        raise FileNotFoundError(f"Dataset folder not found: {folder}")

    return sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_hash(path):
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def convert_box(row, width, height):
    x1, y1 = float(row["w_min"]), float(row["h_min"])
    x2, y2 = float(row["w_max"]), float(row["h_max"])

    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(
            f"Invalid box for {row['fname']}: "
            f"{(x1, y1, x2, y2)}, image size {(width, height)}"
        )

    return (
        ((x1 + x2) / 2) / width,
        ((y1 + y2) / 2) / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    )


def main():
    if OUTPUT.exists():
        raise FileExistsError(
            f"{OUTPUT} already exists. Choose a new output path "
            "or deliberately remove the old expanded dataset."
        )

    df = pd.read_excel(ANNOTATIONS)
    df["fname"] = df["fname"].astype(str).str.strip()
    df["structure"] = df["structure"].astype(str).str.strip()

    unknown = set(df["structure"]) - set(ALIASES)
    if unknown:
        raise ValueError(f"Unmapped annotation classes: {unknown}")

    df["class_name"] = df["structure"].map(ALIASES)

    # Index all standard candidate images by filename.
    candidates = {}
    for folder in FOLDERS:
        for path in image_files(folder):
            if path.name in candidates:
                raise ValueError(
                    f"Repeated filename across source folders: {path.name}"
                )
            candidates[path.name] = path

    # Hash internal test images to prevent exact-image leakage.
    test_hashes = set()
    for folder in TEST_FOLDERS:
        for path in image_files(folder):
            test_hashes.add(image_hash(path))

    rows_by_name = {name: group for name, group in df.groupby("fname", sort=False)}

    annotated_names = sorted(set(rows_by_name) & set(candidates))
    if not annotated_names:
        raise RuntimeError("No annotated candidate images found.")

    # Exclude exact duplicates of internal test images and duplicate
    # content across the candidate pool.
    seen_hashes = set(test_hashes)
    eligible = []
    excluded_test_duplicates = 0
    excluded_candidate_duplicates = 0

    for name in annotated_names:
        path = candidates[name]
        digest = image_hash(path)

        if digest in test_hashes:
            excluded_test_duplicates += 1
            continue

        if digest in seen_hashes:
            excluded_candidate_duplicates += 1
            continue

        seen_hashes.add(digest)
        eligible.append(name)

    if len(eligible) < 2:
        raise RuntimeError("Not enough unique annotated images to split.")

    random.Random(SEED).shuffle(eligible)
    split_index = int(len(eligible) * TRAIN_RATIO)

    splits = {
        "train": eligible[:split_index],
        "val": eligible[split_index:],
    }

    try:
        for split, names in splits.items():
            image_dir = OUTPUT / "images" / split
            label_dir = OUTPUT / "labels" / split
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)

            for name in names:
                source = candidates[name]

                with Image.open(source) as img:
                    width, height = img.size

                shutil.copy2(source, image_dir / name)

                label_path = label_dir / f"{Path(name).stem}.txt"
                with label_path.open("w", encoding="utf-8") as f:
                    for _, row in rows_by_name[name].iterrows():
                        class_id = CLASS_TO_ID[row["class_name"]]
                        box = convert_box(row, width, height)
                        f.write(
                            f"{class_id} "
                            f"{box[0]:.6f} {box[1]:.6f} "
                            f"{box[2]:.6f} {box[3]:.6f}\n"
                        )

        yaml_text = (
            f"path: {OUTPUT.resolve()}\n"
            "train: images/train\n"
            "val: images/val\n"
            "names:\n" + "".join(f"  {i}: '{name}'\n" for i, name in enumerate(CLASSES))
        )
        (OUTPUT / "dataset.yaml").write_text(yaml_text, encoding="utf-8")

    except Exception:
        shutil.rmtree(OUTPUT, ignore_errors=True)
        raise

    print("Expanded YOLO dataset created.")
    print("Unique annotated images:", len(eligible))
    print("Training images:", len(splits["train"]))
    print("Validation images:", len(splits["val"]))
    print("Excluded exact internal-test duplicates:", excluded_test_duplicates)
    print("Excluded duplicate candidate images:", excluded_candidate_duplicates)
    print("Internal test set remains outside the output dataset.")
    print("Dataset config:", OUTPUT / "dataset.yaml")
    print("\nClass IDs:")
    for class_id, name in enumerate(CLASSES):
        print(f"  {class_id}: {name}")


if __name__ == "__main__":
    main()
