from pathlib import Path
from collections import defaultdict
import shutil

import pandas as pd
from PIL import Image

ROOT = Path("data/images/DatasetForFetus")
TEST_DIR = ROOT / "Internal Test Set" / "Standard"
ANNOTATIONS = Path("data/annotations/ObjectDetection.xlsx")
OUTPUT = Path("data/yolo_internal_test")

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


def main():
    if OUTPUT.exists():
        raise FileExistsError(
            f"{OUTPUT} already exists. Remove it deliberately before rebuilding."
        )

    df = pd.read_excel(ANNOTATIONS)
    df["fname"] = df["fname"].astype(str).str.strip()
    df["structure"] = df["structure"].astype(str).str.strip()

    unknown = set(df["structure"]) - set(ALIASES)
    if unknown:
        raise ValueError(f"Unknown annotation classes: {unknown}")

    df["class_name"] = df["structure"].map(ALIASES)

    images = sorted(
        p
        for p in TEST_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )

    rows_by_name = {name: group for name, group in df.groupby("fname", sort=False)}

    missing = [p.name for p in images if p.name not in rows_by_name]
    if missing:
        raise ValueError(f"Test images without annotations: {missing}")

    try:
        image_out = OUTPUT / "images" / "test"
        label_out = OUTPUT / "labels" / "test"
        image_out.mkdir(parents=True)
        label_out.mkdir(parents=True)

        for source in images:
            name = source.name

            with Image.open(source) as img:
                width, height = img.size

            shutil.copy2(source, image_out / name)

            label_path = label_out / f"{source.stem}.txt"
            with label_path.open("w", encoding="utf-8") as f:
                for _, row in rows_by_name[name].iterrows():
                    x1 = float(row["w_min"])
                    y1 = float(row["h_min"])
                    x2 = float(row["w_max"])
                    y2 = float(row["h_max"])

                    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                        raise ValueError(f"Invalid box for {name}: {(x1, y1, x2, y2)}")

                    xc = ((x1 + x2) / 2) / width
                    yc = ((y1 + y2) / 2) / height
                    bw = (x2 - x1) / width
                    bh = (y2 - y1) / height

                    class_id = CLASS_TO_ID[row["class_name"]]
                    f.write(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        yaml_text = (
            f"path: {OUTPUT.resolve()}\n"
            "train: images/test\n"
            "val: images/test\n"
            "test: images/test\n"
            "names:\n" + "".join(f"  {i}: '{name}'\n" for i, name in enumerate(CLASSES))
        )
        (OUTPUT / "dataset.yaml").write_text(yaml_text, encoding="utf-8")

    except Exception:
        shutil.rmtree(OUTPUT, ignore_errors=True)
        raise

    print("Internal test dataset prepared.")
    print("Images:", len(images))
    print("Labels:", len(list(label_out.glob("*.txt"))))
    print("Dataset config:", OUTPUT / "dataset.yaml")


if __name__ == "__main__":
    main()
