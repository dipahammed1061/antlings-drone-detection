"""
TASK 1 — PREPROCESSING: Class Remapping Script
================================================
VisDrone has 10 classes. We remap to only 2:
    human (0) ← pedestrian (0) + people (1)
    car   (1) ← car (3) + van (4)
    All other classes are DROPPED.

Run:
    conda activate visdrone
    cd F:/ANTS
    python scripts/preprocess.py
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATASET_ROOT = Path("F:/ANTS/VisDrone_Dataset")
OUTPUT_ROOT  = Path("F:/ANTS/outputs/preprocessed")

SPLITS = {
    "train": DATASET_ROOT / "VisDrone2019-DET-train",
    "val":   DATASET_ROOT / "VisDrone2019-DET-val",
    "test":  DATASET_ROOT / "VisDrone2019-DET-test-dev",
}

# VisDrone original class IDs → our new class IDs
# 0=pedestrian, 1=people, 2=bicycle, 3=car, 4=van,
# 5=truck, 6=tricycle, 7=awning-tricycle, 8=bus, 9=motor
REMAP = {
    0: 0,   # pedestrian → human
    1: 0,   # people     → human
    3: 1,   # car        → car
    4: 1,   # van        → car
    # everything else: dropped
}

# ── STATS ─────────────────────────────────────────────────────────────────────
stats = {
    "train": {"human": 0, "car": 0, "dropped": 0, "images_kept": 0, "images_empty": 0},
    "val":   {"human": 0, "car": 0, "dropped": 0, "images_kept": 0, "images_empty": 0},
    "test":  {"human": 0, "car": 0, "dropped": 0, "images_kept": 0, "images_empty": 0},
}


def remap_label_file(src_label: Path, dst_label: Path) -> dict:
    """
    Read a VisDrone YOLO label file, remap classes, drop unwanted,
    write new label file. Returns count dict.
    """
    counts = {"human": 0, "car": 0, "dropped": 0, "kept_lines": 0}

    if not src_label.exists():
        dst_label.write_text("")
        return counts

    lines = src_label.read_text().strip().splitlines()
    new_lines = []

    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        orig_cls = int(parts[0])

        if orig_cls in REMAP:
            new_cls = REMAP[orig_cls]
            new_line = f"{new_cls} {' '.join(parts[1:])}"
            new_lines.append(new_line)
            if new_cls == 0:
                counts["human"] += 1
            else:
                counts["car"] += 1
            counts["kept_lines"] += 1
        else:
            counts["dropped"] += 1

    dst_label.write_text("\n".join(new_lines))
    return counts


def process_split(split_name: str, split_path: Path):
    src_images = split_path / "images"
    src_labels = split_path / "labels"

    dst_images = OUTPUT_ROOT / split_name / "images"
    dst_labels = OUTPUT_ROOT / split_name / "labels"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    image_files = sorted(list(src_images.glob("*.jpg")) +
                         list(src_images.glob("*.png")))

    print(f"\n[{split_name.upper()}] Processing {len(image_files)} images...")

    s = stats[split_name]

    for img_path in tqdm(image_files, desc=f"  {split_name}"):
        label_path = src_labels / (img_path.stem + ".txt")
        dst_img    = dst_images / img_path.name
        dst_lbl    = dst_labels / (img_path.stem + ".txt")

        # Remap labels
        counts = remap_label_file(label_path, dst_lbl)

        # Copy image only if it has at least one kept annotation
        if counts["kept_lines"] > 0:
            shutil.copy2(img_path, dst_img)
            s["images_kept"] += 1
        else:
            # Still copy image but note it's empty (YOLO handles empty labels)
            shutil.copy2(img_path, dst_img)
            s["images_empty"] += 1

        s["human"]   += counts["human"]
        s["car"]     += counts["car"]
        s["dropped"] += counts["dropped"]

    print(f"  humans: {s['human']:,}  cars: {s['car']:,}  "
          f"dropped annotations: {s['dropped']:,}")
    print(f"  images with annotations: {s['images_kept']:,}  "
          f"images empty after remap: {s['images_empty']:,}")


def write_yaml():
    """Write the new 2-class visdrone.yaml for YOLO training."""
    yaml_path = Path("F:/ANTS/visdrone_remapped.yaml")
    content = f"""# VisDrone Remapped — 2 classes: human, car
path: F:/ANTS/outputs/preprocessed

train: train/images
val:   val/images
test:  test/images

nc: 2
names:
  0: human
  1: car
"""
    yaml_path.write_text(content)
    print(f"\nYAML written to: {yaml_path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("VisDrone Preprocessing — Class Remapping")
    print("=" * 60)
    print("Mapping:  pedestrian + people → human (0)")
    print("          car + van           → car   (1)")
    print("          all others          → DROPPED")
    print("=" * 60)

    for split_name, split_path in SPLITS.items():
        if not split_path.exists():
            print(f"WARNING: {split_path} not found, skipping.")
            continue
        process_split(split_name, split_path)

    write_yaml()

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Output: {OUTPUT_ROOT}")
    print(f"YAML:   F:/ANTS/visdrone_remapped.yaml")
    print("\nNext step: run notebooks/01_EDA.ipynb")
