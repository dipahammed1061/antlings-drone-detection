"""
TASK 2 — MODEL TRAINING
========================
Fine-tunes YOLOv8s on the remapped VisDrone dataset.
Uses imgsz=1280 for better small object detection.

Run locally (RTX 3060):
    conda activate visdrone
    cd F:/ANTS
    python scripts/train.py

For Colab: upload this file and change BASE paths below.
"""

from ultralytics import YOLO
from pathlib import Path
import torch

# ── VERIFY GPU ────────────────────────────────────────────────────────────────
print("=" * 60)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
print("=" * 60)

# ── CONFIG ────────────────────────────────────────────────────────────────────
YAML_PATH    = "F:/ANTS/visdrone_remapped.yaml"
PROJECT_DIR  = "F:/ANTS/results"
RUN_NAME     = "yolov8s_visdrone_v1"

# RTX 3060 (12GB VRAM) settings
# If you get OOM error, reduce batch to 4
EPOCHS  = 50
IMGSZ   = 960
BATCH   = 4     # safe for 3060 at 1280px; reduce to 4 if OOM

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
# Start from pretrained COCO weights (transfer learning)
model = YOLO("yolov8s.pt")

print(f"\nStarting training:")
print(f"  Model:      YOLOv8s (pretrained)")
print(f"  Dataset:    {YAML_PATH}")
print(f"  Epochs:     {EPOCHS}")
print(f"  Image size: {IMGSZ}")
print(f"  Batch size: {BATCH}")
print(f"  Output:     {PROJECT_DIR}/{RUN_NAME}")
print()

# ── TRAIN ─────────────────────────────────────────────────────────────────────
results = model.train(
    data       = YAML_PATH,
    epochs     = EPOCHS,
    imgsz      = IMGSZ,
    batch      = BATCH,
    device     = 0,             # GPU 0

    # Augmentation (important for small objects)
    augment    = True,
    mosaic     = 0.5,           # mosaic augmentation
    mixup      = 0.0,
    degrees    = 10,            # rotation
    scale      = 0.5,           # scale jitter
    fliplr     = 0.5,
    flipud     = 0.0,

    # Training settings
    optimizer  = "AdamW",
    lr0        = 0.001,
    patience   = 15,            # early stopping
    workers    = 0,

    # Output
    project    = PROJECT_DIR,
    name       = RUN_NAME,
    exist_ok   = True,
    save       = True,
    plots      = True,          # saves training curves

    # Logging
    verbose    = True,
)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"Best weights: {PROJECT_DIR}/{RUN_NAME}/weights/best.pt")
print(f"Results:      {PROJECT_DIR}/{RUN_NAME}/")
print("\nNext step: run scripts/detect.py")
