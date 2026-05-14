"""
IMPROVED DETECTION — SAHI (Slicing Aided Hyper Inference)
==========================================================
SAHI slices large drone images into overlapping patches,
runs detection on each patch, then merges results using NMM.

Why this helps:
- A 15px human in 1920x1080 becomes ~45px in a 640x640 slice
- Dramatically improves recall for tiny/distant humans
- Reduces missed detections in dense crowd scenes

Usage:
    python scripts/detect_sahi.py
    python scripts/detect_sahi.py --source path/to/images
    python scripts/detect_sahi.py --compare   (side-by-side comparison)

Requirements:
    pip install sahi
"""

import cv2
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# SAHI imports
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image_as_pil

# ── CONFIG ────────────────────────────────────────────────────────────────────
WEIGHTS     = "F:/ANTS/results/yolov8s_visdrone_v1/weights/best.pt"
CONF_THRESH = 0.20        # Lower than standard (0.25) — SAHI handles FP via NMM
IOU_THRESH  = 0.45

# SAHI slicing parameters
SLICE_H     = 640         # Height of each slice
SLICE_W     = 640         # Width of each slice
OVERLAP     = 0.2         # 20% overlap between slices (catches objects on edges)
POSTPROCESS = "NMM"       # Non-Maximum Merging (better than NMS for merged results)
POSTPROCESS_THRESH = 0.5  # Merge threshold

OUTPUT_DIR  = Path("F:/ANTS/results/images/detections_sahi")
COMPARE_DIR = Path("F:/ANTS/results/images/sahi_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
COMPARE_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES  = {0: "human", 1: "car"}
COLOR_HUMAN  = (0, 220, 80)    # green
COLOR_CAR    = (50, 50, 220)   # red/blue


def load_sahi_model():
    """Load YOLOv8 model via SAHI wrapper."""
    print("Loading model with SAHI wrapper...")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type   = "ultralytics",
        model_path   = WEIGHTS,
        confidence_threshold = CONF_THRESH,
        device       = "cuda:0",
    )
    print("Model loaded.")
    return detection_model


def run_sahi_on_image(detection_model, img_path: Path) -> tuple:
    """
    Run SAHI sliced inference on a single image.
    Returns (annotated_frame, human_count, car_count)
    """
    # Read image
    frame = cv2.imread(str(img_path))
    if frame is None:
        return None, 0, 0

    H, W = frame.shape[:2]

    # Run SAHI sliced prediction
    result = get_sliced_prediction(
        str(img_path),
        detection_model,
        slice_height          = SLICE_H,
        slice_width           = SLICE_W,
        overlap_height_ratio  = OVERLAP,
        overlap_width_ratio   = OVERLAP,
        postprocess_type      = POSTPROCESS,
        postprocess_match_threshold = POSTPROCESS_THRESH,
        verbose               = 0,
    )

    # Parse results
    human_count = 0
    car_count   = 0
    annotated   = frame.copy()

    for pred in result.object_prediction_list:
        cls_id = pred.category.id
        conf   = pred.score.value
        bbox   = pred.bbox  # BoundingBox object

        x1 = int(bbox.minx)
        y1 = int(bbox.miny)
        x2 = int(bbox.maxx)
        y2 = int(bbox.maxy)

        # Clip to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        color = COLOR_HUMAN if cls_id == 0 else COLOR_CAR
        name  = CLASS_NAMES.get(cls_id, str(cls_id))

        if cls_id == 0:
            human_count += 1
        else:
            car_count += 1

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Count overlay panel
    overlay = annotated.copy()
    cv2.rectangle(overlay, (8, 8), (280, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
    cv2.putText(annotated, f"Humans: {human_count}", (18, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_HUMAN, 2)
    cv2.putText(annotated, f"Cars:   {car_count}", (18, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_CAR, 2)
    cv2.putText(annotated, "SAHI Enhanced", (18, 88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    return annotated, human_count, car_count


def run_standard_on_image(img_path: Path, conf=0.25) -> tuple:
    """Run standard YOLO inference for comparison."""
    from ultralytics import YOLO
    model = YOLO(WEIGHTS)

    frame = cv2.imread(str(img_path))
    results = model(frame, conf=conf, iou=IOU_THRESH, verbose=False)[0]

    human_count = int((results.boxes.cls == 0).sum()) if len(results.boxes) > 0 else 0
    car_count   = int((results.boxes.cls == 1).sum()) if len(results.boxes) > 0 else 0
    annotated   = frame.copy()

    if len(results.boxes) > 0:
        for box, cls, conf_val in zip(results.boxes.xyxy.cpu().numpy(),
                                      results.boxes.cls.cpu().numpy(),
                                      results.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box)
            cls = int(cls)
            color = COLOR_HUMAN if cls == 0 else COLOR_CAR
            name  = CLASS_NAMES.get(cls, str(cls))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{name} {conf_val:.2f}"
            cv2.putText(annotated, label, (x1, max(y1-4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Count overlay
    overlay = annotated.copy()
    cv2.rectangle(overlay, (8, 8), (280, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
    cv2.putText(annotated, f"Humans: {human_count}", (18, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_HUMAN, 2)
    cv2.putText(annotated, f"Cars:   {car_count}", (18, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_CAR, 2)
    cv2.putText(annotated, "Standard YOLO", (18, 88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    return annotated, human_count, car_count


def generate_comparison_grid(img_paths, detection_model, save_path, n=4):
    """
    Generate side-by-side comparison: Standard YOLO vs SAHI.
    This is a great visualization for the submission report.
    """
    print(f"\nGenerating comparison grid ({n} images)...")
    from ultralytics import YOLO
    std_model = YOLO(WEIGHTS)

    fig, axes = plt.subplots(n, 2, figsize=(20, 6 * n))
    fig.suptitle(
        "Detection Comparison: Standard YOLO vs SAHI Enhanced\n"
        "(Green=Human, Red/Blue=Car)",
        fontsize=16, fontweight="bold"
    )

    axes[0][0].set_title("Standard YOLO (640px crops)", fontsize=13, fontweight="bold", color="red")
    axes[0][1].set_title("SAHI Enhanced (sliced inference)", fontsize=13, fontweight="bold", color="green")

    import random
    samples = random.sample(img_paths, min(n, len(img_paths)))

    for row, img_path in enumerate(samples):
        # Standard
        std_ann, std_h, std_c = run_standard_on_image(img_path)
        std_rgb = cv2.cvtColor(std_ann, cv2.COLOR_BGR2RGB)
        axes[row][0].imshow(std_rgb)
        axes[row][0].set_title(f"Standard: {std_h} humans, {std_c} cars", fontsize=10)
        axes[row][0].axis("off")

        # SAHI
        sahi_ann, sahi_h, sahi_c = run_sahi_on_image(detection_model, img_path)
        sahi_rgb = cv2.cvtColor(sahi_ann, cv2.COLOR_BGR2RGB)
        axes[row][1].imshow(sahi_rgb)
        axes[row][1].set_title(
            f"SAHI: {sahi_h} humans (+{sahi_h - std_h}), {sahi_c} cars (+{sahi_c - std_c})",
            fontsize=10,
            color="green" if sahi_h > std_h else "black"
        )
        axes[row][1].axis("off")

        print(f"  Image {row+1}: Standard={std_h}H/{std_c}C  SAHI={sahi_h}H/{sahi_c}C  "
              f"(+{sahi_h-std_h} humans, +{sahi_c-std_c} cars)")

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Comparison saved: {save_path}")


def run_sahi_on_folder(detection_model, source_dir: str, max_images=548):
    """Run SAHI on all images in a folder."""
    img_files = sorted(list(Path(source_dir).glob("*.jpg")) +
                       list(Path(source_dir).glob("*.png")))[:max_images]

    print(f"\nRunning SAHI on {len(img_files)} images...")
    print(f"Slice size: {SLICE_W}x{SLICE_H}  Overlap: {int(OVERLAP*100)}%")
    print(f"Confidence: {CONF_THRESH}  Postprocess: {POSTPROCESS}\n")

    total_h = total_c = 0

    for i, img_path in enumerate(img_files):
        annotated, h, c = run_sahi_on_image(detection_model, img_path)
        if annotated is None:
            continue
        total_h += h
        total_c += c

        # Save output
        cv2.imwrite(str(OUTPUT_DIR / img_path.name), annotated)

        if (i + 1) % 50 == 0 or i < 5:
            print(f"  [{i+1}/{len(img_files)}] {img_path.name} — humans:{h} cars:{c}")

    print(f"\nSAHI Detection Complete!")
    print(f"  Total humans detected: {total_h:,}")
    print(f"  Total cars detected:   {total_c:,}")
    print(f"  Saved to: {OUTPUT_DIR}")
    return img_files, total_h, total_c


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAHI Enhanced Detection")
    parser.add_argument("--source", default="F:/ANTS/VisDrone_Dataset/VisDrone2019-DET-val/images",
                        help="Image folder path OR video file path")
    parser.add_argument("--compare", action="store_true",
                        help="Generate side-by-side comparison with standard YOLO")
    parser.add_argument("--max", type=int, default=548,
                        help="Max images to process")
    args = parser.parse_args()

    print("=" * 60)
    print("SAHI Enhanced Detection — Drone Human & Car Detection")
    print("=" * 60)

    detection_model = load_sahi_model()

    source_path = Path(args.source)
    is_video = source_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']

    if is_video:
        # ── VIDEO MODE ─────────────────────────────────────────────────
        print(f"Video mode: {args.source}")
        out_video = Path("F:/ANTS/results/videos/sahi_detected.mp4")
        out_video.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(source_path))
        W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        FPS = cap.get(cv2.CAP_PROP_FPS) or 10
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = cv2.VideoWriter(
            str(out_video),
            cv2.VideoWriter_fourcc(*'mp4v'),
            FPS, (W, H)
        )

        frame_idx = 0
        print(f"Processing {total_frames} frames...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Save temp frame for SAHI
            temp_path = Path("F:/ANTS/temp_frame.jpg")
            cv2.imwrite(str(temp_path), frame)

            annotated, h, c = run_sahi_on_image(detection_model, temp_path)
            if annotated is not None:
                writer.write(annotated)

            frame_idx += 1
            if frame_idx % 10 == 0:
                print(f"  Frame {frame_idx}/{total_frames} — humans:{h} cars:{c}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        writer.release()
        temp_path.unlink(missing_ok=True)
        print(f"\nSAHI video saved: {out_video}")

    else:
        # ── IMAGE FOLDER MODE ──────────────────────────────────────────
        img_files, total_h, total_c = run_sahi_on_folder(
            detection_model, args.source, max_images=args.max
        )
        if args.compare:
            compare_path = COMPARE_DIR / "sahi_vs_standard_comparison.png"
            generate_comparison_grid(img_files, detection_model, compare_path, n=4)