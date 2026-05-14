"""
TASK 05 - EVALUATION & VISUALIZATION
======================================
Computes and displays:
  - mAP50, mAP50-95
  - Precision, Recall
  - FPS on test set
  - Confusion matrix
  - PR curve
  - Sample prediction grid

Run AFTER training:
    python scripts/evaluate.py
"""

import os
import cv2
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
PROJECT_ROOT = r"F:\ANTS"
MODEL_PATH = os.path.join(PROJECT_ROOT, "results", "yolov8s_visdrone_v1", "weights", "best.pt")
YAML_PATH = os.path.join(PROJECT_ROOT, "visdrone_remapped.yaml")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
METRICS_DIR  = os.path.join(RESULTS_DIR, "metrics")
IMAGES_DIR   = os.path.join(RESULTS_DIR, "images")
os.makedirs(METRICS_DIR, exist_ok=True)

CLASS_NAMES = {0: "human", 1: "car"}
CONF        = 0.25
random.seed(42)

# ─────────────────────────────────────────────

def run_validation(model):
    """Run official YOLO validation — returns metrics."""
    print("\n[1/4] Running official validation...")
    metrics = model.val(
        data=YAML_PATH,
        conf=CONF,
        iou=0.45,
        imgsz=1280,
        verbose=True,
        plots=True,
        save_dir=METRICS_DIR,
    )
    return metrics


def print_metrics_table(metrics):
    """Print a clean metrics summary table."""
    print("\n" + "=" * 50)
    print("  EVALUATION RESULTS")
    print("=" * 50)
    print(f"  {'Metric':<25} {'Value':>10}")
    print("  " + "-" * 36)
    print(f"  {'mAP@50':<25} {metrics.box.map50:>10.4f}")
    print(f"  {'mAP@50-95':<25} {metrics.box.map:>10.4f}")
    print(f"  {'Precision (mean)':<25} {metrics.box.mp:>10.4f}")
    print(f"  {'Recall (mean)':<25} {metrics.box.mr:>10.4f}")
    print("  " + "-" * 36)

    # Per-class
    for i, cls_name in CLASS_NAMES.items():
        try:
            ap50 = metrics.box.ap50[i]
            print(f"  AP@50 [{cls_name}]{'':<14} {ap50:>10.4f}")
        except (IndexError, AttributeError):
            pass
    print("=" * 50)

    # Note about VisDrone scores
    print("\n  📝 Note: VisDrone is a challenging dataset.")
    print("     mAP@50 of 25-45% is considered good for aerial detection.")
    print("     Tiny object sizes and dense scenes lower scores vs COCO.")


def benchmark_fps(model, img_dir, n=50):
    img_files = list(Path(img_dir).glob("*.jpg"))
    if len(img_files) == 0:
        print("  No images found for FPS benchmark")
        return 0
    img_files = img_files[:min(n, len(img_files))]
    times = []
    for img_path in img_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        start = time.time()
        model(img, verbose=False)
        times.append(time.time() - start)
    if len(times) == 0:
        return 0
    avg_ms = np.mean(times) * 1000
    fps = 1.0 / np.mean(times)
    print(f"  Avg inference time : {avg_ms:.1f} ms/image")
    print(f"  Avg FPS            : {fps:.1f}")
    return fps


def generate_prediction_grid(model, test_img_dir, save_path, n=6):
    """Generate a grid of sample prediction images."""
    print("\n[3/4] Generating prediction sample grid...")
    img_dir = Path(test_img_dir)
    all_imgs = list(img_dir.glob("*.jpg"))
    samples = random.sample(all_imgs, min(n, len(all_imgs)))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Task 03 — Sample Detection Results\n(Green=Human, Orange=Car)",
                 fontsize=15, fontweight="bold")

    CLASS_COLORS = {0: (0, 255, 80), 1: (0, 100, 255)}

    for idx, img_path in enumerate(samples):
        frame = cv2.imread(str(img_path))
        results = model(frame, conf=CONF, verbose=False)[0]

        boxes   = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        confs   = results.boxes.conf.cpu().numpy()

        human_count = int((classes == 0).sum())
        car_count   = int((classes == 1).sum())

        H, W = frame.shape[:2]
        for box, cls, conf in zip(boxes, classes, confs):
            x1, y1, x2, y2 = map(int, box)
            cls = int(cls)
            color = CLASS_COLORS.get(cls, (255, 255, 0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Count overlay
        cv2.rectangle(frame, (5, 5), (240, 55), (20, 20, 20), -1)
        cv2.putText(frame, f"Humans: {human_count}  Cars: {car_count}",
                    (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 80), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ax = axes[idx // 3][idx % 3]
        ax.imshow(frame_rgb)
        ax.set_title(f"Humans: {human_count} | Cars: {car_count}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def generate_metrics_summary_chart(metrics, fps, save_path):
    """Generate a clean metrics bar chart for the report."""
    print("\n[4/4] Generating metrics summary chart...")

    labels = ["mAP@50", "mAP@50-95", "Precision", "Recall"]
    values = [
        metrics.box.map50,
        metrics.box.map,
        metrics.box.mp,
        metrics.box.mr,
    ]
    colors = ["#3498db", "#2980b9", "#2ecc71", "#e67e22"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Task 05 — Evaluation Metrics Summary", fontsize=14, fontweight="bold")

    # Bar chart
    bars = axes[0].bar(labels, values, color=colors, edgecolor="white", linewidth=1.5)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Detection Metrics")
    for bar, val in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")
    axes[0].axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="0.5 reference")
    axes[0].legend()

    # FPS gauge
    axes[1].text(0.5, 0.55, f"{fps:.1f}", ha="center", va="center",
                 fontsize=60, fontweight="bold", color="#3498db",
                 transform=axes[1].transAxes)
    axes[1].text(0.5, 0.3, "FPS", ha="center", va="center",
                 fontsize=20, color="#7f8c8d", transform=axes[1].transAxes)
    axes[1].text(0.5, 0.15, "(Inference Speed on RTX 3060)",
                 ha="center", va="center", fontsize=10, color="#95a5a6",
                 transform=axes[1].transAxes)
    axes[1].set_title("Inference Speed")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  TASK 05 — Evaluation & Visualization")
    print("=" * 55)

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        print("   Run train.py first!")
        exit(1)

    model = YOLO(MODEL_PATH)

    # 1. Official validation metrics
    metrics = run_validation(model)
    print_metrics_table(metrics)

    # 2. FPS benchmark
    test_img_dir = "F:/ANTS/VisDrone_Dataset/VisDrone2019-DET-val/images"
    fps = benchmark_fps(model, test_img_dir)

    # 3. Prediction grid
    grid_path = os.path.join(IMAGES_DIR, "06_prediction_samples.png")
    generate_prediction_grid(model, test_img_dir, grid_path)

    # 4. Metrics chart
    chart_path = os.path.join(METRICS_DIR, "07_metrics_summary.png")
    generate_metrics_summary_chart(metrics, fps, chart_path)

    print("\n" + "=" * 55)
    print("  ALL EVALUATION COMPLETE")
    print("=" * 55)
    print(f"\n  Results saved to: {RESULTS_DIR}")
    print("\n  ✅ All 5 tasks complete! Now:")
    print("     1. Push to GitHub")
    print("     2. Record demo video")
    print("     3. Submit!")
