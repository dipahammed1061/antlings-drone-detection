"""
track.py — BoT-SORT Multi-Class Object Tracking (Bonus Task)
=============================================================
Uses yolov8x.pt (COCO pretrained, 80 classes) — no custom training needed.
Detects: person, bicycle, car, motorcycle, bus, truck from drone footage.

Usage:
    python scripts/track.py --source F:/ANTS/test_video.mp4
    python scripts/track.py --source F:/ANTS/test_video.mp4 --trails
"""

import os
import cv2
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict, deque
from ultralytics import YOLO

# ─── CONFIG ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = r"F:\ANTS"
MODEL_PATH   = "yolov8x.pt"          # auto-downloads ~130MB on first run
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results", "videos")
os.makedirs(RESULTS_DIR, exist_ok=True)

# COCO class IDs we care about → (label, BGR color)
# These are the EXACT COCO IDs — must match yolov8x.pt
COCO_CLASSES = {
    0:  ("person",     (0,   255,  0  )),   # green
    1:  ("bicycle",    (255, 165,  0  )),   # orange
    2:  ("car",        (0,   0,    255)),   # red
    3:  ("motorcycle", (255, 0,    255)),   # magenta
    5:  ("bus",        (0,   255,  255)),   # cyan
    7:  ("truck",      (255, 255,  0  )),   # yellow
}
TRACKED_IDS  = list(COCO_CLASSES.keys())   # [0,1,2,3,5,7]
CONF         = 0.25
TRAIL_LEN    = 30

# ─── DRAWING HELPERS ─────────────────────────────────────────────────────────

def draw_box(frame, x1, y1, x2, y2, label, color, conf):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_trail(frame, points, color):
    pts = list(points)
    for i in range(1, len(pts)):
        alpha = i / len(pts)
        c = tuple(int(ch * alpha) for ch in color)
        cv2.line(frame, pts[i - 1], pts[i], c, 2)


def draw_overlay(frame, counts, frame_num, total_frames):
    """Top-left semi-transparent stats panel."""
    total = sum(counts.values())
    lines = [
        f"Frame: {frame_num}/{total_frames}",
        f"Tracker: BoT-SORT",
        f"Objects Tracked: {total}",
        "---",
    ]
    for name, cnt in sorted(counts.items()):
        lines.append(f"  {name}: {cnt}")

    pad = 10
    lh  = 22
    w   = 220
    h   = pad * 2 + lh * len(lines)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    y = pad + lh - 4
    for line in lines:
        color = (0, 255, 255) if "Objects" in line else (200, 200, 200)
        if "Frame" in line or "Tracker" in line:
            color = (0, 255, 0)
        cv2.putText(frame, line, (pad, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        y += lh


# ─── MAIN TRACKING ───────────────────────────────────────────────────────────

def run(source: str, show_trails: bool):
    print("=" * 55)
    print("  TASK 04 — BoT-SORT Object Tracking (Bonus)")
    print("=" * 55)
    print(f"[INFO] Loading {MODEL_PATH} (auto-downloads if needed)...")

    model = YOLO(MODEL_PATH)

    # Verify COCO classes are present
    print("[INFO] Confirmed classes in model:")
    for cid, (cname, _) in COCO_CLASSES.items():
        actual = model.names.get(cid, "MISSING")
        print(f"       class {cid} = {actual}  ({'✓' if actual == cname else '✗ MISMATCH'})")

    # Open source video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {source}")

    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS    = cap.get(cv2.CAP_PROP_FPS) or 10.0
    NFRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    out_name = Path(source).stem + "_botsort.mp4"
    out_path = os.path.join(RESULTS_DIR, out_name)
    writer   = cv2.VideoWriter(out_path,
                               cv2.VideoWriter_fourcc(*"mp4v"),
                               FPS, (W, H))

    print(f"\n  Video: {Path(source).name}")
    print(f"  Resolution: {W}x{H} @ {FPS:.1f} FPS  |  Frames: {NFRAMES}")
    print(f"  Output: {out_path}")
    print(f"  Tracking with BoT-SORT (Camera Motion Compensation enabled)...\n")

    trails: dict = defaultdict(lambda: deque(maxlen=TRAIL_LEN))
    frame_num = 0

    # KEY: classes= tells YOLO to only detect the classes we care about
    results = model.track(
        source   = source,
        stream   = True,
        tracker  = "botsort.yaml",
        conf     = CONF,
        iou      = 0.45,
        classes  = TRACKED_IDS,   # ← only detect person/bike/car/moto/bus/truck
        imgsz    = 1280,           # ← higher resolution catches small objects
        persist  = True,           # ← keep IDs across frames
        verbose  = False,
    )

    for result in results:
        frame     = result.orig_img.copy()
        frame_num += 1
        counts: dict = defaultdict(int)

        if result.boxes is not None and len(result.boxes) > 0:
            has_ids = result.boxes.id is not None

            for i, box in enumerate(result.boxes):
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                name, color = COCO_CLASSES.get(cls_id, ("object", (128, 128, 128)))
                counts[name] += 1

                # Track ID (if available)
                tid = int(box.id[0]) if has_ids else i
                label = f"ID:{tid} {name}"

                # Trail
                if show_trails:
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    trails[tid].append((cx, cy))
                    draw_trail(frame, trails[tid], color)

                draw_box(frame, x1, y1, x2, y2, label, color, conf)

        draw_overlay(frame, counts, frame_num, NFRAMES)
        writer.write(frame)

        if frame_num % 5 == 0:
            total = sum(counts.values())
            print(f"  Frame {frame_num:3d}/{NFRAMES}  |  objects={total}  "
                  + "  ".join(f"{k}:{v}" for k, v in sorted(counts.items())))

    writer.release()
    print(f"\n✅ Tracking complete!")
    print(f"   Output video: {out_path}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Path to input video")
    ap.add_argument("--trails", action="store_true", help="Draw motion trails")
    args = ap.parse_args()
    run(args.source, args.trails)