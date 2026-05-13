"""
TASK 04 (BONUS) - OBJECT TRACKING WITH BoT-SORT
=================================================
Tracks humans and cars across video frames using BoT-SORT.
BoT-SORT includes Camera Motion Compensation (CMC) — ideal
for drone footage where the camera itself is moving.

Features:
  - Unique track ID per person/car
  - Persistent IDs across frames
  - Human count display
  - Track history trails (optional)

Usage:
    python scripts/track.py --source path/to/video.mp4

    # With trail visualization:
    python scripts/track.py --source path/to/video.mp4 --trails
"""

import os
import cv2
import argparse
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
PROJECT_ROOT = r"F:\antlings_project"
MODEL_PATH   = os.path.join(PROJECT_ROOT, "runs", "visdrone_yolov8s", "weights", "best.pt")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results", "videos")
os.makedirs(RESULTS_DIR, exist_ok=True)

CLASS_NAMES  = {0: "human", 1: "car"}
COLORS_BY_ID_BASE = [
    (0, 255, 80),   (255, 80, 0),   (0, 80, 255),
    (255, 255, 0),  (0, 255, 255),  (255, 0, 255),
    (128, 255, 0),  (255, 128, 0),  (0, 128, 255),
]
CONF_THRESHOLD = 0.25

# ─────────────────────────────────────────────

def get_color(track_id):
    """Returns a consistent color for a given track ID."""
    return COLORS_BY_ID_BASE[track_id % len(COLORS_BY_ID_BASE)]


def draw_tracked_frame(frame, tracks, track_history, human_count, frame_idx, fps):
    """
    Draw bounding boxes with track IDs, trails, and human count.
    """
    H, W = frame.shape[:2]

    for track in tracks:
        x1, y1, x2, y2 = map(int, track[:4])
        track_id = int(track[4])
        cls_id   = int(track[5]) if len(track) > 5 else 0
        conf     = float(track[6]) if len(track) > 6 else 0.0

        color = get_color(track_id)

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label
        cls_name = CLASS_NAMES.get(cls_id, "obj")
        label    = f"ID:{track_id} {cls_name}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Track center for trail
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        track_history[track_id].append((cx, cy))

        # Keep only last 30 points
        if len(track_history[track_id]) > 30:
            track_history[track_id].pop(0)

        # Draw trail
        points = track_history[track_id]
        for i in range(1, len(points)):
            alpha = i / len(points)
            thickness = max(1, int(alpha * 3))
            cv2.line(frame, points[i-1], points[i], color, thickness)

    # HUD — top-left overlay
    panel_h, panel_w = 90, 300
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, f"Humans Tracked: {human_count}",
                (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 80), 2)
    cv2.putText(frame, f"Frame: {frame_idx}",
                (18, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(frame, f"Tracker: BoT-SORT",
                (18, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 255), 1)

    return frame


def run_tracking(model_path, source_path, save_dir, show_trails=True):
    model = YOLO(model_path)

    source = Path(source_path)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        print(f"❌ Cannot open: {source_path}")
        return

    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS   = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = Path(save_dir) / f"track_{source.stem}_botsort.mp4"
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(str(out_path), fourcc, FPS, (W, H))

    print(f"\n  Video: {source.name}")
    print(f"  Resolution: {W}x{H} @ {FPS:.1f} FPS | Frames: {total}")
    print(f"  Output: {out_path}")
    print("  Tracking with BoT-SORT (Camera Motion Compensation enabled)...")

    track_history = defaultdict(list)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # BoT-SORT tracking — built into Ultralytics
        results = model.track(
            frame,
            persist=True,
            tracker="botsort.yaml",
            conf=CONF_THRESHOLD,
            iou=0.45,
            verbose=False,
        )

        tracks = []
        human_count = 0

        if results[0].boxes.id is not None:
            boxes   = results[0].boxes.xyxy.cpu().numpy()
            ids     = results[0].boxes.id.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            confs   = results[0].boxes.conf.cpu().numpy()

            for box, tid, cls, conf in zip(boxes, ids, classes, confs):
                tracks.append([*box, int(tid), int(cls), float(conf)])
                if int(cls) == 0:
                    human_count += 1

        annotated = draw_tracked_frame(
            frame.copy(), tracks, track_history, human_count, frame_idx, FPS
        )
        writer.write(annotated)
        frame_idx += 1

        if frame_idx % 50 == 0:
            print(f"  Frame {frame_idx}/{total} | Humans in frame: {human_count}")

    cap.release()
    writer.release()
    print(f"\n  ✅ Tracking complete!")
    print(f"  Output video: {out_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Task 04 - BoT-SORT Tracking")
    parser.add_argument("--source",   type=str, required=True,
                        help="Path to video file (.mp4, .avi)")
    parser.add_argument("--model",    type=str, default=MODEL_PATH)
    parser.add_argument("--trails",   action="store_true", default=True,
                        help="Show tracking trail lines")
    parser.add_argument("--save_dir", type=str, default=RESULTS_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 55)
    print("  TASK 04 — BoT-SORT Object Tracking (Bonus)")
    print("=" * 55)

    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        print("   Run train.py first!")
        exit(1)

    run_tracking(args.model, args.source, args.save_dir, args.trails)
