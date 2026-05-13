"""
TASK 3 — HUMAN & CAR DETECTION WITH COUNTING
=============================================
Usage:
    python scripts/detect.py --source F:/VisDrone_Dataset/VisDrone2019-DET-val/images
    python scripts/detect.py --source path/to/video.mp4
"""

import cv2
import argparse
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import supervision as sv

WEIGHTS     = "F:/ANTS/results/yolov8s_visdrone_v1/weights/best.pt"
CONF_THRESH = 0.25
IOU_THRESH  = 0.45
OUTPUT_DIR  = Path("F:/ANTS/results/images/detections")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {0: "human", 1: "car"}


def process_frame(model, frame):
    results     = model(frame, conf=CONF_THRESH, iou=IOU_THRESH, verbose=False)[0]
    detections  = sv.Detections.from_ultralytics(results)
    human_count = int((detections.class_id == 0).sum()) if len(detections) > 0 else 0
    car_count   = int((detections.class_id == 1).sum()) if len(detections) > 0 else 0

    labels = []
    if len(detections) > 0:
        for cls_id, conf in zip(detections.class_id, detections.confidence):
            labels.append(f"{CLASS_NAMES.get(int(cls_id), str(cls_id))} {conf:.2f}")

    box_annotator   = sv.BoxAnnotator(color=sv.ColorPalette.from_hex(['#00DC50', '#DC3232']), thickness=2)
    label_annotator = sv.LabelAnnotator(color=sv.ColorPalette.from_hex(['#00DC50', '#DC3232']), text_scale=0.4)

    annotated = box_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

    # Semi-transparent count panel
    overlay = annotated.copy()
    cv2.rectangle(overlay, (8, 8), (260, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
    cv2.putText(annotated, f"Humans: {human_count}", (18, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 80), 2)
    cv2.putText(annotated, f"Cars:   {car_count}", (18, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 50, 50), 2)

    return annotated, human_count, car_count


def run_detection(source, save_output=True):
    model = YOLO(WEIGHTS)
    source_path = Path(source)
    is_video  = source_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']
    is_folder = source_path.is_dir()

    if is_folder:
        img_files = sorted(list(source_path.glob("*.jpg")) + list(source_path.glob("*.png")))
        print(f"Found {len(img_files)} images")
        total_h = total_c = 0

        for i, img_path in enumerate(img_files):
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            annotated, h, c = process_frame(model, frame)
            total_h += h; total_c += c

            if save_output:
                cv2.imwrite(str(OUTPUT_DIR / img_path.name), annotated)
            if i < 5:
                cv2.imshow("Detection", annotated)
                print(f"  [{i+1}] {img_path.name} — humans:{h} cars:{c}")
                cv2.waitKey(1500)

        cv2.destroyAllWindows()
        print(f"\nTotal — humans: {total_h}  cars: {total_c}")
        if save_output:
            print(f"Saved to: {OUTPUT_DIR}")

    elif is_video:
        cap = cv2.VideoCapture(str(source_path))
        W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        FPS = cap.get(cv2.CAP_PROP_FPS) or 25
        out_video = OUTPUT_DIR / (source_path.stem + "_detected.mp4")
        writer    = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            annotated, h, c = process_frame(model, frame)
            writer.write(annotated)
            cv2.imshow("Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release(); writer.release(); cv2.destroyAllWindows()
        print(f"Saved: {out_video}")

    else:
        # Single image
        frame = cv2.imread(str(source_path))
        annotated, h, c = process_frame(model, frame)
        print(f"Humans: {h}  Cars: {c}")
        cv2.imshow("Detection", annotated)
        if save_output:
            cv2.imwrite(str(OUTPUT_DIR / source_path.name), annotated)
        cv2.waitKey(0); cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="F:/VisDrone_Dataset/VisDrone2019-DET-val/images")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    run_detection(args.source, save_output=not args.no_save)
