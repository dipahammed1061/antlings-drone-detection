# 🚁 Drone Human & Car Detection System
### Antlings Internship — Phase 2 Technical Assessment (AI/ML)

A computer vision pipeline for detecting humans and cars in drone/aerial imagery using YOLOv8s fine-tuned on the VisDrone2019 dataset.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Setup](#setup)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Results](#results)
- [Limitations & Discussion](#limitations)

---

## Overview

This project implements a full object detection pipeline for aerial drone imagery:

| Task | Description |
|------|-------------|
| Task 01 | Dataset understanding, preprocessing, class remapping, EDA visualizations |
| Task 02 | YOLOv8s fine-tuning with drone-specific augmentation at imgsz=1280 |
| Task 03 | Human & car detection with real-time human count overlay |
| Task 04 | BoT-SORT object tracking with Camera Motion Compensation (bonus) |
| Task 05 | Evaluation metrics (mAP, Precision, Recall, FPS) and visualization |

**Key design decisions:**
- **YOLOv8s** chosen for best small-object accuracy/speed tradeoff on aerial data
- **imgsz=1280** instead of standard 640 — dramatically improves tiny pedestrian detection
- **Class remapping**: VisDrone's 10 classes merged to 2 (`pedestrian`+`people`→`human`, `car`+`van`→`car`)
- **SAHI** (Slicing Aided Hyper Inference) available for highest accuracy on high-res images
- **BoT-SORT** tracker used over ByteTrack — includes Camera Motion Compensation (CMC) essential for moving drone footage

---

## Dataset

**VisDrone2019-DET** — [Kaggle Link](https://www.kaggle.com/datasets/banuprasadb/visdrone-dataset)

| Split | Images | Notes |
|-------|--------|-------|
| Train | 6,471  | Images + YOLO labels |
| Val   | 548    | Images + YOLO labels |
| Test  | 1,610  | Images + YOLO labels |

**Original classes (10):** pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor

**Remapped classes (2):**
- `0: human` ← pedestrian + people
- `1: car`   ← car + van
- (all others dropped)

**Key challenges identified:**
1. Very small object sizes — humans often occupy fewer than 20×20 pixels
2. Dense crowds — high overlap between bounding boxes
3. Varying altitude and lighting conditions
4. Camera motion in video sequences (addressed by BoT-SORT CMC)
5. Class imbalance — significantly more human annotations than car

---

## Setup

### Requirements
- Python 3.10
- CUDA-enabled GPU (tested on RTX 3060 12GB)
- Anaconda

### Installation

```bash
# Create and activate environment
conda create -n visdrone python=3.10 -y
conda activate visdrone

# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install ultralytics supervision opencv-python matplotlib pandas seaborn tqdm sahi
```

### Verify GPU
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True  NVIDIA GeForce RTX 3060
```

---

## Project Structure

```
antlings_project/
├── data/
│   └── processed/
│       ├── train/images/ & labels/   # Remapped 2-class labels
│       ├── val/images/ & labels/
│       └── test/images/ & labels/
├── scripts/
│   ├── preprocess.py     # Task 01 — class remapping
│   ├── eda.py            # Task 01 — EDA visualizations
│   ├── train.py          # Task 02 — YOLOv8s training
│   ├── detect.py         # Task 03 — detection + counting
│   ├── track.py          # Task 04 — BoT-SORT tracking (bonus)
│   └── evaluate.py       # Task 05 — metrics + visualization
├── runs/
│   └── visdrone_yolov8s/
│       └── weights/
│           ├── best.pt   # Best checkpoint
│           └── last.pt
├── results/
│   ├── images/           # Detection output images
│   ├── videos/           # Tracking output videos
│   └── metrics/          # mAP, PR curves, confusion matrix
├── visdrone_remapped.yaml
├── setup_project.py
└── README.md
```

---

## Usage

### Step 1 — Setup project folders
```bash
python setup_project.py
```

### Step 2 — Preprocess dataset (class remapping)
```bash
python scripts/preprocess.py
```

### Step 3 — EDA visualizations (Task 01)
```bash
python scripts/eda.py
```

### Step 4 — Train model (Task 02)
```bash
python scripts/train.py
# ~2-4 hours on RTX 3060
```

### Step 5 — Detection + counting (Task 03)
```bash
# Standard inference
python scripts/detect.py --source F:/VisDrone_Dataset/VisDrone2019-DET-test-dev/images

# SAHI inference (better for tiny objects, slower)
python scripts/detect.py --source path/to/image.jpg --sahi
```

### Step 6 — Tracking (Task 04 Bonus)
```bash
python scripts/track.py --source path/to/video.mp4
```

### Step 7 — Evaluation (Task 05)
```bash
python scripts/evaluate.py
```

---

## Results

| Metric | Value |
|--------|-------|
| mAP@50 | *(run evaluate.py)* |
| mAP@50-95 | *(run evaluate.py)* |
| Precision | *(run evaluate.py)* |
| Recall | *(run evaluate.py)* |
| FPS (RTX 3060) | *(run evaluate.py)* |

> VisDrone benchmark context: mAP@50 of 25–45% is considered competitive due to tiny object sizes and dense scenes.

---

## Limitations & Discussion

**Strengths:**
- Higher input resolution (1280) specifically addresses VisDrone's tiny object challenge
- Class remapping reduces confusion between semantically similar VisDrone classes
- BoT-SORT camera motion compensation improves tracking stability in drone video
- SAHI inference available for maximum accuracy when speed is not critical

**Limitations:**
- Very small objects (<10px) remain difficult even at imgsz=1280
- Counting logic is per-frame (not unique person count) — tracking needed for accurate unique counting
- Model trained only on VisDrone — may not generalize to significantly different drone altitudes or cameras
- Dense crowd scenes cause bounding box overlap and merged detections

**Future improvements:**
- DINO/RT-DETR architectures may improve small object detection further
- Multi-scale training (varying imgsz per epoch)
- Crowd-counting head as an auxiliary task

---

## Demo Video

[Google Drive Link — to be added]

---

*Built for Antlings Internship Program — Phase 2 Technical Assessment*
