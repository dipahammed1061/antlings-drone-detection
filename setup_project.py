"""
STEP 0 — Run this ONCE to create the full project structure.

Anaconda Prompt:
    conda activate visdrone
    cd F:/ANTS
    python setup_project.py
"""

import os

BASE = "F:/ANTS"

folders = [
    "scripts",
    "notebooks",
    "results/images",
    "results/videos",
    "results/metrics",
    "outputs/preprocessed/train/images",
    "outputs/preprocessed/train/labels",
    "outputs/preprocessed/val/images",
    "outputs/preprocessed/val/labels",
    "outputs/preprocessed/test/images",
    "outputs/preprocessed/test/labels",
]

for f in folders:
    path = os.path.join(BASE, f)
    os.makedirs(path, exist_ok=True)
    print(f"  Created: {path}")

print("\nProject structure ready at F:/ANTS/")
