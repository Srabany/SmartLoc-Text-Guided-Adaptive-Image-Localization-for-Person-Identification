"""
Dataset Builder for SmartLoc
=============================
Semi-automated tool to create text-image-bbox triplets from group photos.

Usage:
    python build_dataset.py

Workflow:
    1. Scans all images in ../../data/images/
    2. Runs YOLOv8 person detection on each image
    3. Shows each detected person crop and asks you to type a description
    4. Saves annotations to ../../data/dataset.json

Output format (dataset.json):
    [
        {
            "image": "Group.JPG",
            "bbox": [x1, y1, x2, y2],
            "query": "man in black jacket with glasses",
            "person_index": 0
        },
        ...
    ]
"""

import os
import sys
import json
# pyrefly: ignore [missing-import]
import cv2
import numpy as np
import tkinter as tk
from tkinter import simpledialog
# pyrefly: ignore [missing-import]
from ultralytics import YOLO

# ─── CONFIG ────────────────────────────────────────────────────
IMAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "images"))
OUTPUT_JSON = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "dataset.json"))
YOLO_MODEL = "yolov8n.pt"
YOLO_CONF = 0.3
MIN_W, MIN_H = 50, 70  # minimum crop size to consider
# ───────────────────────────────────────────────────────────────

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_existing_dataset(path):
    """Load existing annotations if the file exists, so we can append."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_dataset(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved {len(data)} annotations to {path}")


def main():
    print("=" * 60)
    print("  SmartLoc Dataset Builder")
    print("=" * 60)

    # Initialize tkinter root for dialogs
    root = tk.Tk()
    root.withdraw()

    # Gather images
    image_files = sorted([
        f for f in os.listdir(IMAGE_DIR)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
    ])

    if not image_files:
        print(f"No images found in {IMAGE_DIR}")
        sys.exit(1)

    print(f"Found {len(image_files)} images in {IMAGE_DIR}\n")

    # Load YOLO
    print(f"Loading YOLOv8 model '{YOLO_MODEL}'...")
    yolo = YOLO(YOLO_MODEL)

    # Load existing annotations
    dataset = load_existing_dataset(OUTPUT_JSON)
    already_annotated = {(d["image"], tuple(d["bbox"])) for d in dataset}
    print(f"Existing annotations: {len(dataset)}\n")

    print("Instructions:")
    print("  - For each detected person crop, type a text description")
    print("  - Press ENTER with empty text to SKIP that person")
    print("  - Type 'quit' to stop and save progress")
    print("  - Type 'next' to skip to the next image")
    print("-" * 60)

    for img_idx, img_name in enumerate(image_files):
        img_path = os.path.join(IMAGE_DIR, img_name)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"  ⚠️ Could not read {img_name}, skipping.")
            continue

        H, W = img_bgr.shape[:2]
        print(f"\n📷 [{img_idx+1}/{len(image_files)}] {img_name} ({W}x{H})")

        # Detect persons
        results = yolo(img_path, conf=YOLO_CONF, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        # Filter: only persons (class 0) with minimum size
        person_boxes = []
        for box, conf, cls in zip(boxes, confs, classes):
            if int(cls) != 0:
                continue
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            if w < MIN_W or h < MIN_H:
                continue
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(W, x2); y2 = min(H, y2)
            person_boxes.append([x1, y1, x2, y2, float(conf)])

        if not person_boxes:
            print(f"  No valid persons detected, skipping.")
            continue

        print(f"  Detected {len(person_boxes)} persons")

        # Show the full image with numbered bounding boxes
        overview = img_bgr.copy()
        for p_idx, (x1, y1, x2, y2, conf) in enumerate(person_boxes):
            cv2.rectangle(overview, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"#{p_idx} ({conf:.2f})"
            cv2.putText(overview, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Scale overview for display
        scale = min(1200 / W, 800 / H, 1.0)
        if scale < 1.0:
            overview = cv2.resize(overview, (int(W * scale), int(H * scale)))
        cv2.imshow("Overview - All Detections", overview)
        cv2.waitKey(500)  # brief pause to render

        skip_image = False

        for p_idx, (x1, y1, x2, y2, conf) in enumerate(person_boxes):
            # Check if already annotated
            if (img_name, (x1, y1, x2, y2)) in already_annotated:
                print(f"    Person #{p_idx}: already annotated, skipping.")
                continue

            # Show the individual crop
            crop = img_bgr[y1:y2, x1:x2]
            crop_display = crop.copy()
            # Scale crop for visibility
            crop_h, crop_w = crop_display.shape[:2]
            crop_scale = max(200 / crop_w, 300 / crop_h)
            if crop_scale > 1.0:
                crop_display = cv2.resize(crop_display,
                                          (int(crop_w * crop_scale), int(crop_h * crop_scale)),
                                          interpolation=cv2.INTER_LINEAR)
            cv2.imshow(f"Person #{p_idx}", crop_display)
            cv2.waitKey(100)

            # Ask for description using a GUI popup to prevent OpenCV from freezing
            prompt_text = f"Describe Person #{p_idx} (conf={conf:.2f}):\n\n- Leave empty to skip\n- Type 'quit' to stop\n- Type 'next' to skip image"
            query = simpledialog.askstring("SmartLoc Annotator", prompt_text, parent=root)
            
            if query is None:
                query = ""
            query = query.strip()

            cv2.destroyWindow(f"Person #{p_idx}")

            if query.lower() == "quit":
                print("\n⏹ Stopping. Saving progress...")
                save_dataset(dataset, OUTPUT_JSON)
                cv2.destroyAllWindows()
                sys.exit(0)

            if query.lower() == "next":
                skip_image = True
                break

            if not query:
                print(f"      ↳ Skipped")
                continue

            # Save annotation
            annotation = {
                "image": img_name,
                "bbox": [x1, y1, x2, y2],
                "query": query,
                "person_index": p_idx,
                "confidence": round(conf, 3)
            }
            dataset.append(annotation)
            already_annotated.add((img_name, (x1, y1, x2, y2)))
            print(f"      ✅ Saved: \"{query}\"")

        cv2.destroyWindow("Overview - All Detections")

        if skip_image:
            continue

    # Final save
    save_dataset(dataset, OUTPUT_JSON)
    cv2.destroyAllWindows()
    print("\n🎉 Dataset building complete!")


if __name__ == "__main__":
    main()
