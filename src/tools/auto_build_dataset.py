import os
import sys
import json
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
from PIL import Image
# pyrefly: ignore [missing-import]
from ultralytics import YOLO
# pyrefly: ignore [missing-import]
from transformers import BlipProcessor, BlipForConditionalGeneration
import warnings

# Suppress annoying huggingface/ultralytics warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# ─── CONFIG ────────────────────────────────────────────────────
IMAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "images"))
OUTPUT_JSON = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "dataset.json"))
YOLO_MODEL = "yolov8n.pt"
BLIP_MODEL = "Salesforce/blip-image-captioning-base"
YOLO_CONF = 0.3
MIN_W, MIN_H = 50, 70  # minimum crop size to consider
# ───────────────────────────────────────────────────────────────

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif"}

def load_existing_dataset(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_dataset(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved {len(data)} annotations to {path}")

def generate_caption(processor, model, device, image_crop):
    """Generates a caption for the given PIL image crop using BLIP."""
    inputs = processor(image_crop, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=20)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def main():
    print("=" * 60)
    print("  SmartLoc Automated Dataset Builder (AI Captioning)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Gather images
    image_files = sorted([
        f for f in os.listdir(IMAGE_DIR)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
    ])

    if not image_files:
        print(f"No images found in {IMAGE_DIR}")
        sys.exit(1)

    print(f"Found {len(image_files)} images to process.\n")

    # Load YOLO
    print(f"Loading YOLOv8 model '{YOLO_MODEL}'...")
    yolo = YOLO(YOLO_MODEL)

    # Load BLIP
    print(f"Loading BLIP Captioning model '{BLIP_MODEL}' (this may take a moment to download)...")
    processor = BlipProcessor.from_pretrained(BLIP_MODEL)
    blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL).to(device)

    # Load existing annotations to avoid duplicates
    dataset = load_existing_dataset(OUTPUT_JSON)
    already_annotated = {(d["image"], tuple(d["bbox"])) for d in dataset}
    print(f"Existing annotations: {len(dataset)}\n")

    print("-" * 60)
    print("Starting automated annotation...")

    total_added = 0

    for img_idx, img_name in enumerate(image_files):
        img_path = os.path.join(IMAGE_DIR, img_name)
        try:
            img_pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  ⚠️ Could not read {img_name}, skipping.")
            continue

        img_w, img_h = img_pil.size
        print(f"\n📷 [{img_idx+1}/{len(image_files)}] {img_name}")

        # Detect persons
        results = yolo(img_path, conf=YOLO_CONF, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        person_boxes = []
        for box, conf, cls in zip(boxes, confs, classes):
            if int(cls) != 0:
                continue
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            if w < MIN_W or h < MIN_H:
                continue
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(img_w, x2); y2 = min(img_h, y2)
            person_boxes.append([x1, y1, x2, y2, float(conf)])

        if not person_boxes:
            print("  ↳ No valid persons detected.")
            continue

        added_this_image = 0
        for p_idx, (x1, y1, x2, y2, conf) in enumerate(person_boxes):
            bbox_tuple = (x1, y1, x2, y2)
            if (img_name, bbox_tuple) in already_annotated:
                continue

            # Crop person
            crop = img_pil.crop(bbox_tuple)
            
            # Generate Caption automatically!
            caption = generate_caption(processor, blip_model, device, crop)

            # Save annotation
            annotation = {
                "image": img_name,
                "bbox": [x1, y1, x2, y2],
                "query": caption,
                "person_index": p_idx,
                "confidence": round(conf, 3)
            }
            dataset.append(annotation)
            already_annotated.add((img_name, bbox_tuple))
            total_added += 1
            added_this_image += 1
            
            print(f"    ↳ Person #{p_idx}: \"{caption}\"")

        # Save progress periodically (every 10 images)
        if img_idx % 10 == 0 and total_added > 0:
            save_dataset(dataset, OUTPUT_JSON)

    # Final save
    save_dataset(dataset, OUTPUT_JSON)
    print(f"\n🎉 Automated dataset building complete! Added {total_added} new annotations.")

if __name__ == "__main__":
    main()
