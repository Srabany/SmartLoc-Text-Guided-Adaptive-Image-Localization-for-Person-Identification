# FILE: similarity_vs_query_two_images.py
"""
Plot CLIP similarity (text vs ground-truth person crop) for TWO images.
- Chooses ground-truth automatically as the largest detected person (by area).
- Plots one line per image (queries can be different).
- Saves figure to ../../graphs/similarity_vs_query_two_images.png
- Saves CSV of results to ../../graphs/similarity_results_two_images.csv
"""

import os
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
from ultralytics import YOLO
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import clip

# ---------------- CONFIG ----------------
IMAGE_FILES = [
    "../../data/images/Image01.JPG",
    "../../data/images/Image02.jpg"
]

QUERIES_IMG1 = [
    "man",
    "man wearing jacket",
    "man wearing black jacket",
    "man in black jacket walking forward",
    "man with black jacket, jeans, hands in pockets"
]

QUERIES_IMG2 = [
    "man",
    "man wearing red",
    "man in red shirt",
    "smiling man in red shirt",
    "young smiling man in red t-shirt outdoors"
]

YOLO_CONF = 0.3
MIN_W, MIN_H = 60, 80

OUT_GRAPHS = "../../graphs"
os.makedirs(OUT_GRAPHS, exist_ok=True)

FIG_PATH = os.path.join(OUT_GRAPHS, "similarity_vs_query_two_images.png")
CSV_PATH = os.path.join(OUT_GRAPHS, "similarity_results_two_images.csv")

# ---------------- LOAD MODELS ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
clip_model, preprocess = clip.load("ViT-B/32", device=device)
yolo = YOLO("yolov8n.pt")

# ---------------- HELPERS ----------------
def detect_person_crops(img_path):
    """Return list of (crop_pil, area, conf) for person detections after simple size filter."""
    res = yolo(img_path, conf=YOLO_CONF)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return []
    boxes = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    classes = res.boxes.cls.cpu().numpy()
    img_bgr = cv2.imread(img_path)
    H, W = img_bgr.shape[:2]

    crops = []
    for (box, conf, cls) in zip(boxes, confs, classes):
        if int(cls) != 0:  # only persons
            continue
        x1, y1, x2, y2 = map(int, box)
        w = x2 - x1; h = y2 - y1
        if w < MIN_W or h < MIN_H:
            continue
        x1 = max(0, x1); y1 = max(0, y1); x2 = min(W, x2); y2 = min(H, y2)
        crop = Image.fromarray(cv2.cvtColor(img_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
        area = w * h
        crops.append((crop, area, float(conf)))
    return crops

def encode_image_feat(pil_img):
    t = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        f = clip_model.encode_image(t)
        f = f / f.norm(dim=-1, keepdim=True)
    return f

def encode_text_feat(text):
    tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        tf = clip_model.encode_text(tokens)
        tf = tf / tf.norm(dim=-1, keepdim=True)
    return tf

# ---------------- PROCESS IMAGES ----------------
results_table = []  # to save CSV rows
similarities_img1 = []
similarities_img2 = []
yolo_times = []
clip_text_times = []

for idx, img_path in enumerate(IMAGE_FILES):
    label = os.path.basename(img_path)
    print(f"\nProcessing {label} ...")
    t0 = time.time()
    crops = detect_person_crops(img_path)
    yolo_elapsed = time.time() - t0
    yolo_times.append(yolo_elapsed)

    if len(crops) == 0:
        print("  No valid person crops found after filtering.")
        if idx == 0:
            similarities_img1 = [0.0]*len(QUERIES_IMG1)
        else:
            similarities_img2 = [0.0]*len(QUERIES_IMG2)
        continue

    # choose ground-truth crop = largest area
    crops_sorted = sorted(enumerate(crops), key=lambda x: x[1][1], reverse=True)
    gt_index = crops_sorted[0][0]
    gt_crop = crops[gt_index][0]
    gt_conf = crops[gt_index][2]
    gt_feat = encode_image_feat(gt_crop)

    print(f"  Found {len(crops)} person crops, chosen GT index = {gt_index}, conf={gt_conf:.2f}, yolo_time={yolo_elapsed:.3f}s")

    # choose query list
    queries = QUERIES_IMG1 if idx == 0 else QUERIES_IMG2
    sim_list = []

    for q in queries:
        t1 = time.time()
        q_feat = encode_text_feat(q)
        clip_text_times.append(time.time() - t1)
        sim = float((q_feat @ gt_feat.T).item())
        sim_list.append(sim)

        # store row
        results_table.append({
            "image": label,
            "gt_index": gt_index,
            "query": q,
            "similarity": sim,
            "yolo_time": yolo_elapsed,
            "gt_conf": gt_conf
        })

    if idx == 0:
        similarities_img1 = sim_list
    else:
        similarities_img2 = sim_list

# ---------------- SAVE CSV ----------------
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["image","gt_index","query","similarity","yolo_time","gt_conf"])
    writer.writeheader()
    for r in results_table:
        writer.writerow(r)
print(f"\nSaved results CSV to: {CSV_PATH}")

# ---------------- METRICS SUMMARY ----------------
sim_all = [r["similarity"] for r in results_table] if results_table else [0.0]
yolo_avg = float(np.mean(yolo_times)) if yolo_times else 0.0
clip_text_avg = float(np.mean(clip_text_times)) if clip_text_times else 0.0
sim_min, sim_max = (float(np.min(sim_all)), float(np.max(sim_all))) if sim_all else (0.0,0.0)

print(f"YOLO avg time: {yolo_avg:.3f}s, CLIP text avg: {clip_text_avg:.3f}s, sim range: {sim_min:.3f}-{sim_max:.3f}")

# ---------------- PLOTTING (single graph with two lines) ----------------
plt.figure(figsize=(12, 6))

# Create combined x labels: first image queries then second image queries
labels_img1 = [f"Img1: {q}" for q in QUERIES_IMG1]
labels_img2 = [f"Img2: {q}" for q in QUERIES_IMG2]
combined_labels = labels_img1 + labels_img2

# x positions for two lines
x1 = np.arange(len(QUERIES_IMG1))
x2 = np.arange(len(QUERIES_IMG2)) + len(QUERIES_IMG1) + 0.5  # offset second group to right for clarity

# Plot lines
plt.plot(x1, similarities_img1, marker='o', linewidth=2.5, markersize=8, label=os.path.basename(IMAGE_FILES[0]), color='tab:blue')
plt.plot(x2, similarities_img2, marker='o', linewidth=2.5, markersize=8, label=os.path.basename(IMAGE_FILES[1]), color='tab:orange')

# X ticks: place ticks centered under each group element
xticks_positions = list(x1) + list(x2)
plt.xticks(xticks_positions, combined_labels, rotation=30, ha='right', fontsize=9)

# Y axis
plt.ylim(0.0, 1.0)
plt.ylabel("CLIP Similarity (cosine)")
plt.title("Similarity vs Query - Two Images (GT(Ground Truth) = largest detected person)")

# annotate similarity values above markers
for xi, y in zip(x1, similarities_img1):
    plt.text(xi, y + 0.02, f"{y:.3f}", ha='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

for xi, y in zip(x2, similarities_img2):
    plt.text(xi, y + 0.02, f"{y:.3f}", ha='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# metrics box
metrics_text = (
    f"YOLO avg time: {yolo_avg:.2f} s\n"
    f"CLIP text avg encode: {clip_text_avg:.2f} s\n"
    f"Similarity range: {sim_min:.3f} – {sim_max:.3f}"
)
plt.gca().text(0.99, 0.02, metrics_text, transform=plt.gca().transAxes,
               fontsize=10, va='bottom', ha='right',
               bbox=dict(boxstyle="round,pad=0.6", fc="wheat", alpha=0.95, ec="0.3"))

plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()

# SAVE BEFORE SHOW
plt.savefig(FIG_PATH, dpi=300)
plt.show()
print(f"Saved figure to: {FIG_PATH}")

