# FILE: similarity_histogram.py
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import clip
# pyrefly: ignore [missing-import]
import cv2
import numpy as np
from PIL import Image
# pyrefly: ignore [missing-import]
from ultralytics import YOLO
import matplotlib.pyplot as plt

# ------------------------------
# Load Models
# ------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
yolo = YOLO("yolov8n.pt")

IMAGE_PATH = "../../data/images/Image02.jpg"
query = "woman with curly hair"

img = cv2.imread(IMAGE_PATH)

# Encode text
text_tokens = clip.tokenize([query]).to(device)
with torch.no_grad():
    text_feat = clip_model.encode_text(text_tokens)
    text_feat /= text_feat.norm(dim=-1, keepdim=True)

# YOLO detection
results = yolo(img)
boxes = results[0].boxes.xyxy.cpu().numpy()

similarities = []

for box in boxes:
    x1, y1, x2, y2 = map(int, box)
    crop = img[y1:y2, x1:x2]

    pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    crop_tensor = preprocess(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        img_feat = clip_model.encode_image(crop_tensor)
        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        sim = (text_feat @ img_feat.T).item()

    similarities.append(sim)

# ------------------------------
# Plot Histogram
# ------------------------------
plt.figure(figsize=(8, 5))
plt.hist(similarities, bins=8, color="skyblue", edgecolor="black")
plt.title("CLIP Similarity Score Distribution")
plt.xlabel("Similarity Score")
plt.ylabel("Frequency")
plt.grid(True)

# SAVE BEFORE SHOW
plt.savefig("../../graphs/similarity_histogram.png", dpi=300)
plt.show()