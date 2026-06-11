# FILE: performance_plot.py
# pyrefly: ignore [missing-import]
import torch\
# pyrefly: ignore [missing-import]
import clip
import time
# pyrefly: ignore [missing-import]
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load models
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
yolo = YOLO("yolov8n.pt")

query = "woman with curly hair"
text_tokens = clip.tokenize([query]).to(device)

def time_yolo():
    start = time.time()
    yolo("../../data/images/Image02.jpg")
    return time.time() - start

def time_clip():
    start = time.time()
    with torch.no_grad():
        clip_model.encode_text(text_tokens)
    return time.time() - start

yolo_time = time_yolo()
clip_time = time_clip()

plt.figure(figsize=(7, 5))
plt.bar(["YOLO Inference", "CLIP Text Encode"], [yolo_time, clip_time], 
        color=["blue", "green"])

plt.ylabel("Time (seconds)")
plt.title("Performance Comparison: YOLO vs. CLIP")
plt.grid(axis='y')

plt.savefig("../../graphs/performance_plot.png", dpi=300)
plt.show()