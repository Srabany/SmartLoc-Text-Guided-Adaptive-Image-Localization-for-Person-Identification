"""Quick smoke test for staged_search.py — verifies all components work."""
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "models"))

from staged_search import FeatureCache, file_hash, encode_text, rank_candidates
import numpy as np

# Test 1: Cache system
cache = FeatureCache("data/.cache")
print("Test 1 - Cache system: OK")

# Test 2: Load models
import torch
import clip
from ultralytics import YOLO

device = "cpu"
print("Loading CLIP...", end=" ", flush=True)
clip_model, preprocess = clip.load("ViT-B/32", device=device)
print("OK")

print("Loading YOLO...", end=" ", flush=True)
yolo_model = YOLO("yolov8m.pt")
print("OK")

# Test 3: Process one image with YOLO
from PIL import Image

test_img = os.path.join("data", "images", "Image01.JPG")
print(f"Processing {test_img}...", end=" ", flush=True)
results = yolo_model.predict(source=test_img, classes=[0], imgsz=1024, verbose=False)
boxes = results[0].boxes.xyxy.cpu().numpy()
print(f"{len(boxes)} persons detected")

if len(boxes) == 0:
    print("No persons found in test image - try a different image")
    sys.exit(1)

# Test 4: CLIP encode a person crop
img_pil = Image.open(test_img).convert("RGB")
x1, y1, x2, y2 = map(int, boxes[0])
crop = img_pil.crop((x1, y1, x2, y2))
inp = preprocess(crop).unsqueeze(0).to(device)

with torch.no_grad():
    feat = clip_model.encode_image(inp)
    feat = feat / feat.norm(dim=-1, keepdim=True)
print(f"Test 4 - CLIP feature shape: {feat.shape} OK")

# Test 5: Text encoding
text_feat = encode_text(clip_model, device, "person wearing red shirt")
print(f"Test 5 - Text encoding shape: {text_feat.shape} OK")

# Test 6: Similarity computation
person = {
    "image_name": "Image01.JPG",
    "person_idx": 0,
    "bbox": [x1, y1, x2, y2],
    "features": feat.cpu().numpy().squeeze(0),
}
scored = rank_candidates([person], text_feat)
score_pct = scored[0]["score"] * 100
print(f"Test 6 - Similarity score: {score_pct:.1f}% OK")

# Test 7: Cache save/load roundtrip
test_hash = "test_smoke"
bboxes = np.array([[x1, y1, x2, y2]], dtype=int)
feats = feat.cpu().numpy()
cache.save(test_hash, bboxes, feats)
loaded = cache.load(test_hash)
assert loaded is not None, "Cache load failed"
assert np.allclose(loaded["features"], feats, atol=1e-5), "Feature mismatch after cache roundtrip"
# Cleanup test cache
os.remove(os.path.join("data", ".cache", f"{test_hash}.npz"))
print("Test 7 - Cache save/load roundtrip: OK")

print()
print("=" * 52)
print("  ALL TESTS PASSED — Model is working correctly!")
print("=" * 52)
print()
print(f"  YOLO detected {len(boxes)} persons in Image01.JPG")
print(f"  CLIP similarity for 'red shirt' query: {score_pct:.1f}%")
print(f"  Feature cache system: working")
print(f"  Device: {device}")
print()
