import os
import sys
import logging
import torch
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
import clip
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# --- CONFIGURATION ---
YOLO_MODEL_NAME = "yolov8m.pt"
CLIP_MODEL_NAME = "ViT-B/32"  # change to "ViT-L/14" if you have GPU & memory
MAX_DISPLAY_WIDTH = 1200
MAX_DISPLAY_HEIGHT = 800
MIN_SIM_THRESHOLD = 0.15  # tune this for acceptance
# ---------------------

# Silence ultralytics (minimal)
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# -------------------------------
# 1. SELECT IMAGE
# -------------------------------
Tk().withdraw()
IMAGE_PATH = askopenfilename(title="Select image file", filetypes=[("JPEG", "*.jpg;*.jpeg"), ("PNG", "*.png")])
if not IMAGE_PATH or not os.path.exists(IMAGE_PATH):
    print("ERROR: No valid image selected.")
    sys.exit(1)

img_pil = Image.open(IMAGE_PATH).convert("RGB")
img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
ORIGINAL_H, ORIGINAL_W = img_cv.shape[:2]

# -------------------------------
# 2. ASK USER FOR QUERY
# -------------------------------
query_text = input("\nEnter the query description (example: 'man in black jacket'): ").strip()
if not query_text:
    print("No query entered. Using default query: 'man in a black jacket'")
    query_text = "man in a black jacket"

print(f"\nUsing query: '{query_text}' (Full Body matching by default)")

# -------------------------------
# 3. LOAD CLIP MODEL
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading CLIP model '{CLIP_MODEL_NAME}' on device: {device}")
clip_model, preprocess = clip.load(CLIP_MODEL_NAME, device=device)

# -------------------------------
# 4. LOAD YOLOv8 MODEL
# -------------------------------
print(f"Loading YOLOv8 model '{YOLO_MODEL_NAME}' (this may download weights once)...")
yolo_model = YOLO(YOLO_MODEL_NAME)

# -------------------------------
# 5. DETECT PERSONS
# -------------------------------
results = yolo_model.predict(source=IMAGE_PATH, classes=[0], imgsz=1024, verbose=False)
boxes = results[0].boxes.xyxy.cpu().numpy()

if len(boxes) == 0:
    print("No persons detected by YOLOv8.")
    # show scaled original
    h, w = img_cv.shape[:2]
    scale = min(MAX_DISPLAY_WIDTH / w, MAX_DISPLAY_HEIGHT / h, 1.0)
    img_to_show = cv2.resize(img_cv, (int(w*scale), int(h*scale))) if scale < 1.0 else img_cv
    cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Original Image", img_to_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sys.exit(0)
else:
    print(f"Found {len(boxes)} persons. Proceeding to CLIP matching.")

# -------------------------------
# 6. ENCODE QUERY (multiple templates + head-detection)
# -------------------------------
short_variant = query_text.split(',')[0].strip()
templates = [
    query_text,
    f"a photo of {short_variant}",
    f"a portrait of {short_variant}",
    f"{short_variant}"
]
templates = list(dict.fromkeys(templates))  # dedupe & preserve order

# Detect head/hairstyle focused queries
hair_keywords = {"hair", "hairstyle", "afro", "curly", "bun", "ponytail", "beard", "mustache"}
query_words = set([w.lower().strip(".,") for w in query_text.split()])
HEAD_FOCUSED_QUERY = len(hair_keywords.intersection(query_words)) > 0
if HEAD_FOCUSED_QUERY:
    print("Detected head/hairstyle-related query → preferring HEAD crop for matching.")

text_tokens = clip.tokenize(templates).to(device)
with torch.no_grad():
    text_features_all = clip_model.encode_text(text_tokens)
    text_features_all /= text_features_all.norm(dim=-1, keepdim=True)  # (T, D)

# -------------------------------
# 7. MATCH DETECTIONS (evaluate head/torso/full)
# -------------------------------
best_score = -1.0
best_box = None
best_template = None
best_crop_used = "full"

for i, box in enumerate(boxes):
    x1_full, y1_full, x2_full, y2_full = map(int, box)
    w = x2_full - x1_full
    h = y2_full - y1_full
    if w <= 10 or h <= 10:
        continue

    # define crops
    full_crop = (x1_full, y1_full, x2_full, y2_full)
    head_y2 = y1_full + int(0.45 * h)  # top ~45%
    head_crop = (x1_full, y1_full, x2_full, head_y2)
    torso_y1 = y1_full + int(0.25 * h)
    torso_y2 = y1_full + int(0.75 * h)
    torso_crop = (x1_full, torso_y1, x2_full, torso_y2)

    if HEAD_FOCUSED_QUERY:
        candidates = [("head", head_crop)]
    else:
        candidates = [("head", head_crop), ("torso", torso_crop), ("full", full_crop)]

    for crop_type, (cx1, cy1, cx2, cy2) in candidates:
        cx1, cy1 = max(0, cx1), max(0, cy1)
        cx2, cy2 = min(ORIGINAL_W, cx2), min(ORIGINAL_H, cy2)
        if (cx2 - cx1) <= 10 or (cy2 - cy1) <= 10:
            continue

        person_crop = img_pil.crop((cx1, cy1, cx2, cy2))
        person_input = preprocess(person_crop).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(person_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            sims = (text_features_all @ image_features.T).squeeze(-1).cpu().numpy()  # (T,)
            top_idx = int(np.argmax(sims))
            top_sim = float(sims[top_idx])

        # boost head crop for head-focused queries
        boost = 1.2 if (HEAD_FOCUSED_QUERY and crop_type == "head") else 1.0
        scored = top_sim * boost

        if scored > best_score:
            best_score = scored
            best_box = (x1_full, y1_full, x2_full, y2_full)  # display full person for context
            best_template = templates[top_idx]
            best_crop_used = crop_type

# -------------------------------
# 8. DISPLAY RESULT (SHOW FULL IMAGE)
# -------------------------------
print(f"\nQUERY: '{query_text}'")
if best_box is None:
    print("No valid person crops found.")
else:
    print(f"Best template: '{best_template}'  → Score: {best_score:.3f}  (crop used: {best_crop_used})")

# draw bounding box (on original img copy)
display_img = img_cv.copy()
if best_box and best_score > MIN_SIM_THRESHOLD:
    x1, y1, x2, y2 = best_box
    cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Prepare label
    display_text = f"{best_template} (Score: {best_score:.3f})"

    # Text parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    line_spacing = 6  # pixels between wrapped lines

    img_h, img_w = display_img.shape[:2]

    # Helper to wrap text into lines that fit the image width
    def wrap_text(text, max_width):
        words = text.split()
        lines = []
        if not words:
            return [""]
        cur_line = words[0]
        for w in words[1:]:
            test_line = cur_line + " " + w
            (tw, th), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
            if tw <= max_width:
                cur_line = test_line
            else:
                lines.append(cur_line)
                cur_line = w
        lines.append(cur_line)
        return lines

    # compute available width to the right of the box; if not enough use full image width minus margin
    margin = 8
    available_right = img_w - x1 - margin
    max_text_width = max(available_right, img_w - 2 * margin)

    # wrap into lines
    lines = wrap_text(display_text, max_text_width)

    # compute text block size
    line_heights = []
    max_tw = 0
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
        line_heights.append(th)
        if tw > max_tw:
            max_tw = tw
    text_block_h = sum(line_heights) + (len(lines)-1) * line_spacing
    text_block_w = max_tw

    # decide text origin: prefer above the box; if not enough space put it inside box top or below
    x_text = x1
    y_text_top = y1 - 10  # ideal top baseline
    # if top is too small (text would go above image), move inside box or below
    # compute baseline y for first line
    first_line_h = line_heights[0] if line_heights else 0
    if y1 - text_block_h - 10 < 0:
        # not enough space above box -> try inside box (slightly below y1)
        y_text_start = y1 + first_line_h + 6
        if y_text_start + text_block_h > img_h:
            # still overflows -> place at bottom with margin
            y_text_start = img_h - text_block_h - 10
    else:
        # place above
        y_text_start = y1 - text_block_h - 6

    # ensure x_text keeps text inside image horizontally
    if x_text + text_block_w + margin > img_w:
        x_text = max(margin, img_w - text_block_w - margin)

    # draw semi-transparent rectangle as background for readability
    rect_x1 = x_text - 4
    rect_y1 = int(y_text_start - first_line_h - 4)  # adjust to top of first baseline
    rect_x2 = int(x_text + text_block_w + 4)
    rect_y2 = int(y_text_start + text_block_h + 4)
    # clamp rectangle inside image bounds
    rect_x1 = max(0, rect_x1)
    rect_y1 = max(0, rect_y1)
    rect_x2 = min(img_w, rect_x2)
    rect_y2 = min(img_h, rect_y2)

    overlay = display_img.copy()
    cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
    alpha = 0.45  # transparency
    cv2.addWeighted(overlay, alpha, display_img, 1 - alpha, 0, display_img)

    # draw each wrapped line
    y = int(y_text_start)
    for i, line in enumerate(lines):
        (tw, th), baseline = cv2.getTextSize(line, font, font_scale, thickness)
        # baseline used by OpenCV is the baseline; we draw the text baseline at y + (th - baseline)
        cv2.putText(display_img, line, (int(x_text), y + th), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
        y += th + line_spacing

    print(f"Match accepted (score {best_score:.3f}) and highlighted on image.")
else:
    print(f"No strong match (best score: {best_score:.3f}). Showing image without selection.")

# scale to fit screen/window while preserving aspect ratio
h, w = display_img.shape[:2]
scale_w = MAX_DISPLAY_WIDTH / w
scale_h = MAX_DISPLAY_HEIGHT / h
scale = min(scale_w, scale_h, 1.0)
if scale < 1.0:
    disp_w, disp_h = int(w * scale), int(h * scale)
    img_to_show = cv2.resize(display_img, (disp_w, disp_h))
else:
    img_to_show = display_img

cv2.namedWindow("CLIP-YOLO Matched Person", cv2.WINDOW_NORMAL)
cv2.imshow("CLIP-YOLO Matched Person", img_to_show)
cv2.waitKey(0)
cv2.destroyAllWindows()
