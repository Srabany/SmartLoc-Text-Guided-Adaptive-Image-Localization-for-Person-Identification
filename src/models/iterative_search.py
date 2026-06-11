import os
import sys
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import clip
# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from PIL import Image
# pyrefly: ignore [missing-import]
from ultralytics import YOLO

# --- CONFIGURATION ---
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "images"))
YOLO_MODEL_NAME = "yolov8m.pt"
CLIP_MODEL_NAME = "ViT-B/32"

GRID_W = 4
GRID_H = 2
CELL_W = 200
CELL_H = 300
# ---------------------

import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

def create_grid(results, grid_w=4, grid_h=2, cell_w=200, cell_h=300):
    grid = np.zeros((grid_h * cell_h, grid_w * cell_w, 3), dtype=np.uint8)
    
    for i, (sim, item) in enumerate(results):
        if i >= grid_w * grid_h:
            break
            
        r = i // grid_w
        c = i % grid_w
        
        pct = sim * 100
        
        # Convert PIL to CV2 BGR
        img_bgr = cv2.cvtColor(np.array(item['crop']), cv2.COLOR_RGB2BGR)
        
        # Resize image
        img_resized = cv2.resize(img_bgr, (cell_w, cell_h - 30))
        
        # Create a text bar
        bar = np.zeros((30, cell_w, 3), dtype=np.uint8)
        color = (0, 255, 0) if pct > 25 else ((0, 165, 255) if pct > 18 else (200, 200, 200))
        text = f"{pct:.1f}% Match"
        cv2.putText(bar, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Combine
        cell = np.vstack([img_resized, bar])
        
        # Place in grid
        grid[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w] = cell
        
    return grid

def main():
    print("=" * 60)
    print("  SmartLoc Terminal Iterative Search")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model on {device}...")
    clip_model, preprocess = clip.load(CLIP_MODEL_NAME, device=device)
    
    print(f"Loading YOLOv8 model...")
    yolo_model = YOLO(YOLO_MODEL_NAME)
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: Image directory not found at {DATA_DIR}")
        sys.exit(1)
        
    image_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print("\nScanning images and pre-computing features...")
    crops_data = []
    
    for i, img_name in enumerate(image_files):
        print(f"\rProcessing {i+1}/{len(image_files)}...", end="")
        img_path = os.path.join(DATA_DIR, img_name)
        try:
            img_pil = Image.open(img_path).convert("RGB")
        except:
            continue
            
        results = yolo_model.predict(source=img_path, classes=[0], imgsz=1024, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            if (x2 - x1) < 20 or (y2 - y1) < 20:
                continue
                
            person_crop = img_pil.crop((x1, y1, x2, y2))
            person_input = preprocess(person_crop).unsqueeze(0).to(device)
            
            with torch.no_grad():
                image_features = clip_model.encode_image(person_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
            crops_data.append({
                "image_name": img_name,
                "crop": person_crop,
                "features": image_features.cpu().numpy()
            })
            
    print(f"\nDone! Extracted {len(crops_data)} persons.")
    
    if not crops_data:
        print("No persons found. Exiting.")
        sys.exit(0)
        
    prompts = []
    cv2.namedWindow("Top Matches", cv2.WINDOW_NORMAL)
    
    while True:
        print("\n" + "-"*60)
        query = input("Enter description (or 'quit' to exit, 'clear' to reset prompt): ").strip()
        
        if query.lower() == 'quit':
            break
        elif query.lower() == 'clear':
            prompts = []
            print("Search history cleared.")
            continue
        elif not query:
            continue
            
        prompts.append(query)
        combined_prompt = ", ".join(prompts)
        print(f"Current Combined Prompt: '{combined_prompt}'")
        
        text_tokens = clip.tokenize([combined_prompt]).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features_np = text_features.cpu().numpy()
            
        # Score crops
        results = []
        for item in crops_data:
            img_feat = item["features"]
            sim = (text_features_np @ img_feat.T).squeeze()
            results.append((sim, item))
            
        # Sort top K
        results.sort(key=lambda x: x[0], reverse=True)
        
        # Display Grid
        grid_img = create_grid(results, GRID_W, GRID_H, CELL_W, CELL_H)
        cv2.imshow("Top Matches", grid_img)
        cv2.waitKey(1) # Refresh window, don't block
        
        print(f"Showing Top {min(GRID_W * GRID_H, len(results))} matches in the OpenCV window.")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
