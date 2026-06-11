import os
import json
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from torch.utils.data import Dataset
from PIL import Image
# pyrefly: ignore [missing-import]
import clip

class SmartLocDataset(Dataset):
    def __init__(self, json_path, image_dir, preprocess_fn):
        """
        Args:
            json_path (str): Path to dataset.json
            image_dir (str): Path to the directory containing images
            preprocess_fn (callable): CLIP preprocessing function
        """
        self.image_dir = image_dir
        self.preprocess = preprocess_fn
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Dataset annotation file not found: {json_path}")
            
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        
        img_name = item["image"]
        img_path = os.path.join(self.image_dir, img_name)
        bbox = item["bbox"]  # [x1, y1, x2, y2]
        query = item["query"]
        
        # Load image and crop
        img = Image.open(img_path).convert("RGB")
        x1, y1, x2, y2 = bbox
        crop = img.crop((x1, y1, x2, y2))
        
        # Apply CLIP preprocessing
        image_tensor = self.preprocess(crop)
        
        # Tokenize text
        # clip.tokenize returns a tensor of shape [1, 77], we squeeze it to [77]
        text_tensor = clip.tokenize([query], truncate=True).squeeze(0)
        
        return image_tensor, text_tensor

# Example usage/testing block
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, preprocess = clip.load("ViT-B/32", device=device)
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    json_path = os.path.join(base_dir, "data", "dataset.json")
    image_dir = os.path.join(base_dir, "data", "images")
    
    try:
        dataset = SmartLocDataset(json_path, image_dir, preprocess)
        print(f"Dataset loaded with {len(dataset)} samples.")
        img_tensor, text_tensor = dataset[0]
        print(f"Image tensor shape: {img_tensor.shape}")
        print(f"Text tensor shape: {text_tensor.shape}")
    except FileNotFoundError as e:
        print(e)
