import os
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from torch.utils.data import DataLoader
# pyrefly: ignore [missing-import]
import clip
from dataset import SmartLocDataset
import sys
import numpy as np

# --- CONFIGURATION ---
BATCH_SIZE = 32
CLIP_MODEL_NAME = "ViT-B/32"
# ---------------------

def evaluate(checkpoint_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print(f"Loading base CLIP model '{CLIP_MODEL_NAME}'...")
    model, preprocess = clip.load(CLIP_MODEL_NAME, device=device, jit=False)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading fine-tuned weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']+1}")
    else:
        print("No valid checkpoint provided. Evaluating ZERO-SHOT base CLIP model.")
        
    model.eval()
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    json_path = os.path.join(base_dir, "data", "dataset.json")
    image_dir = os.path.join(base_dir, "data", "images")
    
    if not os.path.exists(json_path):
        print(f"Error: {json_path} does not exist.")
        sys.exit(1)
        
    dataset = SmartLocDataset(json_path, image_dir, preprocess)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Evaluating on {len(dataset)} samples...")
    
    all_image_features = []
    all_text_features = []
    
    with torch.no_grad():
        for images, texts in dataloader:
            images = images.to(device)
            texts = texts.to(device)
            
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)
            
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            all_image_features.append(image_features.cpu())
            all_text_features.append(text_features.cpu())
            
    # Concatenate all features
    all_image_features = torch.cat(all_image_features, dim=0).numpy() # (N, D)
    all_text_features = torch.cat(all_text_features, dim=0).numpy() # (N, D)
    
    N = all_image_features.shape[0]
    
    # Calculate similarity matrix (Texts x Images)
    # rows = texts, cols = images
    sim_matrix = all_text_features @ all_image_features.T
    
    top1_correct = 0
    top5_correct = 0
    
    for i in range(N):
        # The correct matching image for text i is at index i (since we are drawing pairs from the same dataset)
        # Note: If multiple crops have identical queries, this strict diagonal evaluation might be slightly pessimistic.
        # But it's standard for unique text-image pairs.
        row_sims = sim_matrix[i]
        
        # Get indices of top 5 highest similarities
        top5_indices = np.argsort(row_sims)[-5:][::-1]
        
        if i == top5_indices[0]:
            top1_correct += 1
        if i in top5_indices:
            top5_correct += 1
            
    top1_acc = (top1_correct / N) * 100
    top5_acc = (top5_correct / N) * 100
    
    print("\n--- Evaluation Results ---")
    print(f"Total Samples: {N}")
    print(f"Text-to-Image Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Text-to-Image Top-5 Accuracy: {top5_acc:.2f}%")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate CLIP model")
    parser.add_argument("--weights", type=str, default=None, help="Path to fine-tuned checkpoint (.pt file)")
    args = parser.parse_args()
    
    evaluate(args.weights)
