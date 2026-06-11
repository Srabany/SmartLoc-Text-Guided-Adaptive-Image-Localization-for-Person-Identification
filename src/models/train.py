import os
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torch.optim as optim
# pyrefly: ignore [missing-import]
from torch.utils.data import DataLoader
# pyrefly: ignore [missing-import]
import clip
from dataset import SmartLocDataset
import sys

# --- CONFIGURATION ---
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 5e-6  # Low LR is crucial for fine-tuning CLIP
CLIP_MODEL_NAME = "ViT-B/32"
# ---------------------

def get_loss(image_features, text_features, logit_scale):
    """
    Symmetric contrastive loss (InfoNCE) used in CLIP.
    """
    # Normalize features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Cosine similarity scaled by learned temperature
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logit_scale * text_features @ image_features.t()
    
    # Labels are just the diagonal (matching image i with text i)
    batch_size = image_features.shape[0]
    labels = torch.arange(batch_size, device=image_features.device)
    
    # Cross entropy loss over both axes
    loss_i = nn.CrossEntropyLoss()(logits_per_image, labels)
    loss_t = nn.CrossEntropyLoss()(logits_per_text, labels)
    
    return (loss_i + loss_t) / 2

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print(f"Loading CLIP model '{CLIP_MODEL_NAME}'...")
    model, preprocess = clip.load(CLIP_MODEL_NAME, device=device, jit=False)
    
    # Unfreeze model weights for fine-tuning
    for param in model.parameters():
        param.requires_grad = True
        
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    json_path = os.path.join(base_dir, "data", "dataset.json")
    image_dir = os.path.join(base_dir, "data", "images")
    
    if not os.path.exists(json_path):
        print(f"Error: {json_path} does not exist. Please run build_dataset.py first.")
        sys.exit(1)
        
    dataset = SmartLocDataset(json_path, image_dir, preprocess)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Optimizer (AdamW is standard for transformers)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.2)
    
    # Directory to save weights
    weights_dir = os.path.join(base_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    
    print("\nStarting Fine-tuning...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (images, texts) in enumerate(dataloader):
            images = images.to(device)
            texts = texts.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            image_features, text_features = model(images, texts)
            
            # Compute loss
            loss = get_loss(image_features, text_features, model.logit_scale.exp())
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (good practice for stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            print(f"Epoch [{epoch+1}/{EPOCHS}] Batch [{batch_idx+1}/{len(dataloader)}] Loss: {loss.item():.4f}", end='\r')
            
        avg_loss = total_loss / len(dataloader)
        print(f"\nEpoch [{epoch+1}/{EPOCHS}] Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(weights_dir, f"clip_finetuned_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    print("Fine-tuning complete!")

if __name__ == "__main__":
    train()
