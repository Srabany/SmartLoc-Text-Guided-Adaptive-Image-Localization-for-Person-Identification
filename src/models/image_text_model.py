import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ImageTextModel(nn.Module):
    def __init__(self, proj_dim: int = 512):
        super(ImageTextModel, self).__init__()

        # pretrained ResNet18 backbone
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])  # (B,512,1,1)

        # projection layer
        self.image_fc = nn.Linear(512, proj_dim)

    def forward(self, image_tensor: torch.Tensor, text_embedding: torch.Tensor = None):
        """
        RETURNS ONLY IMAGE EMBEDDING → 512-D
        text_embedding is NOT required
        """
        batch_size = image_tensor.shape[0]

        # CNN: (B,512,1,1)
        img_feat = self.cnn(image_tensor)
        img_feat = img_feat.view(batch_size, -1)  # (B,512)

        img_proj = self.image_fc(img_feat)  # (B,512)

        # Normalize
        img_norm = F.normalize(img_proj, dim=1)

        return img_norm
