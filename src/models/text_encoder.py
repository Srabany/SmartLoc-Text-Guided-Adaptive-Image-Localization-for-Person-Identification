import torch
import torch.nn as nn
import torch.nn.functional as F

class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int = 5000, embed_dim: int = 512):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(tokens)
        pooled = emb.mean(dim=1)
        proj = self.fc(pooled)
        norm = F.normalize(proj, dim=1)
        return norm

    def encode_text(self, text: str) -> torch.Tensor:
        token_ids = [ord(c) % self.embedding.num_embeddings for c in text]
        if len(token_ids) == 0:
            token_ids = [0]
        tokens = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(self.embedding.weight.device)
        return self.forward(tokens)

