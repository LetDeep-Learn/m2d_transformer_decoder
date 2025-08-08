# model_ssl.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SSLProjectionHead(nn.Module):
    def __init__(self, in_dim, proj_dim=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, proj_dim),
            nn.LayerNorm(proj_dim)
        )

    def forward(self, x):
        return self.net(x)


class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),  # [64, 64, 256]
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2), # [128, 32, 128]
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2), # [256, 16, 64]
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)) # [512, 1, 1]
        )

    def forward(self, x):
        feat = self.encoder(x)          # [B, 512, 1, 1]
        feat = feat.view(feat.size(0), -1)  # [B, 512]
        return feat


class ModiSSLModel(nn.Module):
    def __init__(self, proj_dim=256, dropout=0.3):
        super().__init__()
        self.encoder = CNNEncoder()
        self.projection = SSLProjectionHead(512, proj_dim, dropout)

    def forward(self, x1, x2):
        # Extract embeddings
        z1 = self.projection(self.encoder(x1))
        z2 = self.projection(self.encoder(x2))

        # Normalize for cosine similarity
        z1_norm = F.normalize(z1, dim=-1)
        z2_norm = F.normalize(z2, dim=-1)

        return z1_norm, z2_norm


def ssl_loss(z1, z2, alpha=0.5):
    # Cosine similarity loss
    cos_loss = 1 - F.cosine_similarity(z1, z2, dim=-1).mean()

    # MSE loss on raw normalized vectors
    mse_loss = F.mse_loss(z1, z2)

    return alpha * cos_loss + (1 - alpha) * mse_loss
