# model_ssl.py

import torch.nn as nn

class SSLModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Example simple CNN backbone (replace with your real architecture)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.projection = nn.Linear(128 * 32 * 128, 256)

    def forward(self, x):
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        return self.projection(features)
