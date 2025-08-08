# model_ssl.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------
# Residual block
# ------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out

# ------------------------
# Simple SPP (Spatial Pyramid Pooling)
# ------------------------
class SpatialPyramidPooling(nn.Module):
    def __init__(self, in_ch, out_dim=512):
        super().__init__()
        # We'll perform adaptive pooling at multiple scales and concatenate
        self.pool_sizes = [1, 2, 4]  # global, coarse, fine
        self.out_dim = out_dim
        # After concat we'll use a small FC to project to out_dim
        self.fc = nn.Sequential(
            nn.Linear(in_ch * sum([s * s for s in self.pool_sizes]), out_dim),
            nn.ReLU(),
            nn.BatchNorm1d(out_dim)
        )

    def forward(self, x):
        """
        x: [B, C, H, W]
        returns: [B, out_dim]
        """
        B, C, H, W = x.shape
        pooled = []
        for p in self.pool_sizes:
            out = F.adaptive_avg_pool2d(x, (p, p))  # [B, C, p, p]
            pooled.append(out.view(B, -1))          # [B, C*p*p]
        cat = torch.cat(pooled, dim=1)              # [B, C * sum(p*p)]
        return self.fc(cat)                         # [B, out_dim]

# ------------------------
# Encoder (ResNet-ish + SPP)
# ------------------------
class CNNEncoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, out_dim=512):
        super().__init__()
        # initial conv: keep spatial resolution reduction gentle
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),  # [B, base, H/2, W/2]
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # [B, base, H/4, W/4]
        )

        # residual layers (a compact ResNet-like stack)
        self.layer1 = self._make_layer(base_channels, base_channels, blocks=2, stride=1)   # [B, base, H/4, W/4]
        self.layer2 = self._make_layer(base_channels, base_channels*2, blocks=2, stride=2) # [B, base*2, H/8, W/8]
        self.layer3 = self._make_layer(base_channels*2, base_channels*4, blocks=2, stride=2)# [B, base*4, H/16, W/16]
        self.layer4 = self._make_layer(base_channels*4, base_channels*8, blocks=1, stride=2)# [B, base*8, H/32, W/32]

        # small conv to reduce channels before SPP (keeps compute reasonable)
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(base_channels*8, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

        # SPP to get a fixed-size descriptor
        self.spp = SpatialPyramidPooling(out_dim, out_dim)

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_ch, out_ch, stride=stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        x: [B, 1, 128, 512]
        returns: [B, out_dim]  (e.g., [B, 512])
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.reduce_conv(x)
        feat = self.spp(x)  # [B, out_dim]
        return feat

# ------------------------
# Projection Head (2-layer MLP with BatchNorm)
# ------------------------
class SSLProjectionHead(nn.Module):
    def __init__(self, in_dim, proj_dim=256, hidden_dim=1024, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim)  # as in SimCLR, BN on projection
        )

    def forward(self, x):
        return self.net(x)

# ------------------------
# Full SSL model
# ------------------------
class ModiSSLModel(nn.Module):
    def __init__(self, proj_dim=256, encoder_out=512, dropout=0.3):
        super().__init__()
        self.encoder = CNNEncoder(in_channels=1, base_channels=64, out_dim=encoder_out)
        self.projection = SSLProjectionHead(encoder_out, proj_dim=proj_dim, hidden_dim=1024, dropout=dropout)

    def forward(self, x1, x2):
        # x1, x2: [B, 1, H, W]
        z1 = self.encoder(x1)             # [B, encoder_out]
        z2 = self.encoder(x2)             # [B, encoder_out]

        p1 = self.projection(z1)          # [B, proj_dim]
        p2 = self.projection(z2)          # [B, proj_dim]

        # Normalize outputs for contrastive loss
        p1_norm = F.normalize(p1, dim=-1)
        p2_norm = F.normalize(p2, dim=-1)
        return p1_norm, p2_norm

# ------------------------
# Optional combined loss (kept for reference)
# ------------------------
def ssl_loss(z1, z2, alpha=0.5):
    """
    Compatibility wrapper if you still want to use the earlier cosine+MSE style loss.
    For contrastive learning prefer NT-Xent (InfoNCE) implemented in train code.
    """
    cos_loss = 1 - F.cosine_similarity(z1, z2, dim=-1).mean()
    mse_loss = F.mse_loss(z1, z2)
    return alpha * cos_loss + (1 - alpha) * mse_loss

# Quick sanity: if file executed directly, run a tiny forward pass
if __name__ == "__main__":
    model = ModiSSLModel().eval()
    x = torch.randn(2, 1, 128, 512)
    z1, z2 = model(x, x)
    print("z1.shape, z2.shape:", z1.shape, z2.shape)
