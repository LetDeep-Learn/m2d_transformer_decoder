# model_ssl.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ResNetBackbone(nn.Module):
    """
    ResNet50 backbone adapted to accepts 1-channel input.
    We remove the final fc and return the pooled features (2048-d).
    """
    def __init__(self, pretrained=False, in_channels=1):
        super().__init__()
        resnet = models.resnet50(weights=None) if not pretrained else models.resnet50(weights="IMAGENET1K_V2")
        # modify first conv to accept 'in_channels' (1 for grayscale)
        if in_channels != 3:
            # create new conv with same params but different in_channels
            old_conv = resnet.conv1
            new_conv = nn.Conv2d(in_channels, old_conv.out_channels,
                                 kernel_size=old_conv.kernel_size,
                                 stride=old_conv.stride,
                                 padding=old_conv.padding,
                                 bias=(old_conv.bias is not None))
            # If input is grayscale, initialize new conv weight by summing rgb channels of original or by copying
            with torch.no_grad():
                if in_channels == 1:
                    # average the original weights across channel dim if pretrained available
                    try:
                        w = old_conv.weight.data
                        new_conv.weight.data = w.mean(dim=1, keepdim=True)
                    except Exception:
                        nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
                else:
                    nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
            resnet.conv1 = new_conv

        # remove classifier head
        resnet.fc = nn.Identity()
        self.backbone = resnet

    def forward(self, x):
        # returns [B, 2048]
        return self.backbone(x)


class ProjectionHead(nn.Module):
    """
    Two-layer MLP projection head with BatchNorm (SimCLR style).
    Input dim should match backbone output (2048 for ResNet50).
    """
    def __init__(self, in_dim=2048, hidden_dim=1024, proj_dim=256, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim)  # note: final BN as used in SimCLR
        )

    def forward(self, x):
        return self.net(x)


class ModiSSLModel(nn.Module):
    """
    SSL model: backbone + projection head.
    Forward returns normalized projection vectors for two views.
    """
    def __init__(self, proj_dim=256, hidden_dim=1024, pretrained_backbone=False, dropout=0.0):
        super().__init__()
        self.backbone = ResNetBackbone(pretrained=pretrained_backbone, in_channels=1)
        self.projection = ProjectionHead(in_dim=2048, hidden_dim=hidden_dim, proj_dim=proj_dim, dropout=dropout)

    def forward(self, x1, x2):
        """
        x1, x2: [B, 1, H, W] (grayscale)
        returns: p1_norm, p2_norm  (both [B, proj_dim]) normalized
        """
        # backbone returns [B, 2048]
        f1 = self.backbone(x1)
        f2 = self.backbone(x2)

        p1 = self.projection(f1)
        p2 = self.projection(f2)

        p1 = F.normalize(p1, dim=1)
        p2 = F.normalize(p2, dim=1)
        return p1, p2


# optional compatibility wrapper for old ssl_loss
def ssl_loss(z1, z2, alpha=0.5):
    cos_loss = 1 - F.cosine_similarity(z1, z2, dim=-1).mean()
    mse_loss = F.mse_loss(z1, z2)
    return alpha * cos_loss + (1 - alpha) * mse_loss


if __name__ == "__main__":
    # quick sanity check
    model = ModiSSLModel()
    x = torch.randn(2, 1, 128, 512)
    z1, z2 = model(x, x)
    print("z shapes:", z1.shape, z2.shape)
