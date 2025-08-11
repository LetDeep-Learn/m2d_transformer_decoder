# # model_ssl.py
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import models

# class ResNetBackbone(nn.Module):
#     """
#     ResNet50 backbone adapted to accepts 1-channel input.
#     We remove the final fc and return the pooled features (2048-d).
#     """
#     def __init__(self, pretrained=False, in_channels=1):
#         super().__init__()
#         resnet = models.resnet50(weights=None) if not pretrained else models.resnet50(weights="IMAGENET1K_V2")
#         # modify first conv to accept 'in_channels' (1 for grayscale)
#         if in_channels != 3:
#             # create new conv with same params but different in_channels
#             old_conv = resnet.conv1
#             new_conv = nn.Conv2d(in_channels, old_conv.out_channels,
#                                  kernel_size=old_conv.kernel_size,
#                                  stride=old_conv.stride,
#                                  padding=old_conv.padding,
#                                  bias=(old_conv.bias is not None))
#             # If input is grayscale, initialize new conv weight by summing rgb channels of original or by copying
#             with torch.no_grad():
#                 if in_channels == 1:
#                     # average the original weights across channel dim if pretrained available
#                     try:
#                         w = old_conv.weight.data
#                         new_conv.weight.data = w.mean(dim=1, keepdim=True)
#                     except Exception:
#                         nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
#                 else:
#                     nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
#             resnet.conv1 = new_conv

#         # remove classifier head
#         resnet.fc = nn.Identity()
#         self.backbone = resnet

#     def forward(self, x):
#         # returns [B, 2048]
#         return self.backbone(x)


# class ProjectionHead(nn.Module):
#     """
#     Two-layer MLP projection head with BatchNorm (SimCLR style).
#     Input dim should match backbone output (2048 for ResNet50).
#     """
#     def __init__(self, in_dim=2048, hidden_dim=1024, proj_dim=256, dropout=0.0):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(in_dim, hidden_dim, bias=False),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, proj_dim, bias=False),
#             nn.BatchNorm1d(proj_dim)  # note: final BN as used in SimCLR
#         )

#     def forward(self, x):
#         return self.net(x)


# class ModiSSLModel(nn.Module):
#     """
#     SSL model: backbone + projection head.
#     Forward returns normalized projection vectors for two views.
#     """
#     def __init__(self, proj_dim=256, hidden_dim=1024, pretrained_backbone=False, dropout=0.0):
#         super().__init__()
#         self.backbone = ResNetBackbone(pretrained=pretrained_backbone, in_channels=1)
#         self.projection = ProjectionHead(in_dim=2048, hidden_dim=hidden_dim, proj_dim=proj_dim, dropout=dropout)

#     def forward(self, x1, x2):
#         """
#         x1, x2: [B, 1, H, W] (grayscale)
#         returns: p1_norm, p2_norm  (both [B, proj_dim]) normalized
#         """
#         # backbone returns [B, 2048]
#         f1 = self.backbone(x1)
#         f2 = self.backbone(x2)

#         p1 = self.projection(f1)
#         p2 = self.projection(f2)

#         p1 = F.normalize(p1, dim=1)
#         p2 = F.normalize(p2, dim=1)
#         return p1, p2


# # optional compatibility wrapper for old ssl_loss
# def ssl_loss(z1, z2, alpha=0.5):
#     cos_loss = 1 - F.cosine_similarity(z1, z2, dim=-1).mean()
#     mse_loss = F.mse_loss(z1, z2)
#     return alpha * cos_loss + (1 - alpha) * mse_loss


# if __name__ == "__main__":
#     # quick sanity check
#     model = ModiSSLModel()
#     x = torch.randn(2, 1, 128, 512)
#     z1, z2 = model(x, x)
#     print("z shapes:", z1.shape, z2.shape)





# model_ssl.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ---------- Helpers ----------
def _make_resnet(backbone_name: str, pretrained: bool):
    """
    Robustly create a torchvision resnet (handles different torchvision versions).
    backbone_name: 'resnet18', 'resnet34', 'resnet50', 'resnet101'
    """
    factory = getattr(models, backbone_name)
    try:
        # Newer torchvision uses weights=...; pass string or enum if pretrained True
        if pretrained:
            # prefer passing an enum-like name if available
            try:
                # some installations accept string names for weights (older code used this)
                return factory(weights="IMAGENET1K_V2")
            except Exception:
                # fallback to boolean-style API
                return factory(pretrained=True)
        else:
            try:
                return factory(weights=None)
            except Exception:
                return factory(pretrained=False)
    except TypeError:
        # final fallback
        return factory(pretrained=pretrained) if 'pretrained' in factory.__code__.co_varnames else factory()

# ---------- Backbone ----------
class ResNetBackbone(nn.Module):
    """
    ResNet backbone adapted to accept `in_channels` (1 for grayscale).
    Removes final fc and returns pooled features (out_dim is inferred).
    """
    def __init__(self, backbone_name="resnet50", pretrained=False, in_channels=1):
        """
        backbone_name: 'resnet18'|'resnet34'|'resnet50'|'resnet101'
        pretrained: whether to use ImageNet weights (best-effort)
        in_channels: typically 1 for grayscale
        """
        super().__init__()
        # build resnet
        resnet = _make_resnet(backbone_name, pretrained=pretrained)

        # record feature dimension from the existing classifier if present
        # Some torchvision versions: resnet.fc is nn.Linear(in_features, out_features)
        try:
            encoder_out_dim = resnet.fc.in_features
        except Exception:
            # fallback guess (resnet50 -> 2048)
            encoder_out_dim = 2048

        # adapt first conv if needed
        if in_channels != 3:
            old_conv = resnet.conv1
            new_conv = nn.Conv2d(in_channels, old_conv.out_channels,
                                 kernel_size=old_conv.kernel_size,
                                 stride=old_conv.stride,
                                 padding=old_conv.padding,
                                 bias=(old_conv.bias is not None))
            with torch.no_grad():
                # try to initialize sensibly if pretrained weights exist
                try:
                    # average RGB channels to initialize single-channel conv
                    w = old_conv.weight.data
                    new_conv.weight.data = w.mean(dim=1, keepdim=True)
                except Exception:
                    nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
            resnet.conv1 = new_conv

        # remove final fc classifier (we'll use pooled features)
        resnet.fc = nn.Identity()

        self.backbone = resnet
        self.out_dim = encoder_out_dim

    def forward(self, x):
        # returns [B, out_dim]
        return self.backbone(x)


# ---------- Projection Head ----------
class ProjectionHead(nn.Module):
    """
    Two-layer MLP projection head. SimCLR-style: Linear -> BN -> ReLU -> (Dropout) -> Linear -> BN
    """
    def __init__(self, in_dim, hidden_dim=1024, proj_dim=256, dropout=0.0, norm_last="bn"):
        """
        norm_last: 'bn'|'ln'|None - final normalization on projection (SimCLR uses BN)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim, bias=False),
        )
        # final normalization layer
        if norm_last == "bn":
            self.norm = nn.BatchNorm1d(proj_dim)
        elif norm_last == "ln":
            self.norm = nn.LayerNorm(proj_dim)
        else:
            self.norm = None

    def forward(self, x):
        out = self.net(x)
        if self.norm is not None:
            out = self.norm(out)
        return out


# ---------- Optional Predictor (SimSiam) ----------
class Predictor(nn.Module):
    """
    Small MLP predictor used by SimSiam (optional).
    """
    def __init__(self, in_dim, hidden_dim=512, out_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


# ---------- Full SSL model ----------
class ModiSSLModel(nn.Module):
    """
    Full SSL model: ResNet backbone + projection head.
    - proj_dim: projection (embedding) dimension
    - hidden_dim: hidden dim of projection MLP
    - backbone_name: 'resnet18'|'resnet34'|'resnet50'|'resnet101'
    - pretrained_backbone: bool attempt to load ImageNet weights
    - dropout: projection dropout
    - freeze_backbone: if True, backbone's parameters are frozen
    - use_predictor: optional small predictor (for SimSiam-style training)
    - norm_last: 'bn'|'ln'|None for final proj head norm
    """
    def __init__(self,
                 proj_dim=256,
                 hidden_dim=1024,
                 backbone_name="resnet50",
                 pretrained_backbone=False,
                 dropout=0.0,
                 freeze_backbone=False,
                 use_predictor=False,
                 norm_last="bn",
                 in_channels=1):
        super().__init__()
        self.backbone = ResNetBackbone(backbone_name=backbone_name,
                                       pretrained=pretrained_backbone,
                                       in_channels=in_channels)
        encoder_dim = self.backbone.out_dim
        self.encoder_dim = encoder_dim

        self.projection = ProjectionHead(in_dim=encoder_dim,
                                         hidden_dim=hidden_dim,
                                         proj_dim=proj_dim,
                                         dropout=dropout,
                                         norm_last=norm_last)
        self.proj_dim = proj_dim

        self.use_predictor = use_predictor
        if use_predictor:
            # predictor maps proj_dim -> proj_dim
            self.predictor = Predictor(proj_dim, hidden_dim=max(512, proj_dim//2), out_dim=proj_dim)
        else:
            self.predictor = None

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    # convenience methods
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw encoder features (before projection)."""
        return self.backbone(x)

    def project(self, feats: torch.Tensor) -> torch.Tensor:
        """Return projection (unnormalized)."""
        return self.projection(feats)

    def project_norm(self, feats: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized projection vector."""
        p = self.project(feats)
        return F.normalize(p, dim=1)

    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """Single-input forward returning normalized projection."""
        feats = self.encode(x)
        return self.project_norm(feats)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor = None):
        """
        If x2 is None -> returns normalized projection for x1 (B, proj_dim)
        If x2 provided -> returns tuple (p1_norm, p2_norm)
        For SimSiam-style training, if predictor enabled and called externally, you can pass through predictor.
        """
        if x2 is None:
            return self.forward_single(x1)

        f1 = self.encode(x1)
        f2 = self.encode(x2)
        p1 = self.project_norm(f1)
        p2 = self.project_norm(f2)

        if self.use_predictor:
            # return predictor outputs too (p1_pred, p2_pred, p1, p2) — but default behavior is return p1,p2
            p1_pred = self.predictor(p1)
            p2_pred = self.predictor(p2)
            return p1_pred, p2_pred, p1, p2

        return p1, p2

    # partial loading helper for backbone or whole model
    def load_backbone_state_dict(self, state_dict, strict=False):
        """
        Load a state_dict intended for backbone only (prefix-insensitive).
        This is helpful when loading external pretrained backbones.
        """
        # try to detect prefix like 'backbone.' or 'module.backbone.' etc.
        model_state = self.state_dict()
        filtered = {}
        for k, v in state_dict.items():
            # accept keys that match suffix of full model keys
            for full_k in model_state.keys():
                if full_k.endswith(k) or full_k.endswith(k.replace('module.', '')):
                    if model_state[full_k].size() == v.size():
                        filtered[full_k] = v
                        break
        model_state.update(filtered)
        self.load_state_dict(model_state)

    def get_backbone_state_dict(self):
        """Return backbone portion of state dict for saving."""
        sd = {}
        for k, v in self.state_dict().items():
            if k.startswith("backbone."):
                sd[k] = v
        return sd


# ---------- compatibility ssl_loss ----------
def ssl_loss(z1, z2, alpha=0.5):
    """
    Backward-compatible combination of cosine + mse (kept for compatibility).
    z1, z2 expected to be projection outputs (not necessarily normalized).
    """
    z1n = F.normalize(z1, dim=1)
    z2n = F.normalize(z2, dim=1)
    cos_loss = 1 - F.cosine_similarity(z1n, z2n, dim=-1).mean()
    mse_loss = F.mse_loss(z1n, z2n)
    return alpha * cos_loss + (1 - alpha) * mse_loss


# ---------- quick smoke test ----------
if __name__ == "__main__":
    # smoke-check with default resnet50
    m = ModiSSLModel(proj_dim=256, backbone_name="resnet50", pretrained_backbone=False, in_channels=1)
    x = torch.randn(2, 1, 128, 512)
    out = m(x)            # single forward
    p1, p2 = m(x, x)      # pair forward
    print("single out:", out.shape, "pair:", p1.shape, p2.shape)
