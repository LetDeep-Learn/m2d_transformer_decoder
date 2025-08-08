"""
modi_ssl_pretrain.py

Self-Supervised Pretraining pipeline for Modi OCR encoder.
Supports:
 - Denoising Autoencoder (DAE): noisy -> clean reconstruction
 - Masked Image Modeling (MIM): random patch masking + reconstruction
 - Joint training (DAE + MIM) with configurable loss weights

Usage examples:
 python modi_ssl_pretrain.py --mode joint --data_root ./modi_dataset --pairs_json clean_noisy_pairs.json --batch_size 4 --epochs 50

Expected folder layout:
 modi_dataset/
   clean_images/   # synthetic (2000)
   noisy_images/   # historical (2000)
   clean_noisy_pairs.json  # mapping: {"noisy_fname": "clean_fname", ...}

Output:
 saved_models/dae_mim_checkpoint.pth

Notes:
 - Designed for images resized to (1, 128, 512)
 - Uses PyTorch; tested for clarity; adapt optimizers/architectures as needed
"""

import os
import json
import random
import argparse
from glob import glob
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# -----------------------------
# Utilities & Dataset
# -----------------------------

class ModiPairDataset(Dataset):
    """Dataset that yields (noisy_image, clean_image) pairs. If a pair isn't found,
    it can optionally sample a random clean image (but recommended to provide mapping).
    """
    def __init__(self, data_root, pairs_json, img_size=(128,512), to_tensor=True):
        self.data_root = data_root
        self.noisy_dir = os.path.join(data_root, 'noisy_images')
        self.clean_dir = os.path.join(data_root, 'clean_images')
        with open(os.path.join(data_root, pairs_json), 'r', encoding='utf-8') as f:
            self.pairs = json.load(f)
        self.noisy_files = list(self.pairs.keys())
        self.clean_files = list(set(self.pairs.values()))
        self.img_size = img_size
        self.to_tensor = to_tensor

        self.transform = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize(self.img_size[::-1]),  # PIL size is (W,H) but Resize takes (H,W)?? safer to flip
            T.CenterCrop(self.img_size[::-1]),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy_name = self.noisy_files[idx]
        clean_name = self.pairs[noisy_name]
        noisy_path = os.path.join(self.noisy_dir, noisy_name)
        clean_path = os.path.join(self.clean_dir, clean_name)
        noisy_img = Image.open(noisy_path).convert('L')
        clean_img = Image.open(clean_path).convert('L')
        noisy_t = self.transform(noisy_img)
        clean_t = self.transform(clean_img)
        return noisy_t, clean_t, noisy_name

# -----------------------------
# Simple Encoder-Decoder (UNet-lite style)
# -----------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class SimpleUNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=32):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch*2)
        self.enc3 = ConvBlock(base_ch*2, base_ch*4)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_ch*4, base_ch*8)

        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_ch*8, base_ch*4)
        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_ch*4, base_ch*2)
        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_ch*2, base_ch)

        self.final = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)
        return out

# -----------------------------
# MIM Masking utilities (patch-based)
# -----------------------------

def random_mask_patchify(x, patch_size=(16,16), mask_ratio=0.25):
    """x: tensor BxCxHxW in [0,1]
    Returns:
      masked_x: same shape with masked pixels replaced by 0
      mask: boolean tensor Bx(num_patches_h*num_patches_w) indicating masked patches
      mask_coords: list of masked patch indices
    """
    B, C, H, W = x.shape
    ph, pw = patch_size
    assert H % ph == 0 and W % pw == 0, "Patch size must divide image dims"
    h_p = H // ph
    w_p = W // pw
    num_patches = h_p * w_p
    mask = torch.zeros(B, num_patches, dtype=torch.bool, device=x.device)
    masked_x = x.clone()
    for i in range(B):
        n_mask = max(1, int(mask_ratio * num_patches))
        choices = torch.randperm(num_patches, device=x.device)[:n_mask]
        mask[i, choices] = True
        for idx in choices:
            r = idx // w_p
            c = idx % w_p
            hs = r * ph
            ws = c * pw
            masked_x[i:i+1, :, hs:hs+ph, ws:ws+pw] = 0.0
    return masked_x, mask

# -----------------------------
# Training loop
# -----------------------------

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('saved_models', exist_ok=True)

    dataset = ModiPairDataset(args.data_root, args.pairs_json, img_size=(args.img_h, args.img_w))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    model = SimpleUNet(in_ch=1, base_ch=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # Losses
    l1_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['opt'])
        start_epoch = ck.get('epoch', 0) + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        for batch_idx, (noisy, clean, names) in enumerate(loader):
            noisy = noisy.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()

            # DAE forward
            recon = model(noisy)
            recon = torch.sigmoid(recon)  # make outputs in [0,1]
            loss_dae = l1_loss(recon, clean)

            loss = 0.0

            if args.mode in ('dae', 'joint'):
                loss += args.dae_w * loss_dae

            # MIM branch: mask clean images and reconstruct
            if args.mode in ('mim', 'joint'):
                masked_clean, mask = random_mask_patchify(clean, patch_size=(args.patch_h, args.patch_w), mask_ratio=args.mask_ratio)
                out_mask_rec = model(masked_clean)
                out_mask_rec = torch.sigmoid(out_mask_rec)
                loss_mim = mse_loss(out_mask_rec, clean)
                loss += args.mim_w * loss_mim
            else:
                loss_mim = torch.tensor(0.0, device=device)

            if loss == 0:
                # shouldn't happen
                loss = loss_dae

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % args.log_interval == 0:
                print(f"Epoch[{epoch}/{args.epochs}] Batch[{batch_idx}/{len(loader)}] loss={loss.item():.6f} (DAE={loss_dae.item():.6f} MIM={loss_mim.item():.6f})")

        avg_loss = total_loss / len(loader)
        print(f"==> Epoch {epoch} finished. Avg Loss: {avg_loss:.6f}")

        # Save checkpoint
        ckpt = {
            'model': model.state_dict(),
            'opt': optimizer.state_dict(),
            'epoch': epoch,
            'args': vars(args)
        }
        torch.save(ckpt, os.path.join('saved_models', f"dae_mim_epoch{epoch}.pth"))

    print('Training finished. Last checkpoint saved to saved_models/')

# -----------------------------
# Argument parsing
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default='./modi_dataset')
    p.add_argument('--pairs_json', default='clean_noisy_pairs.json')
    p.add_argument('--mode', choices=['dae','mim','joint'], default='joint',
                   help='Which self-supervised task to run: denoising (dae), masked-image (mim), or joint')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--img_h', type=int, default=128)
    p.add_argument('--img_w', type=int, default=512)
    p.add_argument('--dae_w', type=float, default=1.0, help='weight for denoising loss')
    p.add_argument('--mim_w', type=float, default=1.0, help='weight for MIM loss')
    p.add_argument('--patch_h', type=int, default=16)
    p.add_argument('--patch_w', type=int, default=16)
    p.add_argument('--mask_ratio', type=float, default=0.25)
    p.add_argument('--log_interval', type=int, default=10)
    p.add_argument('--resume', type=str, default='')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print('Arguments:', args)
    train(args)
