import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import DRIVE_SSL_SAVE_DIR, EPOCHS, BATCH_SIZE, LEARNING_RATE
from dataset_ssl import ModiSSLDataset
from model_ssl import SSLPretrainModel
from torchvision import transforms
import numpy as np
import random
import cv2

# --- Extra corruption functions ---
def add_gaussian_noise(img, mean=0.0, std=0.1):
    noise = torch.randn_like(img) * std + mean
    return torch.clamp(img + noise, 0.0, 1.0)

def random_blur(img, ksize=3):
    if random.random() < 0.3:  # 30% of the time
        img_np = img.squeeze(0).numpy()  # (H, W)
        img_np = cv2.GaussianBlur(img_np, (ksize, ksize), 0)
        img = torch.from_numpy(img_np).unsqueeze(0)
    return img

def mask_random_patches(img, mask_ratio=0.5):
    B, C, H, W = img.shape
    mask = torch.rand(B, 1, H, W, device=img.device) < mask_ratio
    img = img.clone()
    img[mask.expand_as(img)] = 0.0
    return img

# --- Dataset & DataLoader ---
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 512)),
    transforms.ToTensor()
])

dataset = ModiSSLDataset(transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Model, Loss, Optimizer ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SSLPretrainModel().to(device)
criterion = nn.L1Loss()  # MAE instead of MSE
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- Checkpoint Loading ---
start_epoch = 0
latest_ckpt = None
if os.path.exists(DRIVE_SSL_SAVE_DIR):
    ckpts = [f for f in os.listdir(DRIVE_SSL_SAVE_DIR) if f.endswith(".pth")]
    if ckpts:
        latest_ckpt = sorted(ckpts)[-1]
        ckpt_path = os.path.join(DRIVE_SSL_SAVE_DIR, latest_ckpt)
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming training from {ckpt_path}")
else:
    os.makedirs(DRIVE_SSL_SAVE_DIR, exist_ok=True)

if latest_ckpt is None:
    print("Starting fresh training...")

# --- Training Loop ---
for epoch in range(start_epoch, EPOCHS):
    model.train()
    epoch_loss = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for clean_imgs, noisy_imgs in pbar:
        clean_imgs = clean_imgs.to(device)
        noisy_imgs = noisy_imgs.to(device)

        # Add extra corruption to noisy images
        noisy_imgs = add_gaussian_noise(noisy_imgs)
        noisy_imgs = torch.stack([random_blur(img) for img in noisy_imgs])
        noisy_imgs = mask_random_patches(noisy_imgs, mask_ratio=0.5)

        optimizer.zero_grad()
        outputs = model(noisy_imgs)
        loss = criterion(outputs, clean_imgs)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {avg_loss:.4f}")

    # Save checkpoint
    ckpt_path = os.path.join(DRIVE_SSL_SAVE_DIR, f"ssl_epoch_{epoch+1}.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss
    }, ckpt_path)

print("SSL Pretraining complete.")
