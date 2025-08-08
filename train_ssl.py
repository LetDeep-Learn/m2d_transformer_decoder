# # train_ssl.py

# import os
# import torch
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from config import *
# from dataset_ssl import ModiSSLPretrainDataset
# from model_ssl import SSLModel

# def train_ssl():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Create save directory in Google Drive
#     os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)
#     checkpoint_path = os.path.join(DRIVE_SAVE_DIR, CHECKPOINT_FILE)

#     # Dataset + Transform
#     transform = transforms.Compose([
#         transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
#         transforms.ToTensor()
#     ])

#     dataset = ModiSSLPretrainDataset(NOISY_IMAGES_DIR, SYNTHETIC_IMAGES_DIR, transform)
#     dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

#     # Model, Loss, Optimizer
#     model = SSLModel().to(device)
#     criterion = torch.nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
#     start_epoch = 0

#     # Resume if checkpoint exists
#     if os.path.exists(checkpoint_path):
#         checkpoint = torch.load(checkpoint_path, map_location=device)
#         model.load_state_dict(checkpoint["model"])
#         optimizer.load_state_dict(checkpoint["optimizer"])
#         start_epoch = checkpoint["epoch"] + 1
#         print(f"Resuming training from {checkpoint_path} at epoch {start_epoch}")
#     else:
#         print("Starting fresh training")

#     # Training Loop
#     for epoch in range(start_epoch, EPOCHS):
#         model.train()
#         epoch_loss = 0

#         for noisy_img, synthetic_img in dataloader:
#             noisy_img, synthetic_img = noisy_img.to(device), synthetic_img.to(device)
#             optimizer.zero_grad()
#             output = model(noisy_img)
#             target = model(synthetic_img)
#             loss = criterion(output, target)
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item()

#         print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {epoch_loss/len(dataloader):.4f}")

#         # Save checkpoint
#         torch.save({
#             "epoch": epoch,
#             "model": model.state_dict(),
#             "optimizer": optimizer.state_dict()
#         }, checkpoint_path)

#     print("Training complete.")

# if __name__ == "__main__":
#     train_ssl()











# train_ssl.py
# train_ssl.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_ssl import ModiSSLPretrainDataset
from model_ssl import ModiSSLModel
# from config import DRIVE_SAVE_DIR, CHECKPOINT_FILE, NOISY_IMAGES_DIR, SYNTHETIC_IMAGES_DIR, BATCH_SIZE, EPOCHS, DEVICE,LEARNING_RATE

from config import (
    DRIVE_SAVE_DIR,
    NOISY_IMAGES_DIR,
    SYNTHETIC_IMAGES_DIR,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    CHECKPOINT_FILE
)
def train_ssl():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    # Dataset & DataLoader
    dataset = ModiSSLPretrainDataset(
        noisy_dir=NOISY_IMAGES_DIR,
        synthetic_dir=SYNTHETIC_IMAGES_DIR
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # Model, Loss, Optimizer
    model = ModiSSLModel().to(device)
    criterion = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    checkpoint_path = os.path.join(DRIVE_SAVE_DIR, CHECKPOINT_FILE)
    start_epoch = 0

    # Resume training logic with backward compatibility
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt.get("epoch", -1) + 1
            print(f"Resumed training from epoch {start_epoch}")
        else:
            model.load_state_dict(ckpt)
            print("Loaded old format checkpoint (model only). Starting from epoch 0.")
            start_epoch = 0
    else:
        print("No checkpoint found. Starting fresh training...")

    # Training loop
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0.0

        for noisy_img, synthetic_img in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            noisy_img, synthetic_img = noisy_img.to(device), synthetic_img.to(device)

            noisy_feat = model(noisy_img)
            synthetic_feat = model(synthetic_img)

            target = torch.ones(noisy_feat.size(0)).to(device)  # Positive pairs
            loss = criterion(noisy_feat, synthetic_feat, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f}")

        # Save checkpoint in new format
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch
        }, checkpoint_path)


if __name__ == "__main__":
    train_ssl()
