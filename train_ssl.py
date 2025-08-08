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
from torch.utils.data import DataLoader
from config import (
    DRIVE_SAVE_DIR,
    NOISY_IMAGES_DIR,
    SYNTHETIC_IMAGES_DIR,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    CHECKPOINT_FILE
)
from dataset_ssl import ModiSSLPretrainDataset
from model_ssl import ModiSSLModel, ssl_loss


def train_ssl():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)
    checkpoint_path = os.path.join(DRIVE_SAVE_DIR, CHECKPOINT_FILE)

    # Dataset — augmentations already inside ModiSSLPretrainDataset
    dataset = ModiSSLPretrainDataset(
        noisy_dir=NOISY_IMAGES_DIR,
        synthetic_dir=SYNTHETIC_IMAGES_DIR
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # Model + Optimizer
    model = ModiSSLModel(proj_dim=256, dropout=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Resume checkpoint if exists
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0.0

        for step, (noisy_imgs, synthetic_imgs) in enumerate(dataloader):
            noisy_imgs, synthetic_imgs = noisy_imgs.to(device), synthetic_imgs.to(device)

            optimizer.zero_grad()
            z1, z2 = model(noisy_imgs, synthetic_imgs)
            loss = ssl_loss(z1, z2, alpha=0.5)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (step + 1) % 20 == 0:  # print every 20 batches
                print(f"Epoch {epoch+1} | Step {step+1}/{len(dataloader)} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Avg Loss: {avg_loss:.4f}")

        # Save checkpoint
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }, checkpoint_path)

    print("Training complete!")


if __name__ == "__main__":
    train_ssl()
