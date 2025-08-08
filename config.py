# config.py

import os

# Google Drive path for saving models
DRIVE_SAVE_DIR = "/content/drive/MyDrive/m2d_transformer_decoder/saved_models_ssl"

# Dataset paths (relative to cloned repo in Colab)
DATASET_DIR = "./modi_dataset"
NOISY_IMAGES_DIR = os.path.join(DATASET_DIR, "noisy_images")
SYNTHETIC_IMAGES_DIR = os.path.join(DATASET_DIR, "clean_images")
DEVICE="cuda"
# Training constants
BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 1e-4
IMG_HEIGHT = 128
IMG_WIDTH = 512

# Checkpoint file name
CHECKPOINT_FILE = "ssl_checkpoint.pth"


from torchvision import transforms as T

IMAGE_SIZE = (IMG_HEIGHT, IMG_WIDTH)

BASE_TRANSFORM = T.Compose([
    T.Resize(IMAGE_SIZE),
    T.ToTensor()
])

# AUGMENT_TRANSFORM = T.Compose([
#     T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.5),
#     T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4)], p=0.5),
#     T.RandomRotation(2),
#     T.RandomResizedCrop(IMAGE_SIZE, scale=(0.9, 1.0))
# ])

AUGMENT_TRANSFORM = T.Compose([
    T.RandomApply([T.GaussianBlur(kernel_size=5)], p=0.7),
    T.RandomApply([T.ColorJitter(brightness=0.6, contrast=0.6)], p=0.7),
    T.RandomRotation(degrees=10),
    T.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0))
])

