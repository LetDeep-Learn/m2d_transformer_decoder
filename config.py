# config.py

import os

# Google Drive path for saving models
DRIVE_SAVE_DIR = "/content/drive/MyDrive/m2d_transformer_decoder/saved_models_ssl"

# Dataset paths (relative to cloned repo in Colab)
DATASET_DIR = "./modi_dataset"
NOISY_IMAGES_DIR = os.path.join(DATASET_DIR, "noisy_images")
SYNTHETIC_IMAGES_DIR = os.path.join(DATASET_DIR, "clean_images")

# Training constants
BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 1e-4
IMG_HEIGHT = 128
IMG_WIDTH = 512

# Checkpoint file name
CHECKPOINT_FILE = "ssl_checkpoint.pth"
