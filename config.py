# config.py
import os
from torchvision import transforms as T

# -----------------------
# Paths
# -----------------------
# Google Drive path for saving models
DRIVE_SAVE_DIR = "/content/drive/MyDrive/m2d_transformer_decoder/saved_models_ssl"

# Dataset paths (relative to cloned repo in Colab)
DATASET_DIR = "./modi_dataset"
NOISY_IMAGES_DIR = os.path.join(DATASET_DIR, "noisy_images")
SYNTHETIC_IMAGES_DIR = os.path.join(DATASET_DIR, "clean_images")

# optional checkpoint path you used earlier (kept for compatibility)
FREEZE_CHECKPOINT = "/content/drive/MyDrive/m2d_transformer_decoder/saved_models_ssl/ssl_checkpoint.pth"

# -----------------------
# Device / runtime
# -----------------------
DEVICE = "cuda"  # train script checks torch.cuda.is_available()
NUM_WORKERS = 4
PIN_MEMORY = True

# -----------------------
# Image & transform params
# -----------------------
# IMG_HEIGHT = 128
# IMG_WIDTH = 512
# IMAGE_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# # For grayscale images use single-channel normalization.
# # If you later switch to RGB or add CoordConv channels, update mean/std accordingly.
# NORMALIZE_MEAN = [0.5]
# NORMALIZE_STD = [0.5]

# # Base transform: resize -> to tensor -> normalize
# BASE_TRANSFORM = T.Compose([
#     T.Resize(IMAGE_SIZE),
#     T.ToTensor(),
#     T.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
# ])

# Augmentations applied *before* base transform (PIL ops)
# AUGMENT_TRANSFORM = T.Compose([
#     # strong-ish augmentations but don't completely destroy text
#     T.RandomApply([T.GaussianBlur(kernel_size=5)], p=0.6),
#     T.RandomApply([T.ColorJitter(brightness=0.6, contrast=0.6)], p=0.6),
#     T.RandomRotation(degrees=10),
#     T.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
#     T.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3))
# ])

# If you want lighter augmentation for debugging, set this to None at runtime
# AUGMENT_TRANSFORM = None

# -----------------------
# Training hyperparams
# -----------------------
BATCH_SIZE = 4              # real batch per GPU step
ACCUM_STEPS = 8             # gradients accumulation -> effective batch = BATCH_SIZE * ACCUM_STEPS
EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-6

CHECKPOINT_FILE = "ssl_checkpoint.pth"
LOG_EVERY = 50              # print logs every N batches (used if you add logging)

# -----------------------
# SSL / model params
# -----------------------
PROJ_DIM = 256              # projection head output dim (matches model default)
NTXENT_TEMPERATURE = 0.1

# Encoder related (kept so other modules can reference)
ENCODER_OUT_DIM = 512
BASE_CHANNELS = 64

# -----------------------
# Misc
# -----------------------
SEED = 42

# Create drive save directory early (no-op if exists)
os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)
IMG_HEIGHT = 128
IMG_WIDTH = 512
IMAGE_SIZE = (IMG_HEIGHT, IMG_WIDTH)

NORMALIZE_MEAN = [0.5]
NORMALIZE_STD = [0.5]

# PIL-only augmentations (must run on PIL.Image)
AUGMENT_TRANSFORM = T.Compose([
    T.RandomApply([T.GaussianBlur(kernel_size=5)], p=0.6),
    T.RandomApply([T.ColorJitter(brightness=0.6, contrast=0.6)], p=0.6),
    T.RandomRotation(degrees=10),
    T.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
    # DO NOT include RandomErasing here — it expects tensors
])

# Base transform: resize -> to tensor -> normalize
BASE_TRANSFORM = T.Compose([
    T.Resize(IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
])

# Tensor-only augmentations (applied after BASE_TRANSFORM)
TENSOR_AUGMENT = T.Compose([
    T.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3))
])

# Worker recommendation
NUM_WORKERS = 2
PIN_MEMORY = True
# -------