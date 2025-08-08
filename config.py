# config.py
import os
from torchvision import transforms as T

# -----------------------
# Paths
# -----------------------
# Google Drive path for saving models
DRIVE_SAVE_DIR = "/content/drive/MyDrive/m2d_transformer_decoder/saved_models_ssl"
EMMBEDINGS="/content/drive/MyDrive/m2d_transformer_decoder/emmbedings"

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
# Training hyperparams
# -----------------------
BATCH_SIZE = 8              # real batch per GPU step
ACCUM_STEPS = 8             # gradients accumulation -> effective batch = BATCH_SIZE * ACCUM_STEPS
EPOCHS = 100
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-6

CHECKPOINT_FILE = "ssl_checkpoint.pth"
LOG_EVERY = 50              # print logs every N batches (used if you add logging)
# -----------------------
# SSL / model params
# -----------------------
PROJ_DIM = 256              # projection head output dim (matches model default)
NTXENT_TEMPERATURE = 0.07

# Encoder related (kept so other modules can reference)
ENCODER_OUT_DIM = 512
BASE_CHANNELS = 64

# -----------------------
# Misc
# -----------------------
SEED = 42

# Create drive save directory early (no-op if exists)
os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)
IMAGE_SIZE = (128, 512)
NORMALIZE_MEAN = [0.5]
NORMALIZE_STD = [0.5]

# PIL-only augmentations (operate on PIL.Image)
AUGMENT_TRANSFORM = T.Compose([
    T.RandomApply([T.GaussianBlur(kernel_size=5)], p=0.5),
    T.RandomApply([T.ColorJitter(brightness=0.5, contrast=0.5)], p=0.5),
    T.RandomRotation(degrees=6),
    T.RandomAffine(degrees=5, translate=(0.02,0.02), scale=(0.95,1.05), shear=2),
    T.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0), ratio=(0.95,1.05)),
])

BASE_TRANSFORM = T.Compose([
    T.Resize(IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
])

# Tensor-only augmentations (after ToTensor)
TENSOR_AUGMENT = T.Compose([
    T.RandomErasing(p=0.25, scale=(0.01, 0.08), ratio=(0.3, 3.3)),
])