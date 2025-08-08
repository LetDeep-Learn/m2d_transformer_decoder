import os
import json
import multiprocessing
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
# ========== ✅ Detect Colab ==========
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# # ========== ✅ Mount Google Drive if in Colab ==========
if IN_COLAB:
    from google.colab import drive
    # drive.mount('/content/drive', force_remount=True)
    BASE_DIR = "/content/drive/MyDrive/modi2marathi"
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
######################## for git hub and google drive ########################3

# REPO_ROOT = "/content/m2d/modi2marathi"  # <-- change if your repo name is different

# # ==== DATASET PATHS ====
# IMAGE_DIR = os.path.join(REPO_ROOT, "mixed/images")
# LABEL_DIR = os.path.join(REPO_ROOT, "mixed/labels")
# CHAR_TO_IDX_PATH = os.path.join(REPO_ROOT, "mixed/char_to_idx.json")
# IDX_TO_CHAR_PATH = os.path.join(REPO_ROOT, "mixed/idx_to_char.json")

# ==== CHECKPOINT PATH ====
# Saved inside Google Drive
# CHECKPOINT_DIR = "/content/drive/MyDrive/ModiOCRCheckpoints"
# BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")








# ========== ✅ Define Dataset and Checkpoint Paths ==========
##################################################################################################
#########  for Synthetic images
#######################CURRENT DATASET: SYNTHETIC (SynthMoDe)####################################
# IMAGE_DIR = os.path.join(BASE_DIR, "modi_dataset_syn/images")
# LABEL_DIR = os.path.join(BASE_DIR, "modi_dataset_syn/labels")
# CHAR_TO_IDX_PATH = os.path.join(BASE_DIR, "modi_dataset_syn/char_to_idx.json")
# IDX_TO_CHAR_PATH = os.path.join(BASE_DIR, "modi_dataset_syn/idx_to_char.json")


###################   Mixed Dataset  ################################

IMAGE_DIR = os.path.join(BASE_DIR, "mixed/images")
LABEL_DIR = os.path.join(BASE_DIR, "mixed/labels")
CHAR_TO_IDX_PATH = os.path.join(BASE_DIR, "mixed/char_to_idx.json")
IDX_TO_CHAR_PATH = os.path.join(BASE_DIR, "mixed/idx_to_char.json")



###################   ORG Dataset  ################################

# IMAGE_DIR = os.path.join(BASE_DIR, "modi_dataset/images")
# LABEL_DIR = os.path.join(BASE_DIR, "modi_dataset/labels")
# CHAR_TO_IDX_PATH = os.path.join(BASE_DIR, "modi_dataset/char_to_idx.json") <- not in use old versions
# IDX_TO_CHAR_PATH = os.path.join(BASE_DIR, "modi_dataset/idx_to_char.json") <- not in use old versions
###############################################################################3
CHECKPOINT_DIR = os.path.join(BASE_DIR, "saved_models")
LATEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "ocr_model_latest.pt")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def get_checkpoint_path(epoch):
    return os.path.join(CHECKPOINT_DIR, f"ocr_model_epoch_{epoch}.pt")

# ========== ✅ Training Hyperparameters ==========
BATCH_SIZE = 4
EPOCHS = 25 # 45 WILL BE SEMI FINAL
LEARNING_RATE = 1e-4 # sset to 1e-4 if not work or loss function suddern JUMP
IMG_HEIGHT =128 #128 is best but
IMG_WIDTH =720 #1268 is best but 950
MAX_LABEL_LENGTH=452
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = multiprocessing.cpu_count()

# ========== ✅ Load Vocabulary ==========
with open(CHAR_TO_IDX_PATH, "r", encoding="utf-8") as f:
    CHAR_TO_IDX = json.load(f)

with open(IDX_TO_CHAR_PATH, "r", encoding="utf-8") as f:
    IDX_TO_CHAR = {int(k): v for k, v in json.load(f).items()}

NUM_CLASSES = len(CHAR_TO_IDX)  # No +1; blank=0 already handled by CTC setup




if __name__ == "__main__":
    print("Running in Colab:", IN_COLAB)
    print("BASE_DIR:", BASE_DIR)
    # print("REPO_ROOT:", REPO_ROOT)
    print("IMAGE_DIR:", IMAGE_DIR)
    print("LABEL_DIR:", LABEL_DIR)
    print("CHECKPOINT_DIR:", CHECKPOINT_DIR)



import os
import json
import multiprocessing
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

# ========== ✅ Detect Colab ==========
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# ========== ✅ Mount Google Drive if in Colab ==========
if IN_COLAB:
    from google.colab import drive
    # drive.mount('/content/drive', force_remount=True)
    BASE_DIR = "/content/drive/MyDrive/modi2marathi"
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))



# ========== ✅ Mount Google Drive & Clone GitHub Repo ==========
# if IN_COLAB:
#     from google.colab import drive
#     # drive.mount('/content/drive', force_remount=True)

#     # 📌 Change this to your actual repo URL
#     GITHUB_REPO_URL = "https://github.com/yourusername/modi2marathi.git"
#     REPO_NAME = "modi2marathi"
#     REPO_ROOT = f"/content/{REPO_NAME}"

#     # Clone only if not already cloned
#     if not os.path.exists(REPO_ROOT):
#         !git clone {GITHUB_REPO_URL} {REPO_ROOT}

#     # Data inside cloned repo
#     BASE_DIR = REPO_ROOT  

#     # Checkpoints in Google Drive
#     DRIVE_BASE = "/content/drive/MyDrive/modi2marathi"
# else:
#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     DRIVE_BASE = BASE_DIR
    
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DRIVE_BASE = "/content/drive/MyDrive/modi2marathi"
######################## for git hub and google drive ########################

# REPO_ROOT = "/content/m2d/modi2marathi"  # <-- change if your repo name is different

# # ==== DATASET PATHS ====
# IMAGE_DIR = os.path.join(REPO_ROOT, "mixed/images")
# LABEL_DIR = os.path.join(REPO_ROOT, "mixed/labels")
# CHAR_TO_IDX_PATH = os.path.join(REPO_ROOT, "mixed/char_to_idx.json")
# IDX_TO_CHAR_PATH = os.path.join(REPO_ROOT, "mixed/idx_to_char.json")

# ==== CHECKPOINT PATH ====
# Saved inside Google Drive
# CHECKPOINT_DIR = "/content/drive/MyDrive/ModiOCRCheckpoints"
# BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")

# ========== ✅ Define Dataset and Checkpoint Paths ==========
#######################CURRENT DATASET: SYNTHETIC (SynthMoDe)####################################
# IMAGE_DIR = os.path.join(BASE_DIR, "modi_dataset_syn/images")
# LABEL_DIR = os.path.join(BASE_DIR, "modi_dataset_syn/labels")
# CHAR_TO_IDX_PATH = os.path.join(BASE_DIR, "modi_dataset_syn/char_to_idx.json")
# IDX_TO_CHAR_PATH = os.path.join(BASE_DIR, "modi_dataset_syn/idx_to_char.json")

###################   Mixed Dataset  ################################
# IMAGE_DIR = os.path.join(BASE_DIR, "mixed/images")
# LABEL_DIR = os.path.join(BASE_DIR, "mixed/labels")
CHAR_TO_IDX_PATH = os.path.join(BASE_DIR, "mixed/char_to_idx.json")
IDX_TO_CHAR_PATH = os.path.join(BASE_DIR, "mixed/idx_to_char.json")

###################   ORG Dataset  ################################
IMAGE_DIR = os.path.join(BASE_DIR, "modi_dataset/images")
LABEL_DIR = os.path.join(BASE_DIR, "modi_dataset/labels")
# CHAR_TO_IDX_PATH = os.path.join(BASE_DIR, "modi_dataset/char_to_idx.json")
# IDX_TO_CHAR_PATH = os.path.join(BASE_DIR, "modi_dataset/idx_to_char.json")

###############################################################################
# ✅ Save checkpoints to Google Drive folder
CHECKPOINT_DIR = os.path.join(DRIVE_BASE, "saved_models")
LATEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "ocr_model_latest.pt")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def get_checkpoint_path(epoch):
    return os.path.join(CHECKPOINT_DIR, f"ocr_model_epoch_{epoch}.pt")

# ========== ✅ Training Hyperparameters ==========
BATCH_SIZE = 4
EPOCHS = 25  # 45 WILL BE SEMI FINAL
LEARNING_RATE = 1e-4
IMG_HEIGHT = 128
IMG_WIDTH = 720
MAX_LABEL_LENGTH = 452
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = multiprocessing.cpu_count()

# ========== ✅ Load Vocabulary ==========
with open(CHAR_TO_IDX_PATH, "r", encoding="utf-8") as f:
    CHAR_TO_IDX = json.load(f)

with open(IDX_TO_CHAR_PATH, "r", encoding="utf-8") as f:
    IDX_TO_CHAR = {int(k): v for k, v in json.load(f).items()}

NUM_CLASSES = len(CHAR_TO_IDX)

# ========== ✅ Special Tokens ==========
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"

PAD_TOKEN_IDX = CHAR_TO_IDX[PAD_TOKEN]
SOS_TOKEN_IDX = CHAR_TO_IDX[SOS_TOKEN]
EOS_TOKEN_IDX = CHAR_TO_IDX[EOS_TOKEN]

# ========== ✅ Debug Info ==========
if __name__ == "__main__":
    print("Running in Colab:", IN_COLAB)
    print("BASE_DIR:", BASE_DIR)
    print("IMAGE_DIR:", IMAGE_DIR)
    print("LABEL_DIR:", LABEL_DIR)
    print("CHECKPOINT_DIR:", CHECKPOINT_DIR)
    print("NUM_CLASSES:", NUM_CLASSES)
    print("PAD_TOKEN_IDX:", PAD_TOKEN_IDX)
    print("SOS_TOKEN_IDX:", SOS_TOKEN_IDX)
    print("EOS_TOKEN_IDX:", EOS_TOKEN_IDX)
