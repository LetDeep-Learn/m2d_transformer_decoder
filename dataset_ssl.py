# dataset_ssl.py

import os
import random
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
import torchvision.transforms as T
from config import BASE_TRANSFORM, AUGMENT_TRANSFORM

class ModiSSLPretrainDataset(Dataset):
    def __init__(self, noisy_dir, synthetic_dir):
        self.noisy_dir = noisy_dir
        self.synthetic_dir = synthetic_dir
        self.noisy_images = sorted(os.listdir(noisy_dir))
        self.synthetic_images = sorted(os.listdir(synthetic_dir))

    def __len__(self):
        return min(len(self.noisy_images), len(self.synthetic_images))

    def __getitem__(self, idx):
        noisy_path = os.path.join(self.noisy_dir, self.noisy_images[idx])
        synthetic_path = os.path.join(self.synthetic_dir, self.synthetic_images[idx])

        noisy_img = Image.open(noisy_path).convert("L")
        synthetic_img = Image.open(synthetic_path).convert("L")

        # Apply augmentations first (on PIL)
        noisy_aug = AUGMENT_TRANSFORM(noisy_img)
        synthetic_aug = AUGMENT_TRANSFORM(synthetic_img)

        # Then convert to tensor and resize
        noisy_tensor = BASE_TRANSFORM(noisy_aug)
        synthetic_tensor = BASE_TRANSFORM(synthetic_aug)

        return noisy_tensor, synthetic_tensor

    # Legacy version (kept for reference)
    # def __getitem__(self, idx):
    #     noisy_path = os.path.join(self.noisy_dir, self.noisy_images[idx])
    #     synthetic_path = os.path.join(self.synthetic_dir, self.synthetic_images[idx])

    #     noisy_img = Image.open(noisy_path).convert("L")
    #     synthetic_img = Image.open(synthetic_path).convert("L")

    #     # Apply augmentations differently for each view
    #     noisy_img = self.base_transform(noisy_img)
    #     synthetic_img = self.base_transform(synthetic_img)

    #     # Extra augmentations for difficulty
    #     noisy_aug = self.augment_transform(noisy_img)
    #     synthetic_aug = self.augment_transform(synthetic_img)

    #     return noisy_aug, synthetic_aug
