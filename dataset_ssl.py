# dataset_ssl.py

import os
import random
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
import torchvision.transforms as T

class ModiSSLPretrainDataset(Dataset):
    def __init__(self, noisy_dir, synthetic_dir, image_size=(128, 512)):
        self.noisy_dir = noisy_dir
        self.synthetic_dir = synthetic_dir
        self.noisy_images = sorted(os.listdir(noisy_dir))
        self.synthetic_images = sorted(os.listdir(synthetic_dir))

        # Base resize + tensor transform
        self.base_transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor()
        ])

        # Augmentations for SSL
        self.augment_transform = T.Compose([
            T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.5),
            T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4)], p=0.5),
            T.RandomRotation(2),
            T.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
            T.ToTensor()
        ])

    def __len__(self):
        return min(len(self.noisy_images), len(self.synthetic_images))

    def __getitem__(self, idx):
        noisy_path = os.path.join(self.noisy_dir, self.noisy_images[idx])
        synthetic_path = os.path.join(self.synthetic_dir, self.synthetic_images[idx])

        noisy_img = Image.open(noisy_path).convert("L")
        synthetic_img = Image.open(synthetic_path).convert("L")

        # Apply augmentations differently for each view
        noisy_img = self.base_transform(noisy_img)
        synthetic_img = self.base_transform(synthetic_img)

        # Extra augmentations for difficulty
        noisy_aug = self.augment_transform(noisy_img)
        synthetic_aug = self.augment_transform(synthetic_img)

        return noisy_aug, synthetic_aug
