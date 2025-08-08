# dataset_ssl.py

import os
from PIL import Image
from torch.utils.data import Dataset

class ModiSSLPretrainDataset(Dataset):
    def __init__(self, noisy_dir, synthetic_dir, transform=None):
        self.noisy_dir = noisy_dir
        self.synthetic_dir = synthetic_dir
        self.noisy_images = sorted(os.listdir(noisy_dir))
        self.synthetic_images = sorted(os.listdir(synthetic_dir))
        self.transform = transform

    def __len__(self):
        return min(len(self.noisy_images), len(self.synthetic_images))

    def __getitem__(self, idx):
        noisy_path = os.path.join(self.noisy_dir, self.noisy_images[idx])
        synthetic_path = os.path.join(self.synthetic_dir, self.synthetic_images[idx])

        noisy_img = Image.open(noisy_path).convert("L")
        synthetic_img = Image.open(synthetic_path).convert("L")

        if self.transform:
            noisy_img = self.transform(noisy_img)
            synthetic_img = self.transform(synthetic_img)

        return noisy_img, synthetic_img
