# dataset_ssl.py
import os
from PIL import Image
from torch.utils.data import Dataset
from config import BASE_TRANSFORM, AUGMENT_TRANSFORM
import random

class ModiSSLPretrainDataset(Dataset):
    """
    Dataset for SSL pretraining.
    Pairs noisy <-> synthetic images by filename basename (intersection).
    Applies AUGMENT_TRANSFORM independently to each view, then BASE_TRANSFORM to get tensor.
    """

    def __init__(self, noisy_dir, synthetic_dir, base_transform=BASE_TRANSFORM,
                 augment_transform=AUGMENT_TRANSFORM, debug=False):
        super().__init__()
        self.noisy_dir = noisy_dir
        self.synthetic_dir = synthetic_dir
        self.base_transform = base_transform
        self.augment_transform = augment_transform
        self.debug = debug

        noisy_files = sorted(os.listdir(noisy_dir))
        synth_files = sorted(os.listdir(synthetic_dir))

        # Build a mapping by basename (filename without extension)
        noisy_map = {os.path.splitext(f)[0]: f for f in noisy_files}
        synth_map = {os.path.splitext(f)[0]: f for f in synth_files}

        # intersect basenames to ensure correct pairing
        common = sorted(list(set(noisy_map.keys()) & set(synth_map.keys())))
        if len(common) == 0:
            raise RuntimeError(f"No matching filenames found between {noisy_dir} and {synthetic_dir}")

        # Build pair list (noisy_path, synth_path)
        self.pairs = [
            (os.path.join(noisy_dir, noisy_map[name]), os.path.join(synthetic_dir, synth_map[name]))
            for name in common
        ]

        if self.debug:
            print(f"[ModiSSLPretrainDataset] Found {len(self.pairs)} paired samples")
            print("First 5 pairs:", self.pairs[:5])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        noisy_path, synth_path = self.pairs[idx]

        noisy_img = Image.open(noisy_path).convert("L")      # PIL grayscale
        synth_img = Image.open(synth_path).convert("L")

        # Apply augmentations independently for each view (this creates two different views)
        # AUGMENT_TRANSFORM is designed for PIL ops (RandomRotation, ResizeCrop, etc.)
        if self.augment_transform is not None:
            noisy_aug = self.augment_transform(noisy_img)
            synth_aug = self.augment_transform(synth_img)
        else:
            noisy_aug = noisy_img
            synth_aug = synth_img

        # Then convert & resize to tensor (BASE_TRANSFORM contains Resize + ToTensor)
        noisy_tensor = self.base_transform(noisy_aug)
        synth_tensor = self.base_transform(synth_aug)

        return noisy_tensor, synth_tensor
