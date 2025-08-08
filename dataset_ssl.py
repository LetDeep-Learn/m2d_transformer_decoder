# # dataset_ssl.py
# import os
# from PIL import Image
# from torch.utils.data import Dataset
# from config import BASE_TRANSFORM, AUGMENT_TRANSFORM
# import random

# class ModiSSLPretrainDataset(Dataset):
#     """
#     Dataset for SSL pretraining.
#     Pairs noisy <-> synthetic images by filename basename (intersection).
#     Applies AUGMENT_TRANSFORM independently to each view, then BASE_TRANSFORM to get tensor.
#     """

#     def __init__(self, noisy_dir, synthetic_dir, base_transform=BASE_TRANSFORM,
#                  augment_transform=AUGMENT_TRANSFORM, debug=False):
#         super().__init__()
#         self.noisy_dir = noisy_dir
#         self.synthetic_dir = synthetic_dir
#         self.base_transform = base_transform
#         self.augment_transform = augment_transform
#         self.debug = debug

#         noisy_files = sorted(os.listdir(noisy_dir))
#         synth_files = sorted(os.listdir(synthetic_dir))

#         # Build a mapping by basename (filename without extension)
#         noisy_map = {os.path.splitext(f)[0]: f for f in noisy_files}
#         synth_map = {os.path.splitext(f)[0]: f for f in synth_files}

#         # intersect basenames to ensure correct pairing
#         common = sorted(list(set(noisy_map.keys()) & set(synth_map.keys())))
#         if len(common) == 0:
#             raise RuntimeError(f"No matching filenames found between {noisy_dir} and {synthetic_dir}")

#         # Build pair list (noisy_path, synth_path)
#         self.pairs = [
#             (os.path.join(noisy_dir, noisy_map[name]), os.path.join(synthetic_dir, synth_map[name]))
#             for name in common
#         ]

#         if self.debug:
#             print(f"[ModiSSLPretrainDataset] Found {len(self.pairs)} paired samples")
#             print("First 5 pairs:", self.pairs[:5])

#     def __len__(self):
#         return len(self.pairs)

#     def __getitem__(self, idx):
#         noisy_path, synth_path = self.pairs[idx]

#         noisy_img = Image.open(noisy_path).convert("L")      # PIL grayscale
#         synth_img = Image.open(synth_path).convert("L")

#         # Apply augmentations independently for each view (this creates two different views)
#         # AUGMENT_TRANSFORM is designed for PIL ops (RandomRotation, ResizeCrop, etc.)
#         if self.augment_transform is not None:
#             noisy_aug = self.augment_transform(noisy_img)
#             synth_aug = self.augment_transform(synth_img)
#         else:
#             noisy_aug = noisy_img
#             synth_aug = synth_img

#         # Then convert & resize to tensor (BASE_TRANSFORM contains Resize + ToTensor)
#         noisy_tensor = self.base_transform(noisy_aug)
#         synth_tensor = self.base_transform(synth_aug)

#         return noisy_tensor, synth_tensor


# dataset_ssl.py (numeric-ID pairing + safe fallbacks)
import os
import re
from PIL import Image
from torch.utils.data import Dataset
from config import BASE_TRANSFORM, AUGMENT_TRANSFORM

DIGIT_RE = re.compile(r"(\d+)")

class ModiSSLPretrainDataset(Dataset):
    """
    Pair noisy <-> synthetic images by numeric ID inside filename when possible.
    Fallbacks:
      1) basename intersection
      2) index-based pairing (sorted lists) with warning
    """
    def __init__(self, noisy_dir, synthetic_dir, base_transform=BASE_TRANSFORM,
                 augment_transform=AUGMENT_TRANSFORM, debug=False):
        super().__init__()
        self.noisy_dir = noisy_dir
        self.synthetic_dir = synthetic_dir
        self.base_transform = base_transform
        self.augment_transform = augment_transform
        self.debug = debug

        noisy_files = [f for f in sorted(os.listdir(noisy_dir)) if not f.startswith('.')]
        synth_files = [f for f in sorted(os.listdir(synthetic_dir)) if not f.startswith('.')]

        # helper to extract numeric id as string (preserve leading zeros)
        def extract_id(fname):
            m = DIGIT_RE.search(fname)
            return m.group(1) if m else None

        # build maps by numeric id
        noisy_by_id = {}
        for f in noisy_files:
            fid = extract_id(f)
            if fid is not None:
                noisy_by_id.setdefault(fid, []).append(f)
        synth_by_id = {}
        for f in synth_files:
            fid = extract_id(f)
            if fid is not None:
                synth_by_id.setdefault(fid, []).append(f)

        # If both maps have numeric ids, try to pair by id
        common_ids = sorted(list(set(noisy_by_id.keys()) & set(synth_by_id.keys())))
        pairs = []
        if len(common_ids) > 0:
            if self.debug:
                print(f"[Dataset] Pairing by numeric ID: found {len(common_ids)} common IDs.")
            for cid in common_ids:
                # if multiple files share same id, pair by sorted order (rare)
                nf = sorted(noisy_by_id[cid])[0]
                sf = sorted(synth_by_id[cid])[0]
                pairs.append((os.path.join(noisy_dir, nf), os.path.join(synthetic_dir, sf)))

        else:
            # Numeric ids didn't match — try basename matching (without extension)
            noisy_map = {os.path.splitext(f)[0]: f for f in noisy_files}
            synth_map = {os.path.splitext(f)[0]: f for f in synth_files}
            common_base = sorted(list(set(noisy_map.keys()) & set(synth_map.keys())))
            if len(common_base) > 0:
                if self.debug:
                    print(f"[Dataset] Pairing by basename intersection: {len(common_base)} pairs.")
                pairs = [(os.path.join(noisy_dir, noisy_map[b]), os.path.join(synthetic_dir, synth_map[b])) for b in common_base]
            else:
                # Last resort: index-based pairing (preserve order but warn)
                if self.debug:
                    print("[Dataset WARNING] No numeric ID or basename matches. Falling back to index-based pairing.")
                n = min(len(noisy_files), len(synth_files))
                pairs = [(os.path.join(noisy_dir, noisy_files[i]), os.path.join(synthetic_dir, synth_files[i])) for i in range(n)]

        if len(pairs) == 0:
            raise RuntimeError(f"No paired samples could be constructed from {noisy_dir} and {synthetic_dir}.")

        self.pairs = pairs

        if self.debug:
            print(f"[Dataset] final pair count: {len(self.pairs)}")
            print("First 10 pairs (noisy <---> synth):")
            for i in range(min(10, len(self.pairs))):
                print(" ", os.path.basename(self.pairs[i][0]), "<-->", os.path.basename(self.pairs[i][1]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        noisy_path, synth_path = self.pairs[idx]

        noisy_img = Image.open(noisy_path).convert("L")
        synth_img = Image.open(synth_path).convert("L")

        # independent augmentations per view
        if self.augment_transform is not None:
            noisy_aug = self.augment_transform(noisy_img)
            synth_aug = self.augment_transform(synth_img)
        else:
            noisy_aug = noisy_img
            synth_aug = synth_img

        noisy_tensor = self.base_transform(noisy_aug)
        synth_tensor = self.base_transform(synth_aug)

        return noisy_tensor, synth_tensor
