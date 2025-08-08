import os
from itertools import islice

NOISY = "./modi_dataset/noisy_images"
CLEAN = "./modi_dataset/clean_images"

noisy_files = sorted([f for f in os.listdir(NOISY) if not f.startswith('.')])
clean_files = sorted([f for f in os.listdir(CLEAN) if not f.startswith('.')])

print("Counts -> noisy:", len(noisy_files), "clean:", len(clean_files))
print("\nFirst 10 noisy:", list(islice(noisy_files, 10)))
print("First 10 clean:", list(islice(clean_files, 10)))

# basenames without extension
noisy_base = {os.path.splitext(f)[0] for f in noisy_files}
clean_base = {os.path.splitext(f)[0] for f in clean_files}

common = noisy_base & clean_base
only_noisy = noisy_base - clean_base
only_clean = clean_base - noisy_base

print(f"\nCommon basenames: {len(common)}")
print(f"Only in noisy: {len(only_noisy)} (example 10): {list(islice(only_noisy, 10))}")
print(f"Only in clean: {len(only_clean)} (example 10): {list(islice(only_clean, 10))}")

# Show a few exact unmatched filepairs by index
print("\nSample top-of-folder file pairs by index (may be wrong if filenames don't align):")
for i in range(min(10, len(noisy_files), len(clean_files))):
    print(i, noisy_files[i], "<-->", clean_files[i])
