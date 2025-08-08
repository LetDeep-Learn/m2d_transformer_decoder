import os
import json

clean_dir = "modi_dataset/clean_images"
noisy_dir = "modi_dataset/noisy_images"
output_json = "modi_dataset/clean_noisy_pairs.json"

# Get sorted image names
clean_images = sorted([f for f in os.listdir(clean_dir) if f.endswith(".png")])
noisy_images = sorted([f for f in os.listdir(noisy_dir) if f.endswith(".png")])

# Sanity check
assert len(clean_images) == len(noisy_images), "Mismatch in number of images!"

# Map clean → noisy
pairs = {}
for idx, (clean, noisy) in enumerate(zip(clean_images, noisy_images)):
    pairs[noisy] = clean
    print(f"🔗 {noisy} ↔ {clean}")

# Save JSON
with open(output_json, "w") as f:
    json.dump(pairs, f, indent=2)

print(f"\n✅ Done! {len(pairs)} clean-noisy pairs saved to {output_json}")
