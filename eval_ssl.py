# eval_ssl.py

import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from config import (
    DRIVE_SAVE_DIR,
    CHECKPOINT_FILE,
    NOISY_IMAGES_DIR,
    SYNTHETIC_IMAGES_DIR,
    BATCH_SIZE,
    DEVICE
)
from dataset_ssl import ModiSSLPretrainDataset
from model_ssl import ModiSSLModel

def visualize_images(noisy_tensor, synthetic_tensor, similarity, z1, z2):
    noisy_img = noisy_tensor.squeeze().cpu().numpy()
    synthetic_img = synthetic_tensor.squeeze().cpu().numpy()

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(noisy_img, cmap='gray')
    axs[0].set_title("Noisy Image")
    axs[1].imshow(synthetic_img, cmap='gray')
    axs[1].set_title("Synthetic Image")
    plt.suptitle(f"Cosine Similarity: {similarity:.4f}")
    plt.show()

    # Print first few dimensions of embeddings
    print("z1 (noisy) →", z1[0][:5].cpu().numpy())
    print("z2 (synthetic) →", z2[0][:5].cpu().numpy())
    print("-" * 50)

def evaluate_ssl():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    # Load model
    model = ModiSSLModel().to(device)
    checkpoint_path = os.path.join(DRIVE_SAVE_DIR, CHECKPOINT_FILE)
    if not os.path.exists(checkpoint_path):
        print("Checkpoint not found. Please train the model first.")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Dataset
    dataset = ModiSSLPretrainDataset(NOISY_IMAGES_DIR, SYNTHETIC_IMAGES_DIR)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Evaluation loop
    with torch.no_grad():
        for i, (noisy_tensor, synthetic_tensor) in enumerate(dataloader):
            noisy_tensor = noisy_tensor.to(device)
            synthetic_tensor = synthetic_tensor.to(device)

            z1, z2 = model(noisy_tensor, synthetic_tensor)
            similarity = torch.nn.functional.cosine_similarity(z1, z2).item()

            print(f"[Sample {i+1}] Cosine Similarity: {similarity:.4f}")
            visualize_images(noisy_tensor, synthetic_tensor, similarity, z1, z2)

            if i >= 4:  # Show only first 5 samples
                break

if __name__ == "__main__":
    evaluate_ssl()
