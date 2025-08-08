# train_ssl.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from dataset_ssl import ModiSSLPretrainDataset
from model_ssl import ModiSSLModel
from config import (
    DRIVE_SAVE_DIR,
    NOISY_IMAGES_DIR,
    SYNTHETIC_IMAGES_DIR,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    CHECKPOINT_FILE,
    DEVICE
)

# ---------------------------
# NT-Xent (InfoNCE) loss
# ---------------------------
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        """
        z1, z2: [B, D] normalized
        returns scalar loss (>= 0)
        """
        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)             # [2B, D]
        sim = torch.matmul(z, z.t()) / self.temperature  # [2B, 2B]

        # numerical stability: subtract max per row (keep same basis for pos)
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim_stable = sim - sim_max.detach()        # [2B, 2B]

        # mask to remove self-similarity from denom
        mask = (~torch.eye(2 * B, 2 * B, dtype=torch.bool, device=z.device)).float()

        exp_sim = torch.exp(sim_stable) * mask     # zeros on diagonal

        denom = exp_sim.sum(dim=1)                 # [2B]

        # positive pairs are (i, i+B) and (i+B, i)
        pos_logits = torch.cat([
            sim_stable[:B, B:2*B].diag(),   # sim_stable[i, i+B] for i in 0..B-1
            sim_stable[B:2*B, :B].diag()    # sim_stable[i+B, i] for i in 0..B-1
        ], dim=0)                            # [2B]

        pos_exp = torch.exp(pos_logits)       # uses the same stabilization shift

        loss = -torch.log( (pos_exp / denom).clamp_min(1e-12) )
        return loss.mean()

# ---------------------------
# Partial checkpoint loader
# ---------------------------
def load_partial_checkpoint(model, ckpt_path, device='cpu'):
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt

    model_state = model.state_dict()
    filtered = {}
    mismatched = []
    for k, v in state_dict.items():
        if k in model_state and v.size() == model_state[k].size():
            filtered[k] = v
        else:
            mismatched.append(k)

    model_state.update(filtered)
    model.load_state_dict(model_state)
    print(f"[checkpoint] Loaded {len(filtered)} / {len(model_state)} matching params from checkpoint.")
    if mismatched:
        print(f"[checkpoint] {len(mismatched)} keys mismatched or skipped (first 10): {mismatched[:10]}")

# ---------------------------
# Retrieval metric (Top-1)
# ---------------------------
@torch.no_grad()
def retrieval_top1(model, dataset, device, batch_size=32, num_workers=2):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    all_noisy = []
    all_synth = []

    for noisy, synth in loader:
        noisy = noisy.to(device)
        synth = synth.to(device)
        z_noisy, z_synth = model(noisy, synth)  # normalized projections
        all_noisy.append(z_noisy.cpu())
        all_synth.append(z_synth.cpu())

    all_noisy = torch.cat(all_noisy, dim=0)   # [N, D]
    all_synth = torch.cat(all_synth, dim=0)   # [N, D]

    # cosine similarities (N x N)
    sim = all_noisy @ all_synth.t()
    top1 = sim.argmax(dim=1)
    correct = (top1 == torch.arange(all_noisy.size(0))).float().mean().item()
    return correct

# ---------------------------
# Training function
# ---------------------------
def train_ssl():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)
    checkpoint_path = os.path.join(DRIVE_SAVE_DIR, CHECKPOINT_FILE)

    # Dataset & DataLoader
    dataset = ModiSSLPretrainDataset(noisy_dir=NOISY_IMAGES_DIR, synthetic_dir=SYNTHETIC_IMAGES_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # Model, Loss, Optimizer
    model = ModiSSLModel().to(device)
    criterion = NTXentLoss(temperature=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)

    # Optional gradient accumulation to simulate larger batch size
    # If your GPU is small, bump ACCUM_STEPS to simulate a bigger batch (effective_batch = BATCH_SIZE * ACCUM_STEPS)
    ACCUM_STEPS = 8  # change to 1 if you want no accumulation

    start_epoch = 0
    optimizer_loaded = False

    # Try to load partial checkpoint safely
    if os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
            if isinstance(ckpt, dict) and 'model' in ckpt:
                load_partial_checkpoint(model, checkpoint_path, device=device)
                # Attempt to load optimizer if shapes match; wrap in try/except
                try:
                    if 'optimizer' in ckpt:
                        optimizer.load_state_dict(ckpt['optimizer'])
                        optimizer_loaded = True
                        print("[checkpoint] optimizer state loaded.")
                except Exception as e:
                    print("[checkpoint] optimizer load failed (shapes mismatch) — starting optimizer fresh.", e)
                start_epoch = ckpt.get('epoch', -1) + 1
                print(f"[checkpoint] Resuming from epoch {start_epoch}")
            else:
                # legacy single state-dict
                load_partial_checkpoint(model, checkpoint_path, device=device)
                print("[checkpoint] Loaded old-format checkpoint (model only). Starting from epoch 0.")
                start_epoch = 0
        except Exception as e:
            print("[checkpoint] Failed to load checkpoint; starting fresh. Error:", e)
            start_epoch = 0
    else:
        print("[checkpoint] No checkpoint found. Starting fresh training...")

    # Training loop
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, (noisy_img, synthetic_img) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
            noisy_img = noisy_img.to(device)
            synthetic_img = synthetic_img.to(device)

            p1, p2 = model(noisy_img, synthetic_img)  # normalized outputs [B, D]

            # Diagnostics on first batch of epoch
            if batch_idx == 0:
                pos_sim = (p1 * p2).sum(dim=1)  # per-sample cosine similarity
                print(f"[diag] pos_sim mean={pos_sim.mean().item():.4f}, std={pos_sim.std().item():.4f}")
                print(f"[diag] p1 per-dim std mean={p1.std(dim=0).mean().item():.6f}, p2 per-dim std mean={p2.std(dim=0).mean().item():.6f}")

                # quick pixel check for first sample (to ensure views not identical accidentally)
                try:
                    max_abs = (noisy_img[0] - synthetic_img[0]).abs().max().item()
                    print(f"[diag] first-sample pixel max-abs-diff: {max_abs:.6f}")
                except Exception:
                    pass

            loss = criterion(p1, p2) / ACCUM_STEPS
            loss.backward()
            total_loss += loss.item() * ACCUM_STEPS  # accumulate real loss

            # step when accumulation reached
            if (batch_idx + 1) % ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

        # final step if leftover gradients
        if (batch_idx + 1) % ACCUM_STEPS != 0:
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.6f}")

        # Save model checkpoint (model always saved)
        save_dict = {
            "model": model.state_dict(),
            "epoch": epoch
        }
        # add optimizer state only if it was loaded successfully before or if shapes are consistent
        try:
            save_dict["optimizer"] = optimizer.state_dict()
        except Exception:
            pass

        torch.save(save_dict, checkpoint_path)
        print(f"[checkpoint] Saved epoch {epoch} to {checkpoint_path}")

        # End-of-epoch retrieval evaluation (Top-1)
        try:
            top1 = retrieval_top1(model, dataset, device, batch_size=min(64, BATCH_SIZE*4), num_workers=2)
            print(f"[eval] Retrieval Top-1: {top1:.4f}")
        except Exception as e:
            print("[eval] retrieval failed:", e)

    print("Training complete.")

if __name__ == "__main__":
    train_ssl()
