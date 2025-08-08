# train_ssl.py
"""
SSL training script (ResNet-50 backbone + NT-Xent).
Drop-in for the ModiSSLModel (ResNet50-based) from model_ssl.py.
"""
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Import dataset and model
from dataset_ssl import ModiSSLPretrainDataset
from model_ssl import ModiSSLModel

# Import config with fallbacks to reasonable defaults
try:
    from config import (
        DRIVE_SAVE_DIR,
        NOISY_IMAGES_DIR,
        SYNTHETIC_IMAGES_DIR,
        BATCH_SIZE,
        EPOCHS,
        LEARNING_RATE,
        CHECKPOINT_FILE,
        DEVICE,
        ACCUM_STEPS,
        NTXENT_TEMPERATURE,
        NUM_WORKERS,
        PIN_MEMORY,
        WEIGHT_DECAY,
        PROJ_DIM
    )
except Exception:
    # fallbacks
    DRIVE_SAVE_DIR = "./saved_models_ssl"
    NOISY_IMAGES_DIR = "./modi_dataset/noisy_images"
    SYNTHETIC_IMAGES_DIR = "./modi_dataset/clean_images"
    BATCH_SIZE = 8
    EPOCHS = 100
    LEARNING_RATE = 3e-4
    CHECKPOINT_FILE = "ssl_checkpoint.pth"
    DEVICE = "cuda"
    ACCUM_STEPS = 8
    NTXENT_TEMPERATURE = 0.07
    NUM_WORKERS = 2
    PIN_MEMORY = True
    WEIGHT_DECAY = 1e-6
    PROJ_DIM = 256

os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)
checkpoint_path = os.path.join(DRIVE_SAVE_DIR, CHECKPOINT_FILE)

# ---------------------------
# NT-Xent (stable) implementation
# ---------------------------
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        """
        z1, z2: [B, D] (assumed normalized)
        returns scalar loss (>=0)
        """
        B = z1.size(0)
        assert z2.size(0) == B
        z = torch.cat([z1, z2], dim=0)  # [2B, D]
        sim = torch.matmul(z, z.t()) / self.temperature  # [2B,2B]

        # numerical stability
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim_stable = sim - sim_max.detach()  # [2B,2B]

        # mask out self-similarity
        mask = (~torch.eye(2 * B, 2 * B, dtype=torch.bool, device=z.device)).float()
        exp_sim = torch.exp(sim_stable) * mask  # zeros on diagonal
        denom = exp_sim.sum(dim=1)  # [2B]

        # positive logits: pairs (i, i+B) and (i+B, i)
        pos_logits = torch.cat([
            sim_stable[:B, B:2*B].diag(),
            sim_stable[B:2*B, :B].diag()
        ], dim=0)  # [2B]

        pos_exp = torch.exp(pos_logits)
        loss = -torch.log((pos_exp / denom).clamp_min(1e-12))
        return loss.mean()

# ---------------------------
# Partial checkpoint loader (safe)
# ---------------------------
def load_partial_checkpoint(model, ckpt_path, device='cpu'):
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt

    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in state_dict.items():
        if k in model_state and v.size() == model_state[k].size():
            filtered[k] = v
        else:
            skipped.append(k)

    model_state.update(filtered)
    model.load_state_dict(model_state)
    return len(filtered), len(model_state), skipped

# ---------------------------
# Retrieval metric (Top-K)
# ---------------------------
@torch.no_grad()
def retrieval_topk(model, dataset, device, topk=(1,5), batch_size=64, num_workers=2, pin_memory=True):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    all_noisy = []
    all_synth = []

    for noisy, synth in loader:
        noisy = noisy.to(device)
        synth = synth.to(device)
        p1, p2 = model(noisy, synth)
        all_noisy.append(p1.cpu())
        all_synth.append(p2.cpu())

    all_noisy = torch.cat(all_noisy, dim=0)  # [N, D]
    all_synth = torch.cat(all_synth, dim=0)  # [N, D]

    N = all_noisy.size(0)
    sim = all_noisy @ all_synth.t()  # [N, N] on CPU
    results = {}
    for k in topk:
        topk_idx = torch.topk(sim, k=k, dim=1).indices  # [N, k]
        matches = (topk_idx == torch.arange(N).unsqueeze(1)).any(dim=1).float()
        results[f"top{k}"] = matches.mean().item()
    return results

# ---------------------------
# Training loop
# ---------------------------
def train_ssl():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # dataset + dataloader
    dataset = ModiSSLPretrainDataset(noisy_dir=NOISY_IMAGES_DIR, synthetic_dir=SYNTHETIC_IMAGES_DIR, debug=False)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # model / loss / optimizer
    model = ModiSSLModel(proj_dim=PROJ_DIM).to(device)
    criterion = NTXentLoss(temperature=NTXENT_TEMPERATURE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # cosine annealing scheduler
    total_steps = max(1, math.ceil(len(dataloader) * EPOCHS / max(1, ACCUM_STEPS)))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, EPOCHS), eta_min=1e-7)

    start_epoch = 0
    optimizer_loaded = False

    # checkpoint loading (partial)
    if os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
            if isinstance(ckpt, dict) and 'model' in ckpt:
                n_loaded, n_total, skipped = load_partial_checkpoint(model, checkpoint_path, device=device)
                print(f"[ckpt] Loaded {n_loaded}/{n_total} params from checkpoint. Skipped {len(skipped)} keys.")
                # try optimizer load
                try:
                    if 'optimizer' in ckpt:
                        optimizer.load_state_dict(ckpt['optimizer'])
                        optimizer_loaded = True
                        print("[ckpt] optimizer state loaded.")
                except Exception as e:
                    print("[ckpt] optimizer restore failed:", e)
                start_epoch = ckpt.get('epoch', -1) + 1
            else:
                n_loaded, n_total, skipped = load_partial_checkpoint(model, checkpoint_path, device=device)
                print(f"[ckpt] Loaded {n_loaded}/{n_total} params from legacy checkpoint.")
                start_epoch = 0
        except Exception as e:
            print("[ckpt] Failed to load checkpoint — starting fresh. Error:", e)
            start_epoch = 0
    else:
        print("[ckpt] No checkpoint found. Starting fresh training...")

    # training loop
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, (noisy_img, synthetic_img) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
            noisy_img = noisy_img.to(device)
            synthetic_img = synthetic_img.to(device)

            p1, p2 = model(noisy_img, synthetic_img)  # normalized [B, dim]

            # diagnostics on first batch
            if batch_idx == 0:
                pos_sim = (p1 * p2).sum(dim=1)
                print(f"[diag] pos_sim mean={pos_sim.mean().item():.4f}, std={pos_sim.std().item():.4f}")
                print(f"[diag] p1 per-dim std mean={p1.std(dim=0).mean().item():.6f}, p2 per-dim std mean={p2.std(dim=0).mean().item():.6f}")
                try:
                    max_abs = (noisy_img[0] - synthetic_img[0]).abs().max().item()
                    print(f"[diag] first-sample pixel max-abs-diff: {max_abs:.6f}")
                except Exception:
                    pass

            loss = criterion(p1, p2) / max(1, ACCUM_STEPS)
            loss.backward()
            total_loss += loss.item() * max(1, ACCUM_STEPS)

            # step when accumulation reached
            if (batch_idx + 1) % ACCUM_STEPS == 0:
                # optional gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                optimizer.step()
                optimizer.zero_grad()

        # final step for leftover gradients
        if (batch_idx + 1) % ACCUM_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(dataloader)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.6f} - LR: {current_lr:.6e}")

        # save checkpoint
        save_dict = {"model": model.state_dict(), "epoch": epoch}
        try:
            save_dict["optimizer"] = optimizer.state_dict()
        except Exception:
            pass
        torch.save(save_dict, checkpoint_path)
        print(f"[checkpoint] Saved epoch {epoch} to {checkpoint_path}")

        # quick retrieval eval
        try:
            res = retrieval_topk(model, dataset, device, topk=(1,5), batch_size=min(128, BATCH_SIZE*4), num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
            print(f"[eval] Retrieval Top-1: {res['top1']:.4f} Top-5: {res['top5']:.4f}")
        except Exception as e:
            print("[eval] retrieval failed:", e)

    print("Training complete.")

if __name__ == "__main__":
    train_ssl()
