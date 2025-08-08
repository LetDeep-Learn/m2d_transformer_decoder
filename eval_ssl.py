# eval_ssl.py
"""
Evaluation script for SSL encoder (ModiSSLModel).

- Loads checkpoint (robust partial loader).
- Computes normalized embeddings for noisy and synthetic views.
- Evaluates retrieval: Top-1 and Top-5 (noisy -> synthetic).
- Prints pos/neg similarity stats and a few sample matches.
- Optionally saves embeddings to .npy files.

Usage:
    python eval_ssl.py --ckpt /path/to/ssl_checkpoint.pth --save_emb ./emb/
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import (
    DRIVE_SAVE_DIR,
    NOISY_IMAGES_DIR,
    SYNTHETIC_IMAGES_DIR,
    BATCH_SIZE,
    DEVICE,
    NUM_WORKERS,
    PIN_MEMORY
)
from dataset_ssl import ModiSSLPretrainDataset
from model_ssl import ModiSSLModel

# ---------------------------
# Helpers
# ---------------------------
def load_partial_checkpoint(model, ckpt_path, device='cpu'):
    """
    Load any matching parameters from checkpoint into model.
    Returns number_loaded, total.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt

    model_state = model.state_dict()
    filtered = {}
    for k, v in state_dict.items():
        if k in model_state and v.size() == model_state[k].size():
            filtered[k] = v

    model_state.update(filtered)
    model.load_state_dict(model_state)
    return len(filtered), len(model_state)

@torch.no_grad()
def compute_embeddings(model, dataloader, device):
    """
    Returns two tensors: noisy_embeddings [N, D], synth_embeddings [N, D]
    """
    model.eval()
    all_noisy = []
    all_synth = []
    for noisy, synth in dataloader:
        noisy = noisy.to(device)
        synth = synth.to(device)
        p1, p2 = model(noisy, synth)  # normalized outputs [B, D]
        all_noisy.append(p1.cpu())
        all_synth.append(p2.cpu())
    all_noisy = torch.cat(all_noisy, dim=0)
    all_synth = torch.cat(all_synth, dim=0)
    return all_noisy, all_synth

def retrieval_metrics(noisy_emb, synth_emb, topk=(1,5), chunk_size=512, device='cpu'):
    """
    Compute retrieval top-k metrics (noisy -> synth).
    Uses chunking for memory efficiency.
    Returns dict with topk accuracies and sim stats.
    """
    noisy = noisy_emb.to(device)
    synth = synth_emb.to(device)
    N = noisy.size(0)
    D = noisy.size(1)

    # positive similarity per sample (dot of corresponding rows)
    pos_sim = (noisy * synth).sum(dim=1).cpu().numpy()

    # We compute similarity matrix in chunks to avoid O(N^2) memory blow-up
    topk_hits = {k: 0 for k in topk}
    # for debugging: record predicted indices for first few
    preds_for_display = []

    for i in range(0, N, chunk_size):
        ni = noisy[i: i + chunk_size]  # [c, D]
        # sim block: [c, N]
        sim_block = ni @ synth.t()
        # for each row, get top max(topk) indices
        max_k = max(topk)
        vals, idxs = torch.topk(sim_block, k=max_k, dim=1)
        idxs = idxs.cpu().numpy()
        for r in range(idxs.shape[0]):
            global_idx = i + r
            for k in topk:
                if global_idx < N:
                    topk_inds = idxs[r, :k]
                    if global_idx in topk_inds:
                        topk_hits[k] += 1
        # store some for printing
        if len(preds_for_display) < 10:
            for r in range(idxs.shape[0]):
                preds_for_display.append((i + r, idxs[r, 0], float(vals[r, 0].cpu().item())))
                if len(preds_for_display) >= 10:
                    break

    metrics = {}
    for k in topk:
        metrics[f"top{k}"] = topk_hits[k] / N

    metrics["pos_sim_mean"] = float(np.mean(pos_sim))
    metrics["pos_sim_std"] = float(np.std(pos_sim))
    metrics["preds_sample"] = preds_for_display
    return metrics

# ---------------------------
# CLI / main
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=os.path.join(DRIVE_SAVE_DIR, "ssl_checkpoint.pth"),
                   help="Checkpoint path (model or dict with 'model').")
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--save_emb", type=str, default=None, help="Optional directory to save embeddings as .npy")
    p.add_argument("--chunk_size", type=int, default=512, help="Chunk size for similarity computation")
    p.add_argument("--topk", type=int, nargs="+", default=[1,5], help="Top-k values to compute")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    print("Loading dataset...")
    dataset = ModiSSLPretrainDataset(noisy_dir=NOISY_IMAGES_DIR, synthetic_dir=SYNTHETIC_IMAGES_DIR)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    print("Building model and loading checkpoint...")
    model = ModiSSLModel().to(device)
    if os.path.exists(args.ckpt):
        try:
            n_loaded, n_total = load_partial_checkpoint(model, args.ckpt, device=device)
            print(f"[ckpt] Loaded {n_loaded}/{n_total} matching params from {args.ckpt}")
        except Exception as e:
            print("[ckpt] Failed to partial-load checkpoint:", e)
            print("[ckpt] Attempting strict load ...")
            ckpt = torch.load(args.ckpt, map_location=device)
            state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
            model.load_state_dict(state_dict)
            print("[ckpt] strict load successful.")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    print("Computing embeddings...")
    noisy_emb, synth_emb = compute_embeddings(model, loader, device)
    print("Embeddings shapes:", noisy_emb.shape, synth_emb.shape)

    if args.save_emb:
        os.makedirs(args.save_emb, exist_ok=True)
        np.save(os.path.join(args.save_emb, "noisy_emb.npy"), noisy_emb.numpy())
        np.save(os.path.join(args.save_emb, "synth_emb.npy"), synth_emb.numpy())
        print(f"Saved embeddings to {args.save_emb}")

    print("Evaluating retrieval...")
    metrics = retrieval_metrics(noisy_emb, synth_emb, topk=tuple(args.topk), chunk_size=args.chunk_size, device=device)
    for k in args.topk:
        print(f"Top-{k} retrieval: {metrics[f'top{k}']:.4f}")
    print(f"Pos sim mean/std: {metrics['pos_sim_mean']:.4f} / {metrics['pos_sim_std']:.4f}")

    print("\nSample predictions (noisy_index -> predicted_synth_index, similarity):")
    for i, pred_idx, sim_val in metrics["preds_sample"]:
        print(f"  {i} -> {pred_idx} (sim={sim_val:.4f})")

    print("\nDone.")

if __name__ == "__main__":
    main()
