#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================================================
# Dataset
# =========================================================
class MultiChunkDataset(Dataset):
    """
    Read multiple chunks.npy files as one unified dataset.
    Each file is expected to have shape (N, L).
    """
    def __init__(self, chunk_paths):
        self.chunk_paths = chunk_paths
        self.arrays = []
        self.lengths = []
        self.cum_lengths = []

        total = 0
        input_len = None

        for path in chunk_paths:
            arr = np.load(path, mmap_mode="r")
            if arr.ndim != 2:
                raise ValueError(f"[ERROR] Expected (N, L), got {arr.shape} for {path}")

            if input_len is None:
                input_len = arr.shape[1]
            elif arr.shape[1] != input_len:
                raise ValueError(
                    f"[ERROR] Inconsistent chunk length. "
                    f"{path} has L={arr.shape[1]}, expected L={input_len}"
                )

            self.arrays.append(arr)
            self.lengths.append(len(arr))
            total += len(arr)
            self.cum_lengths.append(total)

        self.input_len = input_len
        self.total_len = total

    def __len__(self):
        return self.total_len

    def _locate(self, idx):
        file_idx = 0
        while idx >= self.cum_lengths[file_idx]:
            file_idx += 1
        prev_cum = 0 if file_idx == 0 else self.cum_lengths[file_idx - 1]
        local_idx = idx - prev_cum
        return file_idx, local_idx

    def __getitem__(self, idx):
        file_idx, local_idx = self._locate(idx)
        x = self.arrays[file_idx][local_idx].astype(np.float32)   # (L,)
        x = torch.from_numpy(x).unsqueeze(0)                      # (1, L)
        return x, file_idx


# =========================================================
# Model1: Canonical Autoencoder
# =========================================================
class CanonicalAutoencoder(nn.Module):
    def __init__(self, input_len=128, latent_dim=32):
        super().__init__()
        if input_len % 16 != 0:
            raise ValueError(f"input_len must be divisible by 16, got {input_len}")

        self.input_len = input_len
        self.latent_dim = latent_dim
        self.reduced_len = input_len // 16

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(16),
            nn.GELU(),

            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )

        self.flatten_dim = 128 * self.reduced_len
        self.to_latent = nn.Linear(self.flatten_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, self.flatten_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),

            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),

            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.GELU(),

            nn.ConvTranspose1d(16, 1, kernel_size=4, stride=2, padding=1),
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.flatten(start_dim=1)
        z = self.to_latent(h)
        return z

    def decode(self, z):
        h = self.from_latent(z)
        h = h.view(z.size(0), 128, self.reduced_len)
        x_hat = self.decoder(h)
        return x_hat

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


# =========================================================
# Inference
# =========================================================
@torch.no_grad()
def run_inference(model, dataset, batch_size, device, out_dir, prefix):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()

    all_x = []
    all_xhat = []
    all_z = []
    all_residual = []
    all_mse = []
    all_source_index = []

    for x, file_idx in loader:
        x = x.to(device)
        x_hat, z = model(x)

        residual = x - x_hat
        mse = ((x - x_hat) ** 2).mean(dim=(1, 2))

        all_x.append(x.cpu().numpy())
        all_xhat.append(x_hat.cpu().numpy())
        all_z.append(z.cpu().numpy())
        all_residual.append(residual.cpu().numpy())
        all_mse.append(mse.cpu().numpy())
        all_source_index.append(file_idx.numpy())

    x_np = np.concatenate(all_x, axis=0)[:, 0, :].astype(np.float32)
    xhat_np = np.concatenate(all_xhat, axis=0)[:, 0, :].astype(np.float32)
    z_np = np.concatenate(all_z, axis=0).astype(np.float32)
    residual_np = np.concatenate(all_residual, axis=0)[:, 0, :].astype(np.float32)
    mse_np = np.concatenate(all_mse, axis=0).astype(np.float32)
    source_index_np = np.concatenate(all_source_index, axis=0).astype(np.int32)

    np.save(os.path.join(out_dir, f"{prefix}_input.npy"), x_np)
    np.save(os.path.join(out_dir, f"{prefix}_recon.npy"), xhat_np)
    np.save(os.path.join(out_dir, f"{prefix}_latent.npy"), z_np)
    np.save(os.path.join(out_dir, f"{prefix}_residual.npy"), residual_np)
    np.save(os.path.join(out_dir, f"{prefix}_recon_mse.npy"), mse_np)
    np.save(os.path.join(out_dir, f"{prefix}_source_index.npy"), source_index_np)


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Run model1 inference on one or more chunks.npy files"
    )
    parser.add_argument(
        "--chunks",
        type=str,
        nargs="+",
        required=True,
        help="One or more paths to chunks.npy files"
    )
    parser.add_argument(
        "--model_ckpt",
        type=str,
        required=True,
        help="Path to model1_best.pt"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="native_model1",
        help="Prefix for output files"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    print("[INFO] Loading chunk files:")
    for p in args.chunks:
        print("  ", p)

    dataset = MultiChunkDataset(args.chunks)
    print(f"[INFO] total_chunks = {len(dataset)}")
    print(f"[INFO] input_len    = {dataset.input_len}")

    ckpt = torch.load(args.model_ckpt, map_location=device)

    input_len = int(ckpt["input_len"])
    latent_dim = int(ckpt["latent_dim"])

    if dataset.input_len != input_len:
        raise ValueError(
            f"[ERROR] Chunk length mismatch: dataset has {dataset.input_len}, "
            f"but model expects {input_len}"
        )

    model = CanonicalAutoencoder(
        input_len=input_len,
        latent_dim=latent_dim
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    print("[INFO] Loaded model checkpoint:", args.model_ckpt)
    print("[INFO] model input_len =", input_len)
    print("[INFO] model latent_dim =", latent_dim)

    run_inference(
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        device=device,
        out_dir=args.out_dir,
        prefix=args.prefix
    )

    source_map = {i: p for i, p in enumerate(args.chunks)}
    with open(os.path.join(args.out_dir, "source_map.json"), "w") as f:
        json.dump(source_map, f, indent=2)

    infer_config = {
        "chunks": args.chunks,
        "model_ckpt": args.model_ckpt,
        "out_dir": args.out_dir,
        "batch_size": args.batch_size,
        "prefix": args.prefix,
        "input_len": input_len,
        "latent_dim": latent_dim,
        "total_chunks": len(dataset),
    }
    with open(os.path.join(args.out_dir, "infer_config.json"), "w") as f:
        json.dump(infer_config, f, indent=2)

    print("[DONE] Saved files:")
    print(f"  {args.prefix}_input.npy")
    print(f"  {args.prefix}_recon.npy")
    print(f"  {args.prefix}_latent.npy")
    print(f"  {args.prefix}_residual.npy")
    print(f"  {args.prefix}_recon_mse.npy")
    print(f"  {args.prefix}_source_index.npy")
    print("  source_map.json")
    print("  infer_config.json")


if __name__ == "__main__":
    main()