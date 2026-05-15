#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import argparse
import bisect
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# =========================================================
# Dataset
# =========================================================
class MultiChunkDataset(Dataset):
    """
    Read multiple chunks.npy files as one unified dataset.
    Each file is expected to have shape (N, L).

    Improvements:
      - mmap loading to avoid full memory load
      - bisect for O(log n_files) global-index lookup
      - optional lightweight startup summary
    """
    def __init__(self, chunk_paths, verbose=True):
        self.chunk_paths = chunk_paths
        self.arrays = []
        self.lengths = []
        self.cum_lengths = []

        total = 0
        input_len = None

        if verbose:
            print("[INFO] Building dataset from chunk files...")

        for i, path in enumerate(chunk_paths):
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

            if verbose:
                print(f"  [{i+1}/{len(chunk_paths)}] {path} | shape={arr.shape}")

        self.input_len = input_len
        self.total_len = total

        if verbose:
            print(f"[INFO] Dataset ready | files={len(self.arrays)} | total_chunks={self.total_len} | input_len={self.input_len}")

    def __len__(self):
        return self.total_len

    def _locate(self, idx):
        # map global idx -> (file_idx, local_idx) using binary search
        file_idx = bisect.bisect_right(self.cum_lengths, idx)
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

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),   # L -> L/2
            nn.BatchNorm1d(16),
            nn.GELU(),

            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),  # -> L/4
            nn.BatchNorm1d(32),
            nn.GELU(),

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),  # -> L/8
            nn.BatchNorm1d(64),
            nn.GELU(),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2), # -> L/16
            nn.BatchNorm1d(128),
            nn.GELU(),
        )

        self.flatten_dim = 128 * self.reduced_len
        self.to_latent = nn.Linear(self.flatten_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, self.flatten_dim)

        # Decoder
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
        h = self.encoder(x)                         # (B,128,L/16)
        h = h.flatten(start_dim=1)                 # (B,128*L/16)
        z = self.to_latent(h)                      # (B,latent_dim)
        return z

    def decode(self, z):
        h = self.from_latent(z)                    # (B,128*L/16)
        h = h.view(z.size(0), 128, self.reduced_len)
        x_hat = self.decoder(h)                    # (B,1,L)
        return x_hat

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


# =========================================================
# Loss
# =========================================================
def reconstruction_loss(x, x_hat, alpha_l1=0.0):
    mse = nn.functional.mse_loss(x_hat, x)
    if alpha_l1 > 0:
        l1 = nn.functional.l1_loss(x_hat, x)
        return mse + alpha_l1 * l1
    return mse


# =========================================================
# Train / Eval
# =========================================================
def run_epoch(
    model,
    loader,
    optimizer,
    device,
    alpha_l1=0.0,
    train=True,
    log_every=100,
    use_tqdm=True,
    debug_first_batch=False,
):
    if train:
        model.train()
        stage = "train"
    else:
        model.eval()
        stage = "val"

    total_loss = 0.0
    total_num = 0
    start_epoch = time.time()

    num_batches = len(loader)
    print(f"[INFO] {stage} epoch start | num_batches={num_batches} | device={device}")

    iterable = loader
    if use_tqdm and tqdm is not None:
        iterable = tqdm(loader, desc=stage, leave=False, dynamic_ncols=True)

    first_batch_t0 = time.time()

    for step, (x, _file_idx) in enumerate(iterable, start=1):
        t_fetch_done = time.time()

        if debug_first_batch and step == 1:
            print(f"[DEBUG] first batch fetched | x.shape={tuple(x.shape)} | fetch_wait={t_fetch_done - first_batch_t0:.3f}s")

        x = x.to(device, non_blocking=(device.type == "cuda"))

        if debug_first_batch and step == 1:
            print(f"[DEBUG] first batch moved to {device}")

        if train:
            optimizer.zero_grad(set_to_none=True)

        t0 = time.time()
        with torch.set_grad_enabled(train):
            x_hat, z = model(x)
            if debug_first_batch and step == 1:
                print(f"[DEBUG] first batch forward done | x_hat.shape={tuple(x_hat.shape)} | z.shape={tuple(z.shape)}")

            loss = reconstruction_loss(x, x_hat, alpha_l1=alpha_l1)

            if train:
                loss.backward()
                if debug_first_batch and step == 1:
                    print("[DEBUG] first batch backward done")
                optimizer.step()
                if debug_first_batch and step == 1:
                    print("[DEBUG] first batch optimizer step done")

        batch_compute_time = time.time() - t0

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_num += bs
        avg_loss = total_loss / max(total_num, 1)

        if use_tqdm and tqdm is not None:
            iterable.set_postfix(loss=f"{avg_loss:.6f}")
        elif (step % log_every == 0) or (step == 1) or (step == num_batches):
            elapsed = time.time() - start_epoch
            print(
                f"[INFO] {stage} step {step}/{num_batches} | "
                f"batch_size={bs} | batch_loss={loss.item():.6f} | avg_loss={avg_loss:.6f} | "
                f"compute={batch_compute_time:.3f}s | elapsed={elapsed:.1f}s"
            )

    epoch_time = time.time() - start_epoch
    avg_epoch_loss = total_loss / max(total_num, 1)
    print(f"[INFO] {stage} epoch done | avg_loss={avg_epoch_loss:.6f} | samples={total_num} | time={epoch_time:.1f}s")
    return avg_epoch_loss, epoch_time


# =========================================================
# Export model1 outputs on full dataset
# =========================================================
@torch.no_grad()
def export_outputs(model, dataset, batch_size, device, out_dir, prefix="canonical_model1", num_workers=0, pin_memory=False):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    model.eval()

    all_x = []
    all_xhat = []
    all_z = []
    all_residual = []
    all_mse = []
    all_source_index = []

    iterable = loader
    if tqdm is not None:
        iterable = tqdm(loader, desc="export", leave=False, dynamic_ncols=True)

    print(f"[INFO] export start | batches={len(loader)}")

    for x, file_idx in iterable:
        x = x.to(device, non_blocking=(device.type == "cuda"))
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
    print("[INFO] export done")


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Train model1 canonical autoencoder from multiple chunks.npy files"
    )
    parser.add_argument(
        "--chunks",
        type=str,
        nargs="+",
        required=True,
        help="One or more paths to chunks.npy files"
    )
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--latent_dim", type=int, default=32, help="Latent dimension")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--alpha_l1", type=float, default=0.0, help="Optional L1 loss weight")
    parser.add_argument("--export_prefix", type=str, default="canonical_model1", help="Prefix for exported npy files")

    # New usability / performance args
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--pin_memory", action="store_true", help="Enable DataLoader pin_memory")
    parser.add_argument("--log_every", type=int, default=100, help="Print every N batches when tqdm is unavailable")
    parser.add_argument("--no_tqdm", action="store_true", help="Disable tqdm progress bar")
    parser.add_argument("--debug_first_batch", action="store_true", help="Verbose debug for the first batch of each epoch")
    parser.add_argument("--max_chunks", type=int, default=0, help="Use only the first N chunks for quick debugging (0 = all)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        pin_memory = args.pin_memory
    else:
        device = torch.device("cpu")
        pin_memory = False

    print("=" * 60)
    print("[INFO] Training canonical autoencoder")
    print(f"[INFO] device         = {device}")
    print(f"[INFO] batch_size     = {args.batch_size}")
    print(f"[INFO] epochs         = {args.epochs}")
    print(f"[INFO] lr             = {args.lr}")
    print(f"[INFO] latent_dim     = {args.latent_dim}")
    print(f"[INFO] val_ratio      = {args.val_ratio}")
    print(f"[INFO] num_workers    = {args.num_workers}")
    print(f"[INFO] pin_memory     = {pin_memory}")
    print(f"[INFO] tqdm_enabled   = {not args.no_tqdm and tqdm is not None}")
    print("[INFO] Loading chunk files:")
    for p in args.chunks:
        print("  ", p)
    print("=" * 60)

    dataset = MultiChunkDataset(args.chunks, verbose=True)
    input_len = dataset.input_len

    if args.max_chunks > 0:
        n_keep = min(args.max_chunks, len(dataset))
        dataset = Subset(dataset, list(range(n_keep)))
        print(f"[INFO] max_chunks active | using first {n_keep} chunks only")

    print(f"[INFO] total_chunks = {len(dataset)}")
    print(f"[INFO] input_len    = {input_len}")

    n_total = len(dataset)
    n_val = int(n_total * args.val_ratio)
    n_train = n_total - n_val

    if n_train <= 0 or n_val <= 0:
        raise ValueError(
            f"Dataset too small for val split. total={n_total}, train={n_train}, val={n_val}"
        )

    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )

    common_loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        **common_loader_kwargs,
    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        **common_loader_kwargs,
    )

    print(f"[INFO] train_chunks = {len(train_set)} | val_chunks = {len(val_set)}")
    print(f"[INFO] train_batches = {len(train_loader)} | val_batches = {len(val_loader)}")

    model = CanonicalAutoencoder(
        input_len=input_len,
        latent_dim=args.latent_dim
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] model params = {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    best_ckpt = os.path.join(args.out_dir, "model1_best.pt")
    history = []

    for epoch in range(1, args.epochs + 1):
        print("-" * 60)
        print(f"[INFO] Epoch {epoch:03d}/{args.epochs}")

        train_loss, train_time = run_epoch(
            model, train_loader, optimizer, device,
            alpha_l1=args.alpha_l1, train=True,
            log_every=args.log_every,
            use_tqdm=(not args.no_tqdm),
            debug_first_batch=args.debug_first_batch,
        )
        val_loss, val_time = run_epoch(
            model, val_loader, optimizer, device,
            alpha_l1=args.alpha_l1, train=False,
            log_every=args.log_every,
            use_tqdm=(not args.no_tqdm),
            debug_first_batch=args.debug_first_batch,
        )

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "train_time_sec": float(train_time),
            "val_time_sec": float(val_time),
        })

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_len": input_len,
                    "latent_dim": args.latent_dim,
                    "best_val_loss": best_val,
                    "chunk_files": args.chunks,
                },
                best_ckpt
            )
            print(f"[INFO] New best checkpoint saved: {best_ckpt}")

    print("[INFO] Best val loss:", best_val)

    config = {
        "chunks": args.chunks,
        "out_dir": args.out_dir,
        "input_len": input_len,
        "latent_dim": args.latent_dim,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "alpha_l1": args.alpha_l1,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "log_every": args.log_every,
        "tqdm_enabled": (not args.no_tqdm and tqdm is not None),
        "max_chunks": args.max_chunks,
        "best_val_loss": best_val,
    }
    with open(os.path.join(args.out_dir, "train_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    with open(os.path.join(args.out_dir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    source_map = {i: p for i, p in enumerate(args.chunks)}
    with open(os.path.join(args.out_dir, "source_map.json"), "w") as f:
        json.dump(source_map, f, indent=2)

    print("[INFO] Reloading best checkpoint for full export...")
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    export_outputs(
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        device=device,
        out_dir=args.out_dir,
        prefix=args.export_prefix,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    print("[DONE] Saved files:")
    print("  model1_best.pt")
    print("  train_config.json")
    print("  train_history.json")
    print("  source_map.json")
    print(f"  {args.export_prefix}_input.npy")
    print(f"  {args.export_prefix}_recon.npy")
    print(f"  {args.export_prefix}_latent.npy")
    print(f"  {args.export_prefix}_residual.npy")
    print(f"  {args.export_prefix}_recon_mse.npy")
    print(f"  {args.export_prefix}_source_index.npy")


if __name__ == "__main__":
    main()