#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import bisect

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# =========================================================
# Dataset
# =========================================================
class MultiChunkDataset(Dataset):
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
            print(
                f"[INFO] Dataset ready | files={len(self.arrays)} | "
                f"total_chunks={self.total_len} | input_len={self.input_len}"
            )

    def __len__(self):
        return self.total_len

    def _locate(self, idx):
        file_idx = bisect.bisect_right(self.cum_lengths, idx)
        prev_cum = 0 if file_idx == 0 else self.cum_lengths[file_idx - 1]
        local_idx = idx - prev_cum
        return file_idx, local_idx

    def __getitem__(self, idx):
        file_idx, local_idx = self._locate(idx)
        x = self.arrays[file_idx][local_idx].astype(np.float32)
        x = torch.from_numpy(x).unsqueeze(0)
        return x, file_idx


# =========================================================
# Model: must match train_model1.py
# =========================================================
def make_group_norm(num_channels: int) -> nn.GroupNorm:
    for g in [32, 16, 8, 4, 2, 1]:
        if num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)
    return nn.GroupNorm(1, num_channels)


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=2, bias=False),
            make_group_norm(channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2, bias=False),
            make_group_norm(channels),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.block(x))


class CanonicalResidualAutoencoder(nn.Module):
    def __init__(self, input_len=128, base_ch=32, latent_ch=32):
        super().__init__()

        if input_len != 128:
            raise ValueError(f"Current version is designed for input_len=128, got {input_len}")

        self.input_len = input_len
        self.base_ch = base_ch
        self.latent_ch = latent_ch
        self.latent_len = 16

        self.stem = nn.Sequential(
            nn.Conv1d(1, base_ch, kernel_size=7, stride=1, padding=3, bias=False),
            make_group_norm(base_ch),
            nn.SiLU(),
        )

        self.down1 = nn.Sequential(
            nn.Conv1d(base_ch, base_ch * 2, kernel_size=7, stride=2, padding=3, bias=False),
            make_group_norm(base_ch * 2),
            nn.SiLU(),
        )

        self.down2 = nn.Sequential(
            nn.Conv1d(base_ch * 2, base_ch * 4, kernel_size=7, stride=2, padding=3, bias=False),
            make_group_norm(base_ch * 4),
            nn.SiLU(),
        )

        self.down3 = nn.Sequential(
            nn.Conv1d(base_ch * 4, base_ch * 4, kernel_size=7, stride=2, padding=3, bias=False),
            make_group_norm(base_ch * 4),
            nn.SiLU(),
        )

        self.enc_res = nn.Sequential(
            ResidualBlock1D(base_ch * 4),
            ResidualBlock1D(base_ch * 4),
        )

        self.to_latent = nn.Sequential(
            nn.Conv1d(base_ch * 4, latent_ch, kernel_size=3, stride=1, padding=1, bias=False),
            make_group_norm(latent_ch),
            nn.SiLU(),
        )

        self.from_latent = nn.Sequential(
            nn.Conv1d(latent_ch, base_ch * 4, kernel_size=3, stride=1, padding=1, bias=False),
            make_group_norm(base_ch * 4),
            nn.SiLU(),
        )

        self.dec_res = nn.Sequential(
            ResidualBlock1D(base_ch * 4),
            ResidualBlock1D(base_ch * 4),
        )

        self.up1 = nn.Sequential(
            nn.Conv1d(base_ch * 4, base_ch * 4, kernel_size=5, padding=2, bias=False),
            make_group_norm(base_ch * 4),
            nn.SiLU(),
        )

        self.up2 = nn.Sequential(
            nn.Conv1d(base_ch * 4, base_ch * 2, kernel_size=5, padding=2, bias=False),
            make_group_norm(base_ch * 2),
            nn.SiLU(),
        )

        self.up3 = nn.Sequential(
            nn.Conv1d(base_ch * 2, base_ch, kernel_size=5, padding=2, bias=False),
            make_group_norm(base_ch),
            nn.SiLU(),
        )

        self.out_proj = nn.Conv1d(base_ch, 1, kernel_size=5, padding=2)

    def encode(self, x):
        h = self.stem(x)
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        h = self.enc_res(h)
        z = self.to_latent(h)
        return z

    def decode(self, z):
        h = self.from_latent(z)
        h = self.dec_res(h)

        h = F.interpolate(h, size=32, mode="linear", align_corners=False)
        h = self.up1(h)

        h = F.interpolate(h, size=64, mode="linear", align_corners=False)
        h = self.up2(h)

        h = F.interpolate(h, size=128, mode="linear", align_corners=False)
        h = self.up3(h)

        x_hat = self.out_proj(h)
        return x_hat

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        residual = x - x_hat
        return x_hat, residual, z


# =========================================================
# IO helpers
# =========================================================
def npy_memmap_writer(path, shape, dtype=np.float32):
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


@torch.no_grad()
def run_inference(
    model,
    dataset,
    batch_size,
    device,
    out_dir,
    prefix="model1",
    num_workers=0,
    pin_memory=False,
    save_input=True,
    save_recon=True,
    save_residual=True,
    save_latent=True,
    save_metrics=True,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )

    model.eval()

    n = len(dataset)
    input_len = dataset.dataset.input_len if isinstance(dataset, Subset) else dataset.input_len

    print(f"[INFO] inference start | samples={n} | batches={len(loader)}")

    mm_input = None
    mm_recon = None
    mm_residual = None
    mm_latent = None
    mm_mse = None
    mm_l1 = None
    mm_residual_mean = None
    mm_residual_std = None
    mm_residual_max_abs = None
    mm_latent_norm = None
    mm_source_index = None

    if save_input:
        mm_input = npy_memmap_writer(
            os.path.join(out_dir, f"{prefix}_input.npy"),
            shape=(n, input_len),
            dtype=np.float32,
        )

    if save_recon:
        mm_recon = npy_memmap_writer(
            os.path.join(out_dir, f"{prefix}_canonical_recon.npy"),
            shape=(n, input_len),
            dtype=np.float32,
        )

    if save_residual:
        mm_residual = npy_memmap_writer(
            os.path.join(out_dir, f"{prefix}_residual.npy"),
            shape=(n, input_len),
            dtype=np.float32,
        )

    if save_latent:
        mm_latent = npy_memmap_writer(
            os.path.join(out_dir, f"{prefix}_latent.npy"),
            shape=(n, model.latent_ch, model.latent_len),
            dtype=np.float32,
        )

    if save_metrics:
        mm_mse = npy_memmap_writer(
            os.path.join(out_dir, f"{prefix}_recon_mse.npy"),
            shape=(n,),
            dtype=np.float32,
        )
        mm_l1 = npy_memmap_writer(
            os.path.join(out_dir, f"{prefix}_recon_l1.npy"),
            shape=(n,),
            dtype=np.float32,
        )
        mm_residual_mean = npy_memmap_writer(
            os.path.join(out_dir, f"{prefix}_residual_mean.npy"),
            shape=(n,),
            dtype=np.float32,
        )
        mm_residual_std = npy_memmap_writer(
            os.path.join(out_dir, f"{prefix}_residual_std.npy"),
            shape=(n,),
            dtype=np.float32,
        )
        mm_residual_max_abs = npy_memmap_writer(
            os.path.join(out_dir, f"{prefix}_residual_max_abs.npy"),
            shape=(n,),
            dtype=np.float32,
        )
        mm_latent_norm = npy_memmap_writer(
            os.path.join(out_dir, f"{prefix}_latent_norm.npy"),
            shape=(n,),
            dtype=np.float32,
        )
        mm_source_index = npy_memmap_writer(
            os.path.join(out_dir, f"{prefix}_source_index.npy"),
            shape=(n,),
            dtype=np.int32,
        )

    iterable = loader
    if tqdm is not None:
        iterable = tqdm(loader, desc="infer", leave=False, dynamic_ncols=True)

    offset = 0

    for x, file_idx in iterable:
        x = x.to(device, non_blocking=(device.type == "cuda"))

        x_hat, residual, z = model(x)

        mse = ((x - x_hat) ** 2).mean(dim=(1, 2))
        l1 = (x - x_hat).abs().mean(dim=(1, 2))
        residual_mean = residual.mean(dim=(1, 2))
        residual_std = residual.std(dim=(1, 2))
        residual_max_abs = residual.abs().amax(dim=(1, 2))
        latent_norm = torch.sqrt((z ** 2).mean(dim=(1, 2)) + 1e-12)

        bs = x.size(0)
        sl = slice(offset, offset + bs)

        if mm_input is not None:
            mm_input[sl] = x[:, 0, :].detach().cpu().numpy().astype(np.float32)

        if mm_recon is not None:
            mm_recon[sl] = x_hat[:, 0, :].detach().cpu().numpy().astype(np.float32)

        if mm_residual is not None:
            mm_residual[sl] = residual[:, 0, :].detach().cpu().numpy().astype(np.float32)

        if mm_latent is not None:
            mm_latent[sl] = z.detach().cpu().numpy().astype(np.float32)

        if mm_mse is not None:
            mm_mse[sl] = mse.detach().cpu().numpy().astype(np.float32)
            mm_l1[sl] = l1.detach().cpu().numpy().astype(np.float32)
            mm_residual_mean[sl] = residual_mean.detach().cpu().numpy().astype(np.float32)
            mm_residual_std[sl] = residual_std.detach().cpu().numpy().astype(np.float32)
            mm_residual_max_abs[sl] = residual_max_abs.detach().cpu().numpy().astype(np.float32)
            mm_latent_norm[sl] = latent_norm.detach().cpu().numpy().astype(np.float32)
            mm_source_index[sl] = file_idx.numpy().astype(np.int32)

        offset += bs

    for arr in [
        mm_input,
        mm_recon,
        mm_residual,
        mm_latent,
        mm_mse,
        mm_l1,
        mm_residual_mean,
        mm_residual_std,
        mm_residual_max_abs,
        mm_latent_norm,
        mm_source_index,
    ]:
        if arr is not None:
            arr.flush()

    print("[INFO] inference done")


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Infer model1 canonical reconstruction / residual / latent from trained checkpoint"
    )

    parser.add_argument("--ckpt", type=str, required=True, help="Path to model1 checkpoint")
    parser.add_argument("--chunks", type=str, nargs="+", required=True, help="One or more chunk .npy files")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--prefix", type=str, default="model1", help="Output file prefix")

    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--max_chunks", type=int, default=0)

    parser.add_argument("--no_input", action="store_true")
    parser.add_argument("--no_recon", action="store_true")
    parser.add_argument("--no_residual", action="store_true")
    parser.add_argument("--no_latent", action="store_true")
    parser.add_argument("--no_metrics", action="store_true")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        pin_memory = args.pin_memory
    else:
        device = torch.device("cpu")
        pin_memory = False

    print("=" * 80)
    print("[INFO] Model1 inference")
    print(f"[INFO] device       = {device}")
    print(f"[INFO] ckpt         = {args.ckpt}")
    print(f"[INFO] batch_size   = {args.batch_size}")
    print(f"[INFO] num_workers  = {args.num_workers}")
    print(f"[INFO] pin_memory   = {pin_memory}")
    print(f"[INFO] out_dir      = {args.out_dir}")
    print(f"[INFO] prefix       = {args.prefix}")
    print("[INFO] chunks:")
    for p in args.chunks:
        print("  ", p)
    print("=" * 80)

    ckpt = torch.load(args.ckpt, map_location=device)

    input_len = int(ckpt.get("input_len", 128))
    base_ch = int(ckpt.get("base_ch", 32))
    latent_ch = int(ckpt.get("latent_ch", 32))

    print(f"[INFO] checkpoint epoch       = {ckpt.get('epoch', 'NA')}")
    print(f"[INFO] checkpoint best_val    = {ckpt.get('best_val_loss', 'NA')}")
    print(f"[INFO] checkpoint current_val = {ckpt.get('current_val_loss', 'NA')}")
    print(f"[INFO] input_len              = {input_len}")
    print(f"[INFO] base_ch                = {base_ch}")
    print(f"[INFO] latent_ch              = {latent_ch}")

    model = CanonicalResidualAutoencoder(
        input_len=input_len,
        base_ch=base_ch,
        latent_ch=latent_ch,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dataset = MultiChunkDataset(args.chunks, verbose=True)

    if dataset.input_len != input_len:
        raise ValueError(
            f"[ERROR] Dataset input_len={dataset.input_len}, checkpoint input_len={input_len}"
        )

    if args.max_chunks > 0:
        n_keep = min(args.max_chunks, len(dataset))
        dataset = Subset(dataset, list(range(n_keep)))
        print(f"[INFO] max_chunks active | using first {n_keep} chunks only")

    run_inference(
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        device=device,
        out_dir=args.out_dir,
        prefix=args.prefix,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        save_input=(not args.no_input),
        save_recon=(not args.no_recon),
        save_residual=(not args.no_residual),
        save_latent=(not args.no_latent),
        save_metrics=(not args.no_metrics),
    )

    infer_config = {
        "ckpt": args.ckpt,
        "chunks": args.chunks,
        "out_dir": args.out_dir,
        "prefix": args.prefix,
        "input_len": input_len,
        "base_ch": base_ch,
        "latent_ch": latent_ch,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "max_chunks": args.max_chunks,
        "save_input": not args.no_input,
        "save_recon": not args.no_recon,
        "save_residual": not args.no_residual,
        "save_latent": not args.no_latent,
        "save_metrics": not args.no_metrics,
    }

    with open(os.path.join(args.out_dir, f"{args.prefix}_infer_config.json"), "w") as f:
        json.dump(infer_config, f, indent=2)

    print("[DONE] Saved outputs:")
    if not args.no_input:
        print(f"  {args.prefix}_input.npy")
    if not args.no_recon:
        print(f"  {args.prefix}_canonical_recon.npy")
    if not args.no_residual:
        print(f"  {args.prefix}_residual.npy")
    if not args.no_latent:
        print(f"  {args.prefix}_latent.npy")
    if not args.no_metrics:
        print(f"  {args.prefix}_recon_mse.npy")
        print(f"  {args.prefix}_recon_l1.npy")
        print(f"  {args.prefix}_residual_mean.npy")
        print(f"  {args.prefix}_residual_std.npy")
        print(f"  {args.prefix}_residual_max_abs.npy")
        print(f"  {args.prefix}_latent_norm.npy")
        print(f"  {args.prefix}_source_index.npy")
    print(f"  {args.prefix}_infer_config.json")


if __name__ == "__main__":
    main()