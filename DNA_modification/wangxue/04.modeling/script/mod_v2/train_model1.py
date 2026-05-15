#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import argparse
import bisect
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# =========================================================
# Utils
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_group_norm(num_channels: int) -> nn.GroupNorm:
    for g in [32, 16, 8, 4, 2, 1]:
        if num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)
    return nn.GroupNorm(1, num_channels)


# =========================================================
# Dataset
# =========================================================
class MultiChunkDataset(Dataset):
    """
    Read multiple chunks.npy files as one unified dataset.
    Each file is expected to have shape (N, L).

    IMPORTANT:
      This model is intended to be trained on canonical-only chunks.
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
        x = self.arrays[file_idx][local_idx].astype(np.float32)   # (L,)
        x = torch.from_numpy(x).unsqueeze(0)                      # (1, L)
        return x, file_idx


# =========================================================
# Model
# =========================================================
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
    """
    Task-oriented Model1:
      - structured latent: (B, latent_ch, latent_len)
      - no flatten -> preserves local structure
      - no skip connections
      - smoother decoder using interpolate + conv
    """
    def __init__(self, input_len=128, base_ch=32, latent_ch=32):
        super().__init__()

        if input_len != 128:
            raise ValueError(f"Current version is designed for input_len=128, got {input_len}")

        self.input_len = input_len
        self.base_ch = base_ch
        self.latent_ch = latent_ch
        self.latent_len = 16

        # Encoder
        self.stem = nn.Sequential(
            nn.Conv1d(1, base_ch, kernel_size=7, stride=1, padding=3, bias=False),
            make_group_norm(base_ch),
            nn.SiLU(),
        )  # (B, 1, 128) -> (B, base_ch, 128)

        self.down1 = nn.Sequential(
            nn.Conv1d(base_ch, base_ch * 2, kernel_size=7, stride=2, padding=3, bias=False),
            make_group_norm(base_ch * 2),
            nn.SiLU(),
        )  # 128 -> 64

        self.down2 = nn.Sequential(
            nn.Conv1d(base_ch * 2, base_ch * 4, kernel_size=7, stride=2, padding=3, bias=False),
            make_group_norm(base_ch * 4),
            nn.SiLU(),
        )  # 64 -> 32

        self.down3 = nn.Sequential(
            nn.Conv1d(base_ch * 4, base_ch * 4, kernel_size=7, stride=2, padding=3, bias=False),
            make_group_norm(base_ch * 4),
            nn.SiLU(),
        )  # 32 -> 16

        self.enc_res = nn.Sequential(
            ResidualBlock1D(base_ch * 4),
            ResidualBlock1D(base_ch * 4),
        )

        self.to_latent = nn.Sequential(
            nn.Conv1d(base_ch * 4, latent_ch, kernel_size=3, stride=1, padding=1, bias=False),
            make_group_norm(latent_ch),
            nn.SiLU(),
        )  # -> (B, latent_ch, 16)

        # Decoder
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
        )  # 16 -> 32

        self.up2 = nn.Sequential(
            nn.Conv1d(base_ch * 4, base_ch * 2, kernel_size=5, padding=2, bias=False),
            make_group_norm(base_ch * 2),
            nn.SiLU(),
        )  # 32 -> 64

        self.up3 = nn.Sequential(
            nn.Conv1d(base_ch * 2, base_ch, kernel_size=5, padding=2, bias=False),
            make_group_norm(base_ch),
            nn.SiLU(),
        )  # 64 -> 128

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
# Augmentation / Loss
# =========================================================
def augment_signal(
    x,
    noise_std=0.05,
    scale_std=0.05,
    mask_prob=0.30,
    mask_len=8,
):
    """
    Denoising-style augmentation:
      - additive gaussian noise
      - mild amplitude scaling
      - random local masking
    """
    x_aug = x.clone()

    if noise_std > 0:
        x_aug = x_aug + torch.randn_like(x_aug) * noise_std

    if scale_std > 0:
        scale = 1.0 + scale_std * torch.randn(x.size(0), 1, 1, device=x.device)
        x_aug = x_aug * scale

    if mask_prob > 0 and mask_len > 0:
        B, C, L = x_aug.shape
        for i in range(B):
            if torch.rand(1, device=x.device).item() < mask_prob:
                start = torch.randint(0, max(1, L - mask_len + 1), (1,), device=x.device).item()
                x_aug[i, :, start:start + mask_len] = 0.0

    return x_aug


def compute_losses(
    x_target,
    x_hat,
    residual,
    z,
    alpha_l1=0.5,
    alpha_smooth=0.1,
    alpha_latent=1e-4,
):
    """
    x_target: clean canonical target
    x_hat   : reconstructed canonical signal
    residual: x_input - x_hat
    z       : structured latent
    """
    recon_mse = F.mse_loss(x_hat, x_target)
    recon_l1 = F.l1_loss(x_hat, x_target)

    dx_hat = x_hat[:, :, 1:] - x_hat[:, :, :-1]
    dx_tgt = x_target[:, :, 1:] - x_target[:, :, :-1]
    smooth_l1 = F.l1_loss(dx_hat, dx_tgt)

    latent_reg = (z ** 2).mean()

    loss = recon_mse + alpha_l1 * recon_l1 + alpha_smooth * smooth_l1 + alpha_latent * latent_reg

    metrics = {
        "loss": loss.detach(),
        "recon_mse": recon_mse.detach(),
        "recon_l1": recon_l1.detach(),
        "smooth_l1": smooth_l1.detach(),
        "latent_reg": latent_reg.detach(),
        "residual_l1": residual.abs().mean().detach(),
    }
    return loss, metrics


# =========================================================
# Train / Eval
# =========================================================
def run_epoch(
    model,
    loader,
    optimizer,
    device,
    train=True,
    log_every=100,
    use_tqdm=True,
    debug_first_batch=False,
    noise_std=0.05,
    scale_std=0.05,
    mask_prob=0.30,
    mask_len=8,
    alpha_l1=0.5,
    alpha_smooth=0.1,
    alpha_latent=1e-4,
):
    stage = "train" if train else "val"
    model.train() if train else model.eval()

    total = {
        "loss": 0.0,
        "recon_mse": 0.0,
        "recon_l1": 0.0,
        "smooth_l1": 0.0,
        "latent_reg": 0.0,
        "residual_l1": 0.0,
    }
    total_num = 0
    t_epoch0 = time.time()

    num_batches = len(loader)
    print(f"[INFO] {stage} epoch start | num_batches={num_batches} | device={device}")

    iterable = loader
    if use_tqdm and tqdm is not None:
        iterable = tqdm(loader, desc=stage, leave=False, dynamic_ncols=True)

    first_batch_t0 = time.time()

    for step, (x_clean, _file_idx) in enumerate(iterable, start=1):
        t_fetch_done = time.time()

        if debug_first_batch and step == 1:
            print(f"[DEBUG] first batch fetched | x.shape={tuple(x_clean.shape)} | fetch_wait={t_fetch_done - first_batch_t0:.3f}s")

        x_clean = x_clean.to(device, non_blocking=(device.type == "cuda"))

        x_in = augment_signal(
            x_clean,
            noise_std=noise_std,
            scale_std=scale_std,
            mask_prob=mask_prob,
            mask_len=mask_len,
        )

        if train:
            optimizer.zero_grad(set_to_none=True)

        t0 = time.time()
        with torch.set_grad_enabled(train):
            x_hat, residual, z = model(x_in)

            if debug_first_batch and step == 1:
                print(
                    f"[DEBUG] first batch forward done | "
                    f"x_in={tuple(x_in.shape)} | x_hat={tuple(x_hat.shape)} | "
                    f"residual={tuple(residual.shape)} | z={tuple(z.shape)}"
                )

            loss, metrics = compute_losses(
                x_target=x_clean,
                x_hat=x_hat,
                residual=residual,
                z=z,
                alpha_l1=alpha_l1,
                alpha_smooth=alpha_smooth,
                alpha_latent=alpha_latent,
            )

            if train:
                loss.backward()
                if debug_first_batch and step == 1:
                    print("[DEBUG] first batch backward done")
                optimizer.step()
                if debug_first_batch and step == 1:
                    print("[DEBUG] first batch optimizer step done")

        batch_compute_time = time.time() - t0
        bs = x_clean.size(0)
        total_num += bs

        for k in total.keys():
            total[k] += float(metrics[k]) * bs

        avg_loss = total["loss"] / max(total_num, 1)

        if use_tqdm and tqdm is not None:
            iterable.set_postfix(loss=f"{avg_loss:.6f}")
        elif (step % log_every == 0) or (step == 1) or (step == num_batches):
            elapsed = time.time() - t_epoch0
            print(
                f"[INFO] {stage} step {step}/{num_batches} | "
                f"batch_size={bs} | batch_loss={float(metrics['loss']):.6f} | avg_loss={avg_loss:.6f} | "
                f"compute={batch_compute_time:.3f}s | elapsed={elapsed:.1f}s"
            )

    epoch_time = time.time() - t_epoch0
    avg = {k: total[k] / max(total_num, 1) for k in total.keys()}

    print(
        f"[INFO] {stage} epoch done | "
        f"loss={avg['loss']:.6f} | recon_mse={avg['recon_mse']:.6f} | "
        f"recon_l1={avg['recon_l1']:.6f} | smooth_l1={avg['smooth_l1']:.6f} | "
        f"latent_reg={avg['latent_reg']:.6f} | residual_l1={avg['residual_l1']:.6f} | "
        f"samples={total_num} | time={epoch_time:.1f}s"
    )
    return avg, epoch_time


# =========================================================
# Checkpoint helpers
# =========================================================
def build_ckpt_payload(
    epoch,
    model,
    optimizer,
    input_len,
    args,
    best_val_loss,
    current_val_loss,
):
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "input_len": input_len,
        "base_ch": args.base_ch,
        "latent_ch": args.latent_ch,
        "latent_len": model.latent_len,
        "best_val_loss": float(best_val_loss),
        "current_val_loss": float(current_val_loss),
        "chunk_files": args.chunks,
    }


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Train model1 as a canonical-only denoising residual autoencoder"
    )
    parser.add_argument(
        "--chunks",
        type=str,
        nargs="+",
        required=True,
        help="One or more canonical chunk .npy files"
    )
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")

    # model
    parser.add_argument("--base_ch", type=int, default=32, help="Base channel width")
    parser.add_argument("--latent_ch", type=int, default=32, help="Structured latent channels")

    # train
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Total epochs to train to")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # loss weights
    parser.add_argument("--alpha_l1", type=float, default=0.5, help="Pointwise L1 loss weight")
    parser.add_argument("--alpha_smooth", type=float, default=0.1, help="Waveform smoothness loss weight")
    parser.add_argument("--alpha_latent", type=float, default=1e-4, help="Latent regularization weight")

    # augmentation
    parser.add_argument("--noise_std", type=float, default=0.05, help="Gaussian noise std")
    parser.add_argument("--scale_std", type=float, default=0.05, help="Mild amplitude scaling std")
    parser.add_argument("--mask_prob", type=float, default=0.30, help="Random local mask probability")
    parser.add_argument("--mask_len", type=int, default=8, help="Random local mask length")

    # performance / debug
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--pin_memory", action="store_true", help="Enable DataLoader pin_memory")
    parser.add_argument("--log_every", type=int, default=100, help="Print every N batches when tqdm is unavailable")
    parser.add_argument("--no_tqdm", action="store_true", help="Disable tqdm progress bar")
    parser.add_argument("--debug_first_batch", action="store_true", help="Verbose debug for the first batch of each epoch")
    parser.add_argument("--max_chunks", type=int, default=0, help="Use only the first N chunks for quick debugging (0 = all)")

    # checkpoint saving
    parser.add_argument("--save_every", type=int, default=1, help="Save latest checkpoint every N epochs")
    parser.add_argument("--keep_epoch_ckpts", action="store_true", help="Keep per-epoch checkpoints")

    # resume
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    set_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        pin_memory = args.pin_memory
    else:
        device = torch.device("cpu")
        pin_memory = False

    print("=" * 80)
    print("[INFO] Training canonical residual autoencoder")
    print(f"[INFO] device         = {device}")
    print(f"[INFO] batch_size     = {args.batch_size}")
    print(f"[INFO] epochs         = {args.epochs}")
    print(f"[INFO] lr             = {args.lr}")
    print(f"[INFO] base_ch        = {args.base_ch}")
    print(f"[INFO] latent_ch      = {args.latent_ch}")
    print(f"[INFO] val_ratio      = {args.val_ratio}")
    print(f"[INFO] alpha_l1       = {args.alpha_l1}")
    print(f"[INFO] alpha_smooth   = {args.alpha_smooth}")
    print(f"[INFO] alpha_latent   = {args.alpha_latent}")
    print(f"[INFO] noise_std      = {args.noise_std}")
    print(f"[INFO] scale_std      = {args.scale_std}")
    print(f"[INFO] mask_prob      = {args.mask_prob}")
    print(f"[INFO] mask_len       = {args.mask_len}")
    print(f"[INFO] num_workers    = {args.num_workers}")
    print(f"[INFO] pin_memory     = {pin_memory}")
    print(f"[INFO] save_every     = {args.save_every}")
    print(f"[INFO] keep_epoch_ckpts = {args.keep_epoch_ckpts}")
    print(f"[INFO] resume         = {args.resume if args.resume else 'None'}")
    print(f"[INFO] tqdm_enabled   = {not args.no_tqdm and tqdm is not None}")
    print("[INFO] Loading chunk files:")
    for p in args.chunks:
        print("  ", p)
    print("=" * 80)

    dataset = MultiChunkDataset(args.chunks, verbose=True)
    input_len = dataset.input_len

    if input_len != 128:
        raise ValueError(f"[ERROR] This version requires input_len=128, got {input_len}")

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

    model = CanonicalResidualAutoencoder(
        input_len=input_len,
        base_ch=args.base_ch,
        latent_ch=args.latent_ch,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] model params = {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_ckpt = os.path.join(args.out_dir, "model1_best.pt")
    latest_ckpt = os.path.join(args.out_dir, "model1_latest.pt")

    history = []
    start_epoch = 1
    best_val = float("inf")

    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"[ERROR] Resume checkpoint not found: {args.resume}")

        print(f"[INFO] Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)

        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        start_epoch = int(ckpt["epoch"]) + 1
        best_val = float(ckpt.get("best_val_loss", float("inf")))

        print(f"[INFO] Resume loaded | start_epoch={start_epoch} | best_val={best_val:.6f}")

        if start_epoch > args.epochs:
            print(f"[WARN] start_epoch ({start_epoch}) > epochs ({args.epochs}). Nothing to train.")
            return

    for epoch in range(start_epoch, args.epochs + 1):
        print("-" * 80)
        print(f"[INFO] Epoch {epoch:03d}/{args.epochs}")

        train_avg, train_time = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            train=True,
            log_every=args.log_every,
            use_tqdm=(not args.no_tqdm),
            debug_first_batch=args.debug_first_batch,
            noise_std=args.noise_std,
            scale_std=args.scale_std,
            mask_prob=args.mask_prob,
            mask_len=args.mask_len,
            alpha_l1=args.alpha_l1,
            alpha_smooth=args.alpha_smooth,
            alpha_latent=args.alpha_latent,
        )

        val_avg, val_time = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=optimizer,
            device=device,
            train=False,
            log_every=args.log_every,
            use_tqdm=(not args.no_tqdm),
            debug_first_batch=args.debug_first_batch,
            noise_std=args.noise_std,
            scale_std=args.scale_std,
            mask_prob=args.mask_prob,
            mask_len=args.mask_len,
            alpha_l1=args.alpha_l1,
            alpha_smooth=args.alpha_smooth,
            alpha_latent=args.alpha_latent,
        )

        history.append({
            "epoch": epoch,
            "train": {k: float(v) for k, v in train_avg.items()},
            "val": {k: float(v) for k, v in val_avg.items()},
            "train_time_sec": float(train_time),
            "val_time_sec": float(val_time),
        })

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_avg['loss']:.6f} | val_loss={val_avg['loss']:.6f} | "
            f"val_recon_mse={val_avg['recon_mse']:.6f} | val_recon_l1={val_avg['recon_l1']:.6f}"
        )

        ckpt_payload = build_ckpt_payload(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            input_len=input_len,
            args=args,
            best_val_loss=best_val,
            current_val_loss=val_avg["loss"],
        )

        if epoch % args.save_every == 0:
            torch.save(ckpt_payload, latest_ckpt)
            print(f"[INFO] Latest checkpoint saved: {latest_ckpt}")

        if args.keep_epoch_ckpts and (epoch % args.save_every == 0):
            epoch_ckpt = os.path.join(args.out_dir, f"model1_epoch_{epoch:03d}.pt")
            torch.save(ckpt_payload, epoch_ckpt)
            print(f"[INFO] Epoch checkpoint saved: {epoch_ckpt}")

        if val_avg["loss"] < best_val:
            best_val = val_avg["loss"]

            best_payload = build_ckpt_payload(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                input_len=input_len,
                args=args,
                best_val_loss=best_val,
                current_val_loss=val_avg["loss"],
            )

            torch.save(best_payload, best_ckpt)
            print(f"[INFO] New best checkpoint saved: {best_ckpt}")

        # incremental write history/config so interruption still leaves useful files
        with open(os.path.join(args.out_dir, "train_history.json"), "w") as f:
            json.dump(history, f, indent=2)

        config = {
            "chunks": args.chunks,
            "out_dir": args.out_dir,
            "input_len": input_len,
            "base_ch": args.base_ch,
            "latent_ch": args.latent_ch,
            "latent_len": model.latent_len,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "alpha_l1": args.alpha_l1,
            "alpha_smooth": args.alpha_smooth,
            "alpha_latent": args.alpha_latent,
            "noise_std": args.noise_std,
            "scale_std": args.scale_std,
            "mask_prob": args.mask_prob,
            "mask_len": args.mask_len,
            "num_workers": args.num_workers,
            "pin_memory": pin_memory,
            "log_every": args.log_every,
            "tqdm_enabled": (not args.no_tqdm and tqdm is not None),
            "max_chunks": args.max_chunks,
            "save_every": args.save_every,
            "keep_epoch_ckpts": args.keep_epoch_ckpts,
            "resume": args.resume,
            "best_val_loss": float(best_val),
            "last_finished_epoch": epoch,
        }

        with open(os.path.join(args.out_dir, "train_config.json"), "w") as f:
            json.dump(config, f, indent=2)

    print("[INFO] Best val loss:", best_val)

    source_map = {i: p for i, p in enumerate(args.chunks)}
    with open(os.path.join(args.out_dir, "source_map.json"), "w") as f:
        json.dump(source_map, f, indent=2)

    print("[DONE] Saved files:")
    print("  model1_best.pt")
    print("  model1_latest.pt")
    print("  train_config.json")
    print("  train_history.json")
    print("  source_map.json")
    if args.keep_epoch_ckpts:
        print("  model1_epoch_XXX.pt (periodic snapshots)")


if __name__ == "__main__":
    main()
