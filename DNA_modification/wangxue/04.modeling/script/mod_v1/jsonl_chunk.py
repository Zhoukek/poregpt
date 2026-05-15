#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np


def nanopore_normalize(signal: np.ndarray) -> np.ndarray:
    """
    Robust normalization:
        x' = (x - median) / MAD
    """
    if signal.size == 0:
        return np.array([], dtype=np.float32)

    med = np.median(signal)
    mad = 1.4826 * np.median(np.abs(signal - med))
    mad = max(mad, 1.0)  # avoid division by near-zero

    normalized = (signal - med) / mad
    return normalized.astype(np.float32)


def make_chunks(signal: np.ndarray, chunk_len: int = 128, stride: int = 64):
    """
    signal: shape (L,)
    return:
        chunks: shape (N, chunk_len)
        starts: shape (N,)
    """
    n = len(signal)
    if n < chunk_len:
        return (
            np.empty((0, chunk_len), dtype=np.float32),
            np.array([], dtype=np.int64),
        )

    starts = np.arange(0, n - chunk_len + 1, stride, dtype=np.int64)
    chunks = np.stack([signal[s:s + chunk_len] for s in starts], axis=0).astype(np.float32)
    return chunks, starts


def compute_chunk_features(chunks: np.ndarray):
    """
    chunks: (N, L)
    return:
        features: (N, F)
        feature_names: list[str]
    """
    feature_names = [
        "mean", "std", "median", "min", "max", "range", "iqr",
        "diff_mean", "diff_std", "diff_abs_mean",
        "diff2_mean", "diff2_std",
        "peak_to_peak", "energy"
    ]

    if len(chunks) == 0:
        return np.empty((0, len(feature_names)), dtype=np.float32), feature_names

    x = chunks
    d1 = np.diff(x, axis=1)
    d2 = np.diff(d1, axis=1)

    mean = np.mean(x, axis=1)
    std = np.std(x, axis=1)
    median = np.median(x, axis=1)
    xmin = np.min(x, axis=1)
    xmax = np.max(x, axis=1)
    xrange = xmax - xmin
    q75 = np.percentile(x, 75, axis=1)
    q25 = np.percentile(x, 25, axis=1)
    iqr = q75 - q25

    diff_mean = np.mean(d1, axis=1)
    diff_std = np.std(d1, axis=1)
    diff_abs_mean = np.mean(np.abs(d1), axis=1)

    diff2_mean = np.mean(d2, axis=1)
    diff2_std = np.std(d2, axis=1)

    peak_to_peak = np.ptp(x, axis=1)
    energy = np.mean(x ** 2, axis=1)

    features = np.stack([
        mean, std, median, xmin, xmax, xrange, iqr,
        diff_mean, diff_std, diff_abs_mean,
        diff2_mean, diff2_std,
        peak_to_peak, energy
    ], axis=1).astype(np.float32)

    return features, feature_names


def main():
    parser = argparse.ArgumentParser(description="Build normalized chunks from jsonl signal data")
    parser.add_argument("--input_jsonl", required=True, help="Input jsonl file")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--chunk_len", type=int, default=128, help="Chunk length")
    parser.add_argument("--stride", type=int, default=64, help="Chunk stride")
    parser.add_argument("--min_len", type=int, default=128, help="Minimum signal length to keep")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    all_chunks = []
    all_features = []

    all_read_ids = []
    all_labels = []
    all_chunk_starts = []
    all_signal_lengths = []
    all_patterns = []

    n_total = 0
    n_kept = 0
    n_skipped_short = 0
    total_chunks = 0

    with open(args.input_jsonl, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            n_total += 1
            record = json.loads(line)

            read_id = record.get("read_id", f"read_{n_total}")
            label = record.get("label", None)
            pattern = record.get("pattern", None)
            signal = record.get("signal", None)

            if signal is None:
                continue

            signal = np.asarray(signal, dtype=np.float32)

            if len(signal) < args.min_len:
                n_skipped_short += 1
                continue

            # 1) normalize, no trim
            signal_processed = nanopore_normalize(signal)

            # 2) chunk
            chunks, starts = make_chunks(
                signal_processed,
                chunk_len=args.chunk_len,
                stride=args.stride
            )

            if len(chunks) == 0:
                n_skipped_short += 1
                continue

            # 3) features
            features, feature_names = compute_chunk_features(chunks)

            all_chunks.append(chunks)
            all_features.append(features)

            all_read_ids.extend([read_id] * len(chunks))
            all_labels.extend([label] * len(chunks))
            all_patterns.extend([pattern] * len(chunks))
            all_chunk_starts.extend(starts.tolist())
            all_signal_lengths.extend([len(signal)] * len(chunks))

            n_kept += 1
            total_chunks += len(chunks)

    if len(all_chunks) > 0:
        all_chunks = np.concatenate(all_chunks, axis=0).astype(np.float32)
        all_features = np.concatenate(all_features, axis=0).astype(np.float32)
    else:
        all_chunks = np.empty((0, args.chunk_len), dtype=np.float32)
        all_features = np.empty((0, 14), dtype=np.float32)
        feature_names = [
            "mean", "std", "median", "min", "max", "range", "iqr",
            "diff_mean", "diff_std", "diff_abs_mean",
            "diff2_mean", "diff2_std",
            "peak_to_peak", "energy"
        ]

    all_read_ids = np.array(all_read_ids, dtype=object)
    all_labels = np.array(all_labels, dtype=object)
    all_patterns = np.array(all_patterns, dtype=object)
    all_chunk_starts = np.array(all_chunk_starts, dtype=np.int64)
    all_signal_lengths = np.array(all_signal_lengths, dtype=np.int32)

    np.save(os.path.join(args.out_dir, "chunks.npy"), all_chunks)
    np.save(os.path.join(args.out_dir, "chunk_features.npy"), all_features)

    np.savez(
        os.path.join(args.out_dir, "chunk_meta.npz"),
        read_ids=all_read_ids,
        labels=all_labels,
        patterns=all_patterns,
        chunk_starts=all_chunk_starts,
        signal_lengths=all_signal_lengths,
        chunk_len=np.array([args.chunk_len], dtype=np.int32),
        stride=np.array([args.stride], dtype=np.int32),
        min_len=np.array([args.min_len], dtype=np.int32),
        input_jsonl=np.array([args.input_jsonl], dtype=object),
    )

    with open(os.path.join(args.out_dir, "feature_names.txt"), "w") as f:
        for name in feature_names:
            f.write(name + "\n")

    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write(f"input_jsonl\t{args.input_jsonl}\n")
        f.write(f"n_total_records\t{n_total}\n")
        f.write(f"n_kept_records\t{n_kept}\n")
        f.write(f"n_skipped_short\t{n_skipped_short}\n")
        f.write(f"total_chunks\t{total_chunks}\n")
        f.write(f"chunk_len\t{args.chunk_len}\n")
        f.write(f"stride\t{args.stride}\n")
        f.write(f"min_len\t{args.min_len}\n")

    print("[DONE]")
    print("saved to:", args.out_dir)
    print("  chunks.npy         ", all_chunks.shape)
    print("  chunk_features.npy ", all_features.shape)
    print("  chunk_meta.npz")
    print("  feature_names.txt")
    print("  summary.txt")


if __name__ == "__main__":
    main()
