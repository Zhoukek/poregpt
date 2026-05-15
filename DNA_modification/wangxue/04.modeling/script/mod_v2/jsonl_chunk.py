#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np


DEFAULT_FEATURE_NAMES = [
    "mean", "std", "median", "min", "max", "range", "iqr",
    "diff_mean", "diff_std", "diff_abs_mean",
    "diff2_mean", "diff2_std",
    "peak_to_peak", "energy"
]


def normalize_one_chunk(signal: np.ndarray) -> np.ndarray:
    """
    对单个 chunk 做 robust normalization:
        normalized = (signal - median) / MAD

    注意：
    1. 这里严格按你前面要求，不乘 1.4826
    2. 若 MAD 过小，则返回空数组，表示该 chunk 不可靠，后续丢弃
    """
    if signal.size == 0:
        return np.array([], dtype=np.float32)

    med = np.median(signal)
    mad = np.median(np.abs(signal - med))

    if mad < 1e-8:
        return np.array([], dtype=np.float32)

    normalized = (signal - med) / mad
    return normalized.astype(np.float32)


def make_normalized_chunks_with_filter(
    signal: np.ndarray,
    chunk_len: int = 128,
    stride: int = 64,
    value_min: float = -3.0,
    value_max: float = 3.0,
):
    """
    先切 chunk，再对每个 chunk 单独标准化。
    仅保留标准化后所有值都在 [value_min, value_max] 的 chunk。

    return:
        chunks: (N_kept, chunk_len)
        starts: (N_kept,)
        total_candidate_chunks: 候选 chunk 数
        dropped_chunks: 被丢弃 chunk 数
    """
    n = len(signal)
    if n < chunk_len:
        return (
            np.empty((0, chunk_len), dtype=np.float32),
            np.array([], dtype=np.int64),
            0,
            0,
        )

    starts = np.arange(0, n - chunk_len + 1, stride, dtype=np.int64)

    kept_chunks = []
    kept_starts = []
    total_candidate_chunks = len(starts)
    dropped_chunks = 0

    for s in starts:
        chunk = signal[s:s + chunk_len].astype(np.float32)
        chunk_norm = normalize_one_chunk(chunk)

        # MAD太小或异常，直接丢
        if chunk_norm.size == 0:
            dropped_chunks += 1
            continue

        # 只要有任意值超出范围，就丢整个 chunk
        if np.any(chunk_norm < value_min) or np.any(chunk_norm > value_max):
            dropped_chunks += 1
            continue

        kept_chunks.append(chunk_norm)
        kept_starts.append(s)

    if len(kept_chunks) == 0:
        return (
            np.empty((0, chunk_len), dtype=np.float32),
            np.array([], dtype=np.int64),
            total_candidate_chunks,
            dropped_chunks,
        )

    return (
        np.stack(kept_chunks, axis=0).astype(np.float32),
        np.array(kept_starts, dtype=np.int64),
        total_candidate_chunks,
        dropped_chunks,
    )


def compute_chunk_features(chunks: np.ndarray):
    """
    chunks: (N, L)
    return:
        features: (N, F)
        feature_names: list[str]
    """
    feature_names = DEFAULT_FEATURE_NAMES

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
    parser = argparse.ArgumentParser(description="Build chunk-level normalized chunks from jsonl signal data")
    parser.add_argument("--input_jsonl", required=True, help="Input jsonl file")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--chunk_len", type=int, default=128, help="Chunk length")
    parser.add_argument("--stride", type=int, default=64, help="Chunk stride")
    parser.add_argument("--min_len", type=int, default=128, help="Minimum signal length to keep")
    parser.add_argument("--value_min", type=float, default=-3.0, help="Min allowed normalized value in a chunk")
    parser.add_argument("--value_max", type=float, default=3.0, help="Max allowed normalized value in a chunk")
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
    n_skipped_no_chunk = 0
    total_candidate_chunks = 0
    dropped_chunks_out_of_range = 0
    total_chunks = 0

    feature_names = DEFAULT_FEATURE_NAMES

    with open(args.input_jsonl, "r", encoding="utf-8") as f:
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

            # 1) 先切 chunk，再对每个 chunk 单独标准化 + 过滤
            chunks, starts, total_candidate_chunks_this_read, dropped_chunks_this_read = make_normalized_chunks_with_filter(
                signal,
                chunk_len=args.chunk_len,
                stride=args.stride,
                value_min=args.value_min,
                value_max=args.value_max,
            )

            total_candidate_chunks += total_candidate_chunks_this_read
            dropped_chunks_out_of_range += dropped_chunks_this_read

            if len(chunks) == 0:
                n_skipped_no_chunk += 1
                continue

            # 2) features
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
        all_features = np.empty((0, len(DEFAULT_FEATURE_NAMES)), dtype=np.float32)

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
        value_min=np.array([args.value_min], dtype=np.float32),
        value_max=np.array([args.value_max], dtype=np.float32),
        input_jsonl=np.array([args.input_jsonl], dtype=object),
    )

    with open(os.path.join(args.out_dir, "feature_names.txt"), "w", encoding="utf-8") as f:
        for name in feature_names:
            f.write(name + "\n")

    with open(os.path.join(args.out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"input_jsonl\t{args.input_jsonl}\n")
        f.write(f"n_total_records\t{n_total}\n")
        f.write(f"n_kept_records\t{n_kept}\n")
        f.write(f"n_skipped_short\t{n_skipped_short}\n")
        f.write(f"n_skipped_no_chunk\t{n_skipped_no_chunk}\n")
        f.write(f"total_candidate_chunks\t{total_candidate_chunks}\n")
        f.write(f"dropped_chunks_out_of_range\t{dropped_chunks_out_of_range}\n")
        f.write(f"total_chunks\t{total_chunks}\n")
        f.write(f"chunk_len\t{args.chunk_len}\n")
        f.write(f"stride\t{args.stride}\n")
        f.write(f"min_len\t{args.min_len}\n")
        f.write(f"value_min\t{args.value_min}\n")
        f.write(f"value_max\t{args.value_max}\n")

    print("[DONE]")
    print("saved to:", args.out_dir)
    print("  chunks.npy         ", all_chunks.shape)
    print("  chunk_features.npy ", all_features.shape)
    print("  chunk_meta.npz")
    print("  feature_names.txt")
    print("  summary.txt")
    print(f"  total_candidate_chunks: {total_candidate_chunks}")
    print(f"  dropped_chunks_out_of_range: {dropped_chunks_out_of_range}")
    print(f"  kept_chunks: {total_chunks}")


if __name__ == "__main__":
    main()