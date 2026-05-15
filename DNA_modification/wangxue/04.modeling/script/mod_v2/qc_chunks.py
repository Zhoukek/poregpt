#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import math
import random
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_feature_names(path, fallback_dim=None):
    if path is not None and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            names = [x.strip() for x in f if x.strip()]
        if len(names) > 0:
            return names

    if fallback_dim is not None:
        return [f"feature_{i}" for i in range(fallback_dim)]

    return []


def load_summary_txt(path):
    if path is None or (not os.path.exists(path)):
        return {}

    summary = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                k, v = parts
                summary[k] = v
    return summary


def save_text_report(report_path, lines):
    with open(report_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def plot_hist(
    values,
    bins,
    title,
    xlabel,
    ylabel,
    out_path,
    logy=False,
):
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins)
    if logy:
        plt.yscale("log")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_random_chunks(chunks, out_path, n_show=12, seed=123):
    if len(chunks) == 0:
        return

    rng = random.Random(seed)
    n_show = min(n_show, len(chunks))
    idxs = rng.sample(range(len(chunks)), n_show)

    ncols = 3
    nrows = math.ceil(n_show / ncols)

    plt.figure(figsize=(5 * ncols, 3 * nrows))
    for i, idx in enumerate(idxs, start=1):
        ax = plt.subplot(nrows, ncols, i)
        ax.plot(chunks[idx])
        ax.set_title(f"chunk {idx}")
        ax.set_xlabel("position")
        ax.set_ylabel("value")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_feature_histograms(features, feature_names, out_dir, max_cols=3):
    if features.shape[0] == 0 or features.shape[1] == 0:
        return

    n_feat = features.shape[1]
    ncols = max_cols
    nrows = math.ceil(n_feat / ncols)

    plt.figure(figsize=(5 * ncols, 3.5 * nrows))
    for i in range(n_feat):
        ax = plt.subplot(nrows, ncols, i + 1)
        ax.hist(features[:, i], bins=100)
        name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
        ax.set_title(name)
        ax.set_xlabel("value")
        ax.set_ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "feature_histograms.png"), dpi=220)
    plt.close()


def plot_correlation_heatmap(features, feature_names, out_path):
    if features.shape[0] == 0 or features.shape[1] == 0:
        return

    corr = np.corrcoef(features.T)

    plt.figure(figsize=(10, 8))
    im = plt.imshow(corr, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    ticks = np.arange(features.shape[1])
    labels = [
        feature_names[i] if i < len(feature_names) else f"f{i}"
        for i in range(features.shape[1])
    ]
    plt.xticks(ticks, labels, rotation=90)
    plt.yticks(ticks, labels)
    plt.title("Feature correlation matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def pca_2d(X, n_samples=50000, seed=123):
    if len(X) == 0:
        return np.empty((0, 2), dtype=np.float32), None

    rng = np.random.default_rng(seed)
    if len(X) > n_samples:
        idx = rng.choice(len(X), size=n_samples, replace=False)
        X_use = X[idx]
    else:
        X_use = X

    X_use = X_use.astype(np.float64)
    X_center = X_use - X_use.mean(axis=0, keepdims=True)

    # SVD PCA
    U, S, Vt = np.linalg.svd(X_center, full_matrices=False)
    coords = X_center @ Vt[:2].T

    total_var = np.sum(np.var(X_center, axis=0, ddof=1))
    explained = (S[:2] ** 2) / (len(X_center) - 1) if len(X_center) > 1 else np.array([0.0, 0.0])
    explained_ratio = explained / total_var if total_var > 0 else np.array([0.0, 0.0])

    return coords.astype(np.float32), explained_ratio


def plot_pca(coords, explained_ratio, out_path):
    if len(coords) == 0:
        return

    plt.figure(figsize=(7, 6))
    plt.scatter(coords[:, 0], coords[:, 1], s=2, alpha=0.5)
    if explained_ratio is not None:
        xlabel = f"PC1 ({explained_ratio[0] * 100:.2f}%)"
        ylabel = f"PC2 ({explained_ratio[1] * 100:.2f}%)"
    else:
        xlabel = "PC1"
        ylabel = "PC2"
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("PCA of chunk features")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="QC statistics and visualization for chunk npy outputs."
    )
    parser.add_argument("--chunks", required=True, help="Path to xxx.chunks.npy")
    parser.add_argument("--features", required=True, help="Path to xxx.chunk_features.npy")
    parser.add_argument("--meta", required=True, help="Path to xxx.chunk_meta.npz")
    parser.add_argument("--feature_names", default=None, help="Path to xxx.feature_names.txt")
    parser.add_argument("--summary", default=None, help="Path to xxx.summary.txt")
    parser.add_argument("--out_dir", required=True, help="Directory to save QC results")
    parser.add_argument("--max_points_hist", type=int, default=2000000, help="Max flattened signal points used in histogram")
    parser.add_argument("--max_points_stats", type=int, default=5000000, help="Max flattened signal points used in global stats")
    parser.add_argument("--pca_samples", type=int, default=50000, help="Number of samples for PCA")
    parser.add_argument("--random_chunk_n", type=int, default=12, help="Number of random chunks to draw")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    print("Loading files...")
    chunks = np.load(args.chunks)
    features = np.load(args.features)
    meta = np.load(args.meta, allow_pickle=True)

    feature_names = load_feature_names(
        args.feature_names,
        fallback_dim=features.shape[1] if features.ndim == 2 else None
    )
    summary = load_summary_txt(args.summary)

    print(f"chunks shape   : {chunks.shape}")
    print(f"features shape : {features.shape}")
    print(f"meta keys      : {list(meta.keys())}")

    report_lines = []
    report_lines.append("=== QC REPORT ===")
    report_lines.append(f"chunks_path\t{args.chunks}")
    report_lines.append(f"features_path\t{args.features}")
    report_lines.append(f"meta_path\t{args.meta}")
    report_lines.append(f"n_chunks\t{chunks.shape[0]}")
    report_lines.append(f"chunk_len\t{chunks.shape[1] if chunks.ndim == 2 and chunks.shape[0] > 0 else 'NA'}")
    report_lines.append(f"n_features\t{features.shape[1] if features.ndim == 2 else 'NA'}")

    # -------- overall value stats --------
    if chunks.size > 0:
        flat_all_n = chunks.size
        rng = np.random.default_rng(args.seed)

        if flat_all_n > args.max_points_stats:
            idx = rng.choice(flat_all_n, size=args.max_points_stats, replace=False)
            flat_stats = chunks.reshape(-1)[idx]
        else:
            flat_stats = chunks.reshape(-1)

        report_lines.append("")
        report_lines.append("[global_signal_stats]")
        report_lines.append(f"n_signal_points_used\t{len(flat_stats)}")
        report_lines.append(f"min\t{np.min(flat_stats):.6f}")
        report_lines.append(f"max\t{np.max(flat_stats):.6f}")
        report_lines.append(f"mean\t{np.mean(flat_stats):.6f}")
        report_lines.append(f"std\t{np.std(flat_stats):.6f}")
        report_lines.append(f"median\t{np.median(flat_stats):.6f}")
        report_lines.append(f"p01\t{np.percentile(flat_stats, 1):.6f}")
        report_lines.append(f"p05\t{np.percentile(flat_stats, 5):.6f}")
        report_lines.append(f"p25\t{np.percentile(flat_stats, 25):.6f}")
        report_lines.append(f"p75\t{np.percentile(flat_stats, 75):.6f}")
        report_lines.append(f"p95\t{np.percentile(flat_stats, 95):.6f}")
        report_lines.append(f"p99\t{np.percentile(flat_stats, 99):.6f}")

        # hist sampling
        if flat_all_n > args.max_points_hist:
            idx = rng.choice(flat_all_n, size=args.max_points_hist, replace=False)
            flat_hist = chunks.reshape(-1)[idx]
        else:
            flat_hist = chunks.reshape(-1)

        plot_hist(
            flat_hist,
            bins=200,
            title="Normalized signal distribution",
            xlabel="value",
            ylabel="count",
            out_path=os.path.join(args.out_dir, "signal_histogram.png"),
            logy=False,
        )

        plot_hist(
            flat_hist,
            bins=200,
            title="Normalized signal distribution (log y)",
            xlabel="value",
            ylabel="count",
            out_path=os.path.join(args.out_dir, "signal_histogram_logy.png"),
            logy=True,
        )

        # chunk min/max
        chunk_min = chunks.min(axis=1)
        chunk_max = chunks.max(axis=1)
        chunk_mean = chunks.mean(axis=1)
        chunk_std = chunks.std(axis=1)

        report_lines.append("")
        report_lines.append("[chunk_level_stats]")
        report_lines.append(f"chunk_min_global\t{chunk_min.min():.6f}")
        report_lines.append(f"chunk_max_global\t{chunk_max.max():.6f}")
        report_lines.append(f"chunk_mean_mean\t{chunk_mean.mean():.6f}")
        report_lines.append(f"chunk_std_mean\t{chunk_std.mean():.6f}")

        plot_hist(
            chunk_min,
            bins=100,
            title="Chunk minimum distribution",
            xlabel="chunk min",
            ylabel="count",
            out_path=os.path.join(args.out_dir, "chunk_min_histogram.png"),
        )

        plot_hist(
            chunk_max,
            bins=100,
            title="Chunk maximum distribution",
            xlabel="chunk max",
            ylabel="count",
            out_path=os.path.join(args.out_dir, "chunk_max_histogram.png"),
        )

        plot_hist(
            chunk_mean,
            bins=100,
            title="Chunk mean distribution",
            xlabel="chunk mean",
            ylabel="count",
            out_path=os.path.join(args.out_dir, "chunk_mean_histogram.png"),
        )

        plot_hist(
            chunk_std,
            bins=100,
            title="Chunk std distribution",
            xlabel="chunk std",
            ylabel="count",
            out_path=os.path.join(args.out_dir, "chunk_std_histogram.png"),
        )

        # random chunks
        plot_random_chunks(
            chunks,
            out_path=os.path.join(args.out_dir, "random_chunks.png"),
            n_show=args.random_chunk_n,
            seed=args.seed,
        )

    # -------- feature stats --------
    if features.size > 0 and features.ndim == 2:
        report_lines.append("")
        report_lines.append("[feature_stats]")

        for i in range(features.shape[1]):
            name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
            col = features[:, i]
            report_lines.append(
                f"{name}\tmean={np.mean(col):.6f};std={np.std(col):.6f};"
                f"min={np.min(col):.6f};max={np.max(col):.6f};median={np.median(col):.6f}"
            )

        plot_feature_histograms(features, feature_names, args.out_dir)
        plot_correlation_heatmap(
            features,
            feature_names,
            out_path=os.path.join(args.out_dir, "feature_correlation_heatmap.png"),
        )

        coords, explained_ratio = pca_2d(features, n_samples=args.pca_samples, seed=args.seed)
        plot_pca(
            coords,
            explained_ratio,
            out_path=os.path.join(args.out_dir, "feature_pca.png"),
        )

    # -------- meta / read distribution --------
    if "read_ids" in meta:
        read_ids = meta["read_ids"]
        cnt = Counter(read_ids.tolist())
        chunks_per_read = np.array(list(cnt.values()), dtype=np.int64)

        report_lines.append("")
        report_lines.append("[read_level_stats]")
        report_lines.append(f"n_reads\t{len(cnt)}")
        report_lines.append(f"chunks_per_read_mean\t{np.mean(chunks_per_read):.6f}")
        report_lines.append(f"chunks_per_read_median\t{np.median(chunks_per_read):.6f}")
        report_lines.append(f"chunks_per_read_min\t{np.min(chunks_per_read)}")
        report_lines.append(f"chunks_per_read_max\t{np.max(chunks_per_read)}")
        report_lines.append(f"chunks_per_read_p95\t{np.percentile(chunks_per_read, 95):.6f}")
        report_lines.append(f"chunks_per_read_p99\t{np.percentile(chunks_per_read, 99):.6f}")

        plot_hist(
            chunks_per_read,
            bins=100,
            title="Chunks per read distribution",
            xlabel="chunks per read",
            ylabel="count",
            out_path=os.path.join(args.out_dir, "chunks_per_read_histogram.png"),
        )

    # -------- starts / trimmed_lengths --------
    if "starts" in meta:
        starts = meta["starts"]
        report_lines.append("")
        report_lines.append("[start_position_stats]")
        report_lines.append(f"start_min\t{np.min(starts)}")
        report_lines.append(f"start_max\t{np.max(starts)}")
        report_lines.append(f"start_mean\t{np.mean(starts):.6f}")
        report_lines.append(f"start_median\t{np.median(starts):.6f}")

        plot_hist(
            starts,
            bins=100,
            title="Chunk start distribution",
            xlabel="start position",
            ylabel="count",
            out_path=os.path.join(args.out_dir, "chunk_start_histogram.png"),
        )

    if "trimmed_lengths" in meta:
        trimmed_lengths = meta["trimmed_lengths"]
        report_lines.append("")
        report_lines.append("[trimmed_length_stats]")
        report_lines.append(f"trimmed_length_min\t{np.min(trimmed_lengths)}")
        report_lines.append(f"trimmed_length_max\t{np.max(trimmed_lengths)}")
        report_lines.append(f"trimmed_length_mean\t{np.mean(trimmed_lengths):.6f}")
        report_lines.append(f"trimmed_length_median\t{np.median(trimmed_lengths):.6f}")

        plot_hist(
            trimmed_lengths,
            bins=100,
            title="Trimmed read length distribution",
            xlabel="trimmed length",
            ylabel="count",
            out_path=os.path.join(args.out_dir, "trimmed_length_histogram.png"),
        )

    # -------- summary / drop ratio --------
    if len(summary) > 0:
        report_lines.append("")
        report_lines.append("[summary_txt]")
        for k, v in summary.items():
            report_lines.append(f"{k}\t{v}")

        if ("total_candidate_chunks" in summary) and ("dropped_chunks_out_of_range" in summary):
            try:
                total_candidate = float(summary["total_candidate_chunks"])
                dropped = float(summary["dropped_chunks_out_of_range"])
                drop_ratio = dropped / total_candidate if total_candidate > 0 else float("nan")
                report_lines.append(f"drop_ratio\t{drop_ratio:.6f}")
            except Exception:
                pass

    report_path = os.path.join(args.out_dir, "qc_report.txt")
    save_text_report(report_path, report_lines)

    print("\nQC finished.")
    print(f"Results saved to: {args.out_dir}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()