#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score


try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# =========================================================
# Logging
# =========================================================
def log(msg):
    print(f"[INFO] {time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}", flush=True)


# =========================================================
# IO
# =========================================================
def load_npy(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return np.load(path, mmap_mode="r")


def take_first_or_random(arr, max_n, seed=42, random_sample=False):
    n = len(arr)

    if max_n <= 0 or n <= max_n:
        log(f"No subsampling | n={n}")
        return np.asarray(arr)

    log(f"Subsampling | original_n={n} | max_n={max_n} | random_sample={random_sample}")

    if random_sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_n, replace=False)
        idx = np.sort(idx)
        return np.asarray(arr[idx])
    else:
        # 对 mmap 文件，连续读取最快
        return np.asarray(arr[:max_n])


# =========================================================
# Feature construction
# =========================================================
def select_window(x, start=None, end=None):
    """
    x: (N, L)
    """
    if start is None or end is None:
        return x

    if start < 0 or end > x.shape[1] or start >= end:
        raise ValueError(f"Invalid window: start={start}, end={end}, residual_len={x.shape[1]}")

    return x[:, start:end]


def normalize_patterns(x, mode="zscore_per_chunk"):
    """
    x: (N, L)

    mode:
      none
      center_per_chunk
      zscore_per_chunk
      global_zscore
      abs
    """
    x = x.astype(np.float32)

    if mode == "none":
        return x

    if mode == "abs":
        return np.abs(x).astype(np.float32)

    if mode == "center_per_chunk":
        mean = x.mean(axis=1, keepdims=True)
        return (x - mean).astype(np.float32)

    if mode == "zscore_per_chunk":
        mean = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True) + 1e-6
        return ((x - mean) / std).astype(np.float32)

    if mode == "global_zscore":
        mean = x.mean()
        std = x.std() + 1e-6
        return ((x - mean) / std).astype(np.float32)

    raise ValueError(f"Unknown normalization mode: {mode}")


def make_features(residual, window_start=None, window_end=None, norm_mode="zscore_per_chunk"):
    """
    residual: (N, 128)
    return features: (N, W)
    """
    x = select_window(residual, window_start, window_end)
    x = normalize_patterns(x, norm_mode)
    return x


# =========================================================
# Statistics
# =========================================================
def state_frequency(labels, k):
    counts = np.bincount(labels, minlength=k).astype(np.float64)
    freq = counts / max(counts.sum(), 1)
    return counts, freq


def enrichment_table(c_labels, n_labels, k, eps=1e-9):
    c_counts, c_freq = state_frequency(c_labels, k)
    n_counts, n_freq = state_frequency(n_labels, k)

    rows = []

    for s in range(k):
        enrich = (n_freq[s] + eps) / (c_freq[s] + eps)
        diff = n_freq[s] - c_freq[s]

        rows.append({
            "state": s,
            "canonical_count": int(c_counts[s]),
            "native_count": int(n_counts[s]),
            "canonical_freq": float(c_freq[s]),
            "native_freq": float(n_freq[s]),
            "native_over_canonical": float(enrich),
            "native_minus_canonical": float(diff),
        })

    rows = sorted(rows, key=lambda r: r["native_over_canonical"], reverse=True)
    return rows, c_freq, n_freq


def save_enrichment_csv(rows, out_csv):
    header = [
        "state",
        "canonical_count",
        "native_count",
        "canonical_freq",
        "native_freq",
        "native_over_canonical",
        "native_minus_canonical",
    ]

    with open(out_csv, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(r[h]) for h in header) + "\n")


def js_divergence(p, q, eps=1e-12):
    """
    Jensen-Shannon divergence.
    """
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps

    p = p / p.sum()
    q = q / q.sum()

    m = 0.5 * (p + q)

    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))

    return float(0.5 * (kl_pm + kl_qm))


def chi_square_stat(c_counts, n_counts, eps=1e-9):
    """
    Simple chi-square-like statistic for distribution difference.
    Not used as a strict p-value, just magnitude indicator.
    """
    c = c_counts.astype(np.float64)
    n = n_counts.astype(np.float64)

    c = c / max(c.sum(), 1)
    n = n / max(n.sum(), 1)

    expected = 0.5 * (c + n) + eps
    stat = np.sum((c - expected) ** 2 / expected) + np.sum((n - expected) ** 2 / expected)

    return float(stat)


# =========================================================
# Plots
# =========================================================
def plot_state_frequency(c_freq, n_freq, out_png):
    states = np.arange(len(c_freq))
    width = 0.42

    plt.figure(figsize=(12, 5))
    plt.bar(states - width / 2, c_freq, width=width, label="canonical")
    plt.bar(states + width / 2, n_freq, width=width, label="native")
    plt.xlabel("Residual state")
    plt.ylabel("Frequency")
    plt.title("State frequency: canonical vs native")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_enrichment(rows, out_png, top_n=20):
    top_rows = rows[:top_n]
    states = [r["state"] for r in top_rows]
    enrich = [r["native_over_canonical"] for r in top_rows]

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(states)), enrich)
    plt.xticks(range(len(states)), states)
    plt.axhline(1.0, linestyle="--")
    plt.xlabel("State")
    plt.ylabel("Native / canonical frequency")
    plt.title(f"Top {top_n} native-enriched residual states")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_pca_scatter(z_pca, labels_group, out_png, max_points=50000, seed=42):
    """
    labels_group: 0 canonical, 1 native
    """
    n = len(z_pca)

    if n > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_points, replace=False)
        z_pca = z_pca[idx]
        labels_group = labels_group[idx]

    plt.figure(figsize=(6, 6))
    plt.scatter(
        z_pca[labels_group == 0, 0],
        z_pca[labels_group == 0, 1],
        s=2,
        alpha=0.25,
        label="canonical",
    )
    plt.scatter(
        z_pca[labels_group == 1, 0],
        z_pca[labels_group == 1, 1],
        s=2,
        alpha=0.25,
        label="native",
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Residual pattern PCA")
    plt.legend(markerscale=4)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_state_examples(features, labels, target_state, out_png, max_examples=50, seed=42):
    idx = np.where(labels == target_state)[0]

    if len(idx) == 0:
        return

    rng = np.random.default_rng(seed)

    if len(idx) > max_examples:
        idx = rng.choice(idx, size=max_examples, replace=False)

    plt.figure(figsize=(8, 4))

    for i in idx:
        plt.plot(features[i], alpha=0.25)

    plt.xlabel("Position in selected residual window")
    plt.ylabel("Normalized residual")
    plt.title(f"Residual examples for state {target_state}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="KMeans baseline for residual state discovery and canonical/native state enrichment."
    )

    parser.add_argument("--canonical_residual", required=True, help="canonical residual .npy, shape (N, 128)")
    parser.add_argument("--native_residual", required=True, help="native residual .npy, shape (N, 128)")
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--max_n", type=int, default=500000, help="Max chunks per group")
    parser.add_argument("--random_sample", action="store_true", help="Random sampling instead of first N rows")

    parser.add_argument("--window_start", type=int, default=-1, help="Optional residual window start")
    parser.add_argument("--window_end", type=int, default=-1, help="Optional residual window end")

    parser.add_argument(
        "--norm",
        type=str,
        default="zscore_per_chunk",
        choices=["none", "abs", "center_per_chunk", "zscore_per_chunk", "global_zscore"],
        help="Residual normalization mode"
    )

    parser.add_argument("--pca_dim", type=int, default=16, help="PCA dimension before KMeans")
    parser.add_argument("--n_states", type=int, default=32, help="Number of residual states")
    parser.add_argument("--batch_size", type=int, default=8192, help="MiniBatchKMeans batch size")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--top_enriched", type=int, default=10, help="Number of enriched states to plot examples")
    parser.add_argument("--example_per_state", type=int, default=50)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    log("=" * 80)
    log("Residual state KMeans evaluation")
    log(f"canonical_residual = {args.canonical_residual}")
    log(f"native_residual    = {args.native_residual}")
    log(f"out_dir            = {args.out_dir}")
    log(f"max_n              = {args.max_n}")
    log(f"random_sample      = {args.random_sample}")
    log(f"window_start       = {args.window_start}")
    log(f"window_end         = {args.window_end}")
    log(f"norm               = {args.norm}")
    log(f"pca_dim            = {args.pca_dim}")
    log(f"n_states           = {args.n_states}")
    log("=" * 80)

    # -----------------------------------------------------
    # Load residuals
    # -----------------------------------------------------
    c_mm = load_npy(args.canonical_residual)
    n_mm = load_npy(args.native_residual)

    log(f"canonical residual shape = {c_mm.shape}")
    log(f"native residual shape    = {n_mm.shape}")

    c_res = take_first_or_random(c_mm, args.max_n, seed=args.seed, random_sample=args.random_sample)
    n_res = take_first_or_random(n_mm, args.max_n, seed=args.seed + 1, random_sample=args.random_sample)

    log(f"canonical selected shape = {c_res.shape}")
    log(f"native selected shape    = {n_res.shape}")

    # -----------------------------------------------------
    # Feature construction
    # -----------------------------------------------------
    w_start = None if args.window_start < 0 else args.window_start
    w_end = None if args.window_end < 0 else args.window_end

    log("Constructing residual pattern features")
    c_feat = make_features(c_res, w_start, w_end, args.norm)
    n_feat = make_features(n_res, w_start, w_end, args.norm)

    log(f"canonical feature shape = {c_feat.shape}")
    log(f"native feature shape    = {n_feat.shape}")

    X = np.concatenate([c_feat, n_feat], axis=0).astype(np.float32)
    group = np.concatenate([
        np.zeros(len(c_feat), dtype=np.int8),
        np.ones(len(n_feat), dtype=np.int8),
    ])

    log(f"combined feature shape = {X.shape}")

    # -----------------------------------------------------
    # PCA
    # -----------------------------------------------------
    pca_dim = min(args.pca_dim, X.shape[1])
    log(f"Running PCA | dim={pca_dim}")

    pca = PCA(n_components=pca_dim, random_state=args.seed)
    X_pca = pca.fit_transform(X).astype(np.float32)

    explained = pca.explained_variance_ratio_.sum()
    log(f"PCA explained variance ratio sum = {explained:.4f}")

    np.save(os.path.join(args.out_dir, "residual_pca_features.npy"), X_pca)
    np.save(os.path.join(args.out_dir, "group_labels.npy"), group)

    # -----------------------------------------------------
    # KMeans
    # -----------------------------------------------------
    log(f"Running MiniBatchKMeans | n_states={args.n_states}")

    kmeans = MiniBatchKMeans(
        n_clusters=args.n_states,
        random_state=args.seed,
        batch_size=args.batch_size,
        n_init="auto",
        max_iter=300,
        verbose=0,
    )

    state_labels = kmeans.fit_predict(X_pca)

    c_labels = state_labels[group == 0]
    n_labels = state_labels[group == 1]

    np.save(os.path.join(args.out_dir, "state_labels.npy"), state_labels)
    np.save(os.path.join(args.out_dir, "canonical_state_labels.npy"), c_labels)
    np.save(os.path.join(args.out_dir, "native_state_labels.npy"), n_labels)

    # -----------------------------------------------------
    # Enrichment
    # -----------------------------------------------------
    log("Computing state enrichment")
    rows, c_freq, n_freq = enrichment_table(c_labels, n_labels, args.n_states)

    c_counts, _ = state_frequency(c_labels, args.n_states)
    n_counts, _ = state_frequency(n_labels, args.n_states)

    jsd = js_divergence(c_freq, n_freq)
    chi_like = chi_square_stat(c_counts, n_counts)

    log(f"Jensen-Shannon divergence = {jsd:.6f}")
    log(f"Chi-square-like statistic = {chi_like:.6f}")

    enrich_csv = os.path.join(args.out_dir, "state_enrichment.csv")
    save_enrichment_csv(rows, enrich_csv)

    # -----------------------------------------------------
    # Plots
    # -----------------------------------------------------
    log("Writing plots")

    plot_state_frequency(
        c_freq,
        n_freq,
        os.path.join(args.out_dir, "state_frequency.png"),
    )

    plot_enrichment(
        rows,
        os.path.join(args.out_dir, "top_native_enriched_states.png"),
        top_n=min(20, args.n_states),
    )

    plot_pca_scatter(
        X_pca[:, :2],
        group,
        os.path.join(args.out_dir, "residual_pca_scatter.png"),
        max_points=50000,
        seed=args.seed,
    )

    # examples for top enriched states
    top_states = [r["state"] for r in rows[:args.top_enriched]]

    for s in top_states:
        plot_state_examples(
            X,
            state_labels,
            target_state=s,
            out_png=os.path.join(args.out_dir, f"state_{s:03d}_examples.png"),
            max_examples=args.example_per_state,
            seed=args.seed,
        )

    # -----------------------------------------------------
    # Save summary
    # -----------------------------------------------------
    summary = {
        "canonical_residual": args.canonical_residual,
        "native_residual": args.native_residual,
        "max_n": args.max_n,
        "random_sample": args.random_sample,
        "window_start": args.window_start,
        "window_end": args.window_end,
        "norm": args.norm,
        "pca_dim": pca_dim,
        "pca_explained_variance_sum": float(explained),
        "n_states": args.n_states,
        "js_divergence": jsd,
        "chi_square_like_stat": chi_like,
        "top_native_enriched_states": rows[:args.top_enriched],
    }

    import json
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log("[DONE]")
    log(f"Saved: {enrich_csv}")
    log(f"Saved: {os.path.join(args.out_dir, 'state_frequency.png')}")
    log(f"Saved: {os.path.join(args.out_dir, 'top_native_enriched_states.png')}")
    log(f"Saved: {os.path.join(args.out_dir, 'residual_pca_scatter.png')}")
    log(f"Saved: {os.path.join(args.out_dir, 'summary.json')}")


if __name__ == "__main__":
    main()