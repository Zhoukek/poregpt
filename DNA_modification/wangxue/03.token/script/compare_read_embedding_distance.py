#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import mannwhitneyu


# =========================================================
# I/O
# =========================================================
def load_and_concat(paths):
    arrs = []
    for p in paths:
        x = np.load(p)
        if x.ndim != 2:
            raise ValueError(f"{p} shape={x.shape}, expected 2D array [N, D]")
        print(f"Loaded {p}, shape={x.shape}")
        arrs.append(x)

    if not arrs:
        raise ValueError("No input files found.")

    dims = {x.shape[1] for x in arrs}
    if len(dims) != 1:
        raise ValueError(f"Embedding dims do not match: {[x.shape for x in arrs]}")

    return np.concatenate(arrs, axis=0)


def ensure_outdir(outdir):
    os.makedirs(outdir, exist_ok=True)


# =========================================================
# Distance
# =========================================================
def cosine_distance(a, b, eps=1e-8):
    num = np.sum(a * b, axis=-1)
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + eps
    return 1.0 - num / denom


def euclidean_distance(a, b):
    return np.linalg.norm(a - b, axis=-1)


def pair_distance(a, b, metric="cosine"):
    if metric == "cosine":
        return cosine_distance(a, b)
    elif metric == "euclidean":
        return euclidean_distance(a, b)
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def distance_to_centroid(X, centroid, metric="cosine"):
    c = np.repeat(centroid[None, :], len(X), axis=0)
    return pair_distance(X, c, metric=metric)


# =========================================================
# Sampling
# =========================================================
def sample_within(X, n_pairs=20000, metric="cosine", seed=42):
    if len(X) < 2:
        raise ValueError("Within-group distance requires at least 2 reads.")

    rng = np.random.default_rng(seed)

    i = rng.integers(0, len(X), size=n_pairs)
    j = rng.integers(0, len(X), size=n_pairs)

    mask = i != j
    i = i[mask]
    j = j[mask]

    if len(i) == 0:
        i = np.array([0])
        j = np.array([1])

    a = X[i]
    b = X[j]
    return pair_distance(a, b, metric=metric)


def sample_between(X1, X2, n_pairs=20000, metric="cosine", seed=42):
    if len(X1) < 1 or len(X2) < 1:
        raise ValueError("Between-group distance requires both groups to be non-empty.")

    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(X1), size=n_pairs)
    j = rng.integers(0, len(X2), size=n_pairs)

    a = X1[i]
    b = X2[j]
    return pair_distance(a, b, metric=metric)


def balanced_downsample(X1, X2, seed=123):
    """
    为了 PCA 可视化公平性，可选地平衡两组数量。
    """
    n = min(len(X1), len(X2))
    rng = np.random.default_rng(seed)

    idx1 = rng.choice(len(X1), size=n, replace=False) if len(X1) > n else np.arange(len(X1))
    idx2 = rng.choice(len(X2), size=n, replace=False) if len(X2) > n else np.arange(len(X2))

    return X1[idx1], X2[idx2]


def downsample_for_points(arr, max_points=1500, seed=123):
    if len(arr) <= max_points:
        return arr
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(arr), size=max_points, replace=False)
    return arr[idx]


# =========================================================
# Stats
# =========================================================
def summarize(name, arr):
    arr = np.asarray(arr)
    return {
        "type": name,
        "count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(np.median(arr)),
        "q05": float(np.quantile(arr, 0.05)),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
        "q95": float(np.quantile(arr, 0.95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def safe_mwu(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    try:
        return mannwhitneyu(a, b, alternative="two-sided").pvalue
    except Exception:
        return np.nan


def cliffs_delta(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    n_x = len(x)
    n_y = len(y)
    if n_x == 0 or n_y == 0:
        return np.nan

    gt = 0
    lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)
    return (gt - lt) / (n_x * n_y)


def compute_effect_size(arr_a, arr_b):
    return {
        "mean_diff": float(np.mean(arr_a) - np.mean(arr_b)),
        "median_diff": float(np.median(arr_a) - np.median(arr_b)),
        "cliffs_delta": float(cliffs_delta(arr_a, arr_b)),
    }


def benjamini_hochberg(pvals):
    """
    Benjamini-Hochberg FDR correction.
    不依赖 statsmodels
    """
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)

    p = pvals.copy()
    nan_mask = np.isnan(p)
    p[nan_mask] = 1.0

    order = np.argsort(p)
    ranked_p = p[order]

    q = np.empty(n, dtype=float)
    prev = 1.0

    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked_p[i] * n / rank
        prev = min(prev, val)
        q[i] = prev

    q = np.clip(q, 0, 1)

    qvals = np.empty(n, dtype=float)
    qvals[order] = q
    qvals[nan_mask] = np.nan
    return qvals


def fmt_p(p):
    if np.isnan(p):
        return "p = NA"
    if p < 1e-10:
        return "p < 1e-10"
    elif p < 1e-4:
        return "p < 1e-4"
    return f"p = {p:.2e}"


# =========================================================
# Bootstrap
# =========================================================
def bootstrap_distance_summary(X1, X2, metric="cosine", n_pairs=10000, n_boot=100, seed=2024):
    rows = []
    for b in range(n_boot):
        d1 = sample_within(X1, n_pairs=n_pairs, metric=metric, seed=seed + b * 3 + 1)
        d2 = sample_within(X2, n_pairs=n_pairs, metric=metric, seed=seed + b * 3 + 2)
        d12 = sample_between(X1, X2, n_pairs=n_pairs, metric=metric, seed=seed + b * 3 + 3)

        rows.append({
            "bootstrap_id": b,
            "type": "within_group1",
            "mean": float(np.mean(d1)),
            "median": float(np.median(d1)),
            "std": float(np.std(d1, ddof=1)) if len(d1) > 1 else 0.0
        })
        rows.append({
            "bootstrap_id": b,
            "type": "within_group2",
            "mean": float(np.mean(d2)),
            "median": float(np.median(d2)),
            "std": float(np.std(d2, ddof=1)) if len(d2) > 1 else 0.0
        })
        rows.append({
            "bootstrap_id": b,
            "type": "between_groups",
            "mean": float(np.mean(d12)),
            "median": float(np.median(d12)),
            "std": float(np.std(d12, ddof=1)) if len(d12) > 1 else 0.0
        })

        rows.append({
            "bootstrap_id": b,
            "type": "delta_between_minus_within1",
            "mean": float(np.mean(d12) - np.mean(d1)),
            "median": float(np.median(d12) - np.median(d1)),
            "std": np.nan
        })
        rows.append({
            "bootstrap_id": b,
            "type": "delta_between_minus_within2",
            "mean": float(np.mean(d12) - np.mean(d2)),
            "median": float(np.median(d12) - np.median(d2)),
            "std": np.nan
        })

    return pd.DataFrame(rows)


# =========================================================
# PCA
# =========================================================
def pca_2d(X):
    """
    纯 numpy PCA，避免额外依赖 sklearn
    X: (N, D)
    """
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu

    # SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    coords = Xc @ Vt[:2].T

    var = (S ** 2) / max(1, (len(X) - 1))
    var_ratio = var / (var.sum() + 1e-12)
    explained = var_ratio[:2]

    return coords, explained


# =========================================================
# Plot helpers
# =========================================================
def add_pvalue_bar(ax, x1, x2, y, h, text, lw=1.4, fontsize=11):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", lw=lw)
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=fontsize)


def ecdf(arr):
    x = np.sort(arr)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def plot_violin_box(ax, data, labels, pvals, metric, max_points=1200, title="Distance comparison"):
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    positions = [1, 2, 3]

    vp = ax.violinplot(
        data,
        positions=positions,
        widths=0.85,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )

    for body, color in zip(vp["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.45)
        body.set_linewidth(1.0)

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.22,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=2.0),
        whiskerprops=dict(color="black", linewidth=1.2),
        capprops=dict(color="black", linewidth=1.2),
        boxprops=dict(color="black", linewidth=1.2),
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    rng = np.random.default_rng(12345)
    for i, arr in enumerate(data, start=1):
        y = downsample_for_points(arr, max_points=max_points, seed=100 + i)
        x = rng.normal(loc=i, scale=0.045, size=len(y))
        ax.scatter(x, y, s=7, alpha=0.12, color="black", linewidths=0, zorder=3)

    medians = [np.median(x) for x in data]
    ns = [len(x) for x in data]
    for pos, med, n in zip(positions, medians, ns):
        ax.text(pos, med, f"median={med:.3f}\nn={n}", ha="center", va="bottom", fontsize=9)

    ymax = max(float(np.max(x)) for x in data)
    ymin = min(float(np.min(x)) for x in data)
    yr = ymax - ymin if ymax > ymin else 1.0

    add_pvalue_bar(ax, 1, 2, ymax + 0.05 * yr, 0.02 * yr, fmt_p(pvals["1_vs_2"]))
    add_pvalue_bar(ax, 1, 3, ymax + 0.14 * yr, 0.02 * yr, fmt_p(pvals["1_vs_3"]))
    add_pvalue_bar(ax, 2, 3, ymax + 0.23 * yr, 0.02 * yr, fmt_p(pvals["2_vs_3"]))

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=12)
    ax.set_ylabel(f"{metric.capitalize()} distance")
    ax.set_title(title, fontsize=14, weight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_ylim(ymin - 0.05 * yr, ymax + 0.34 * yr)


def plot_cdf(ax, data, labels, metric, title="Distance CDF"):
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for arr, label, color in zip(data, labels, colors):
        x, y = ecdf(arr)
        ax.plot(x, y, linewidth=2.0, label=label, color=color)

    ax.set_xlabel(f"{metric.capitalize()} distance")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title(title, fontsize=13, weight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(frameon=False, loc="best")


def plot_hist(ax, data, labels, metric, title="Distance histogram"):
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for arr, label, color in zip(data, labels, colors):
        ax.hist(arr, bins=60, alpha=0.35, density=True, label=label, color=color, edgecolor="white")

    ax.set_xlabel(f"{metric.capitalize()} distance")
    ax.set_ylabel("Density")
    ax.set_title(title, fontsize=13, weight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(frameon=False, loc="best")


def plot_bootstrap(ax, boot_df, value_col="mean", title="Bootstrap stability"):
    color_map = {
        "within_group1": "#4C72B0",
        "within_group2": "#DD8452",
        "between_groups": "#55A868",
        "delta_between_minus_within1": "#C44E52",
        "delta_between_minus_within2": "#8172B2",
    }

    types = [
        "within_group1",
        "within_group2",
        "between_groups",
        "delta_between_minus_within1",
        "delta_between_minus_within2"
    ]

    data = []
    colors = []
    for t in types:
        vals = boot_df.loc[boot_df["type"] == t, value_col].dropna().values
        data.append(vals)
        colors.append(color_map[t])

    bp = ax.boxplot(
        data,
        patch_artist=True,
        showfliers=False,
        widths=0.65,
        medianprops=dict(color="black", linewidth=1.8),
        whiskerprops=dict(color="black", linewidth=1.1),
        capprops=dict(color="black", linewidth=1.1),
        boxprops=dict(color="black", linewidth=1.1),
    )

    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)

    ax.set_xticks(np.arange(1, len(types) + 1))
    ax.set_xticklabels(types, rotation=20, ha="right")
    ax.set_ylabel(value_col)
    ax.set_title(title, fontsize=13, weight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_pca(ax, X1, X2, name1, name2, max_points=3000):
    X1p, X2p = balanced_downsample(X1, X2, seed=123)

    if len(X1p) > max_points:
        idx = np.random.default_rng(1).choice(len(X1p), size=max_points, replace=False)
        X1p = X1p[idx]
    if len(X2p) > max_points:
        idx = np.random.default_rng(2).choice(len(X2p), size=max_points, replace=False)
        X2p = X2p[idx]

    X = np.vstack([X1p, X2p])
    coords, explained = pca_2d(X)

    n1 = len(X1p)
    c1 = coords[:n1]
    c2 = coords[n1:]

    ax.scatter(c1[:, 0], c1[:, 1], s=8, alpha=0.35, label=name1, color="#4C72B0", linewidths=0)
    ax.scatter(c2[:, 0], c2[:, 1], s=8, alpha=0.35, label=name2, color="#DD8452", linewidths=0)

    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
    ax.set_title("Read embedding PCA", fontsize=13, weight="bold")
    ax.grid(True, linestyle="--", alpha=0.20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="best")


def plot_centroid_distance(ax, centroid_df, metric, title="Read-to-centroid distance"):
    color_map = {
        centroid_df["type"].unique()[0]: "#4C72B0",
        centroid_df["type"].unique()[1]: "#DD8452",
        centroid_df["type"].unique()[2]: "#55A868",
        centroid_df["type"].unique()[3]: "#C44E52",
    }

    types = list(centroid_df["type"].unique())
    data = [centroid_df.loc[centroid_df["type"] == t, "distance"].values for t in types]
    colors = [color_map[t] for t in types]

    bp = ax.boxplot(
        data,
        patch_artist=True,
        showfliers=False,
        widths=0.65,
        medianprops=dict(color="black", linewidth=1.8),
        whiskerprops=dict(color="black", linewidth=1.1),
        capprops=dict(color="black", linewidth=1.1),
        boxprops=dict(color="black", linewidth=1.1),
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)

    ax.set_xticks(np.arange(1, len(types) + 1))
    ax.set_xticklabels(types, rotation=15, ha="right")
    ax.set_ylabel(f"{metric.capitalize()} distance")
    ax.set_title(title, fontsize=13, weight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_effect_summary(ax, effect_df):
    """
    简单文本 summary panel
    """
    ax.axis("off")
    lines = ["Effect size summary", ""]

    for _, r in effect_df.iterrows():
        lines.append(f"{r['comparison']}")
        lines.append(f"  mean_diff   = {r['mean_diff']:.4f}")
        lines.append(f"  median_diff = {r['median_diff']:.4f}")
        lines.append(f"  cliffs_delta= {r['cliffs_delta']:.4f}")
        lines.append(f"  pvalue      = {r['pvalue']:.3e}" if not pd.isna(r["pvalue"]) else "  pvalue      = NA")
        lines.append(f"  fdr         = {r['fdr']:.3e}" if not pd.isna(r["fdr"]) else "  fdr         = NA")
        lines.append("")

    ax.text(
        0.01, 0.99, "\n".join(lines),
        va="top", ha="left",
        fontsize=11, family="monospace"
    )


# =========================================================
# Main analysis
# =========================================================
def build_centroid_distance_table(X1, X2, metric, name1, name2):
    c1 = X1.mean(axis=0)
    c2 = X2.mean(axis=0)

    rows = []

    d_11 = distance_to_centroid(X1, c1, metric=metric)
    d_12 = distance_to_centroid(X1, c2, metric=metric)
    d_22 = distance_to_centroid(X2, c2, metric=metric)
    d_21 = distance_to_centroid(X2, c1, metric=metric)

    for v in d_11:
        rows.append({"type": f"{name1}->self_centroid", "distance": v})
    for v in d_12:
        rows.append({"type": f"{name1}->{name2}_centroid", "distance": v})
    for v in d_22:
        rows.append({"type": f"{name2}->self_centroid", "distance": v})
    for v in d_21:
        rows.append({"type": f"{name2}->{name1}_centroid", "distance": v})

    return pd.DataFrame(rows)


def write_summary_txt(summary_df, pval_df, effect_df, centroid_df, out_txt):
    lines = []
    lines.append("=== Distance summary ===")
    lines.append(summary_df.to_string(index=False))
    lines.append("")

    lines.append("=== P-values and FDR ===")
    lines.append(pval_df.to_string(index=False))
    lines.append("")

    lines.append("=== Effect sizes ===")
    lines.append(effect_df.to_string(index=False))
    lines.append("")

    lines.append("=== Read-to-centroid summary ===")
    csum = centroid_df.groupby("type")["distance"].agg(["count", "mean", "median", "std", "min", "max"])
    lines.append(csum.to_string())
    lines.append("")

    with open(out_txt, "w") as f:
        f.write("\n".join(lines))


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive read-level embedding comparison"
    )
    parser.add_argument("--group1", nargs="+", required=True, help="One or more .npy files for group1")
    parser.add_argument("--group2", nargs="+", required=True, help="One or more .npy files for group2")
    parser.add_argument("--name1", default="Group1")
    parser.add_argument("--name2", default="Group2")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--n_pairs", type=int, default=20000)
    parser.add_argument("--n_plot_points", type=int, default=1200)
    parser.add_argument("--n_boot", type=int, default=100)
    parser.add_argument("--bootstrap_pairs", type=int, default=10000)
    parser.add_argument("--pca_max_points", type=int, default=3000)
    args = parser.parse_args()

    ensure_outdir(args.outdir)

    X1 = load_and_concat(args.group1)
    X2 = load_and_concat(args.group2)

    print(f"\nFinal merged shapes:")
    print(f"{args.name1}: {X1.shape}")
    print(f"{args.name2}: {X2.shape}")

    # -----------------------------------------------------
    # Main sampled distributions
    # -----------------------------------------------------
    d1 = sample_within(X1, n_pairs=args.n_pairs, metric=args.metric, seed=42)
    d2 = sample_within(X2, n_pairs=args.n_pairs, metric=args.metric, seed=43)
    d12 = sample_between(X1, X2, n_pairs=args.n_pairs, metric=args.metric, seed=44)

    label1 = f"Within {args.name1}"
    label2 = f"Within {args.name2}"
    label3 = f"Between {args.name1}-{args.name2}"

    data = [d1, d2, d12]
    labels = [label1, label2, label3]

    sample_df = pd.DataFrame({
        "distance": np.concatenate(data),
        "type": [label1] * len(d1) + [label2] * len(d2) + [label3] * len(d12)
    })
    sample_csv = os.path.join(args.outdir, f"sampled_distances_{args.metric}.csv")
    sample_df.to_csv(sample_csv, index=False)

    summary_df = pd.DataFrame([
        summarize(label1, d1),
        summarize(label2, d2),
        summarize(label3, d12),
    ])
    summary_csv = os.path.join(args.outdir, f"distance_summary_{args.metric}.csv")
    summary_df.to_csv(summary_csv, index=False)

    # -----------------------------------------------------
    # p-values / FDR
    # -----------------------------------------------------
    p12 = safe_mwu(d1, d2)
    p13 = safe_mwu(d1, d12)
    p23 = safe_mwu(d2, d12)

    pval_df = pd.DataFrame([
        {"comparison": f"{label1} vs {label2}", "pvalue": p12},
        {"comparison": f"{label1} vs {label3}", "pvalue": p13},
        {"comparison": f"{label2} vs {label3}", "pvalue": p23},
    ])
    pval_df["fdr"] = benjamini_hochberg(pval_df["pvalue"].values)

    pval_csv = os.path.join(args.outdir, f"distance_pvalues_{args.metric}.csv")
    pval_df.to_csv(pval_csv, index=False)

    # -----------------------------------------------------
    # effect size
    # -----------------------------------------------------
    effect_rows = []
    comparisons = [
        (d1, d2, label1, label2, p12),
        (d1, d12, label1, label3, p13),
        (d2, d12, label2, label3, p23),
    ]
    for a, b, n1, n2, p in comparisons:
        eff = compute_effect_size(a, b)
        effect_rows.append({
            "comparison": f"{n1} vs {n2}",
            "mean_diff": eff["mean_diff"],
            "median_diff": eff["median_diff"],
            "cliffs_delta": eff["cliffs_delta"],
            "pvalue": p
        })

    effect_df = pd.DataFrame(effect_rows)
    effect_df["fdr"] = benjamini_hochberg(effect_df["pvalue"].values)

    effect_csv = os.path.join(args.outdir, f"distance_effect_size_{args.metric}.csv")
    effect_df.to_csv(effect_csv, index=False)

    pvals = {
        "1_vs_2": p12,
        "1_vs_3": p13,
        "2_vs_3": p23,
    }

    # -----------------------------------------------------
    # Bootstrap
    # -----------------------------------------------------
    boot_df = bootstrap_distance_summary(
        X1, X2,
        metric=args.metric,
        n_pairs=args.bootstrap_pairs,
        n_boot=args.n_boot,
        seed=2024
    )
    boot_csv = os.path.join(args.outdir, f"bootstrap_summary_{args.metric}.csv")
    boot_df.to_csv(boot_csv, index=False)

    # -----------------------------------------------------
    # Centroid distance
    # -----------------------------------------------------
    centroid_df = build_centroid_distance_table(X1, X2, args.metric, args.name1, args.name2)
    centroid_csv = os.path.join(args.outdir, f"centroid_distance_{args.metric}.csv")
    centroid_df.to_csv(centroid_csv, index=False)

    # -----------------------------------------------------
    # PDF plots
    # -----------------------------------------------------
    pdf_file = os.path.join(args.outdir, f"comprehensive_plots_{args.metric}.pdf")
    with PdfPages(pdf_file) as pdf:
        # Page 1: violin + box
        fig, ax = plt.subplots(figsize=(8.2, 6.0), constrained_layout=True)
        plot_violin_box(
            ax=ax,
            data=data,
            labels=labels,
            pvals=pvals,
            metric=args.metric,
            max_points=args.n_plot_points,
            title="Read-level distance comparison"
        )
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        # Page 2: CDF + Histogram
        fig, axes = plt.subplots(2, 1, figsize=(8.2, 10.0), constrained_layout=True)
        plot_cdf(
            ax=axes[0],
            data=data,
            labels=labels,
            metric=args.metric,
            title="Distance cumulative distribution"
        )
        plot_hist(
            ax=axes[1],
            data=data,
            labels=labels,
            metric=args.metric,
            title="Distance histogram"
        )
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        # Page 3: bootstrap stability
        fig, axes = plt.subplots(2, 1, figsize=(10.5, 10.0), constrained_layout=True)
        plot_bootstrap(
            ax=axes[0],
            boot_df=boot_df,
            value_col="mean",
            title="Bootstrap stability (mean)"
        )
        plot_bootstrap(
            ax=axes[1],
            boot_df=boot_df,
            value_col="median",
            title="Bootstrap stability (median)"
        )
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        # Page 4: PCA + centroid distance
        fig, axes = plt.subplots(2, 1, figsize=(10.0, 12.0), constrained_layout=True)
        plot_pca(
            ax=axes[0],
            X1=X1,
            X2=X2,
            name1=args.name1,
            name2=args.name2,
            max_points=args.pca_max_points
        )
        plot_centroid_distance(
            ax=axes[1],
            centroid_df=centroid_df,
            metric=args.metric,
            title="Read-to-centroid distance"
        )
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        # Page 5: effect size summary
        fig, ax = plt.subplots(figsize=(9.0, 6.5), constrained_layout=True)
        plot_effect_summary(ax=ax, effect_df=effect_df)
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

    # -----------------------------------------------------
    # txt summary
    # -----------------------------------------------------
    txt_file = os.path.join(args.outdir, f"summary_{args.metric}.txt")
    write_summary_txt(summary_df, pval_df, effect_df, centroid_df, txt_file)

    print("\n[DONE]")
    print("Saved files:")
    print(sample_csv)
    print(summary_csv)
    print(pval_csv)
    print(effect_csv)
    print(boot_csv)
    print(centroid_csv)
    print(pdf_file)
    print(txt_file)

    print("\nSummary:")
    print(summary_df.to_string(index=False))

    print("\nP-values:")
    print(pval_df.to_string(index=False))

    print("\nEffect sizes:")
    print(effect_df.to_string(index=False))


if __name__ == "__main__":
    main()