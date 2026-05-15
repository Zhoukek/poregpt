#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load(path, max_n=0):
    arr = np.load(path, mmap_mode="r")
    if max_n > 0 and len(arr) > max_n:
        arr = arr[:max_n]
    return np.asarray(arr)


def auc_simple(c, n):
    scores = np.concatenate([c, n])
    labels = np.concatenate([
        np.zeros(len(c), dtype=np.int8),
        np.ones(len(n), dtype=np.int8),
    ])

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)

    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)
    rank_sum_pos = np.sum(ranks[labels == 1])

    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def cohens_d(c, n):
    vc = np.var(c, ddof=1)
    vn = np.var(n, ddof=1)
    nc = len(c)
    nn = len(n)
    pooled = np.sqrt(((nc - 1) * vc + (nn - 1) * vn) / max(nc + nn - 2, 1))
    return float((np.mean(n) - np.mean(c)) / (pooled + 1e-12))


def cliffs_delta_from_auc(auc):
    return 2 * auc - 1


def print_and_save_summary(name, c, n, rows):
    auc = auc_simple(c, n)
    d = cohens_d(c, n)
    delta = cliffs_delta_from_auc(auc)

    c_mean = np.mean(c)
    n_mean = np.mean(n)

    c_median = np.median(c)
    n_median = np.median(n)

    c_p75 = np.percentile(c, 75)
    n_p75 = np.percentile(n, 75)

    c_p95 = np.percentile(c, 95)
    n_p95 = np.percentile(n, 95)

    fold = n_mean / (c_mean + 1e-12)

    print("=" * 70)
    print(name)
    print(f"canonical mean   = {c_mean:.6g}")
    print(f"native mean      = {n_mean:.6g}")
    print(f"fold mean        = {fold:.4f}")
    print(f"canonical median = {c_median:.6g}")
    print(f"native median    = {n_median:.6g}")
    print(f"canonical p75    = {c_p75:.6g}")
    print(f"native p75       = {n_p75:.6g}")
    print(f"canonical p95    = {c_p95:.6g}")
    print(f"native p95       = {n_p95:.6g}")
    print(f"AUC              = {auc:.4f}")
    print(f"Cohen's d        = {d:.4f}")
    print(f"Cliff's delta    = {delta:.4f}")

    rows.append([
        name,
        len(c),
        len(n),
        c_mean,
        n_mean,
        fold,
        c_median,
        n_median,
        c_p75,
        n_p75,
        c_p95,
        n_p95,
        auc,
        d,
        delta,
    ])


def plot_hist(c, n, name, out_dir):
    plt.figure(figsize=(6, 4))
    plt.hist(c, bins=100, density=True, alpha=0.5, label="canonical")
    plt.hist(n, bins=100, density=True, alpha=0.5, label="native")
    plt.xlabel(name)
    plt.ylabel("density")
    plt.title(f"{name} histogram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_hist.png"), dpi=300)
    plt.close()


def plot_boxplot(c, n, name, out_dir):
    plt.figure(figsize=(5, 5))
    plt.boxplot(
        [c, n],
        labels=["canonical", "native"],
        showfliers=False,
        widths=0.6,
    )
    plt.ylabel(name)
    plt.title(f"{name} boxplot")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_boxplot.png"), dpi=300)
    plt.close()


def plot_log_boxplot(c, n, name, out_dir):
    c_log = np.log10(c + 1e-12)
    n_log = np.log10(n + 1e-12)

    plt.figure(figsize=(5, 5))
    plt.boxplot(
        [c_log, n_log],
        labels=["canonical", "native"],
        showfliers=False,
        widths=0.6,
    )
    plt.ylabel(f"log10({name})")
    plt.title(f"log10 {name} boxplot")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_log_boxplot.png"), dpi=300)
    plt.close()


def plot_violin(c, n, name, out_dir):
    plt.figure(figsize=(5, 5))
    plt.violinplot([c, n], showmeans=True, showmedians=True)
    plt.xticks([1, 2], ["canonical", "native"])
    plt.ylabel(name)
    plt.title(f"{name} violin")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_violin.png"), dpi=300)
    plt.close()


def plot_profile(c_res, n_res, out_dir):
    c_profile = np.mean(np.abs(c_res), axis=0)
    n_profile = np.mean(np.abs(n_res), axis=0)
    diff = n_profile - c_profile

    peak_pos = int(np.argmax(diff))

    plt.figure(figsize=(8, 4))
    plt.plot(c_profile, label="canonical")
    plt.plot(n_profile, label="native")
    plt.xlabel("position")
    plt.ylabel("mean abs residual")
    plt.title("residual position profile")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "residual_profile.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(diff, label="native - canonical")
    plt.axhline(0, linestyle="--")
    plt.xlabel("position")
    plt.ylabel("delta mean abs residual")
    plt.title("residual enrichment profile")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "residual_profile_delta.png"), dpi=300)
    plt.close()

    print("=" * 70)
    print("Residual profile")
    print(f"max native-canonical peak position = {peak_pos}")
    print(f"peak delta = {diff[peak_pos]:.6g}")
    print(f"mean delta = {np.mean(diff):.6g}")
    print(f"native mean profile = {np.mean(n_profile):.6g}")
    print(f"canonical mean profile = {np.mean(c_profile):.6g}")


def save_summary_csv(rows, out_path):
    header = [
        "metric",
        "canonical_n",
        "native_n",
        "canonical_mean",
        "native_mean",
        "fold_native_over_canonical",
        "canonical_median",
        "native_median",
        "canonical_p75",
        "native_p75",
        "canonical_p95",
        "native_p95",
        "auc",
        "cohens_d",
        "cliffs_delta",
    ]

    with open(out_path, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(map(str, row)) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--canonical_prefix", required=True)
    parser.add_argument("--native_prefix", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_n", type=int, default=1000000)
    parser.add_argument("--max_residual_n", type=int, default=100000)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    metrics = [
        "recon_l1",
        "recon_mse",
        "residual_std",
        "residual_max_abs",
        "latent_norm",
    ]

    rows = []

    for m in metrics:
        c_path = os.path.join(args.dir, f"{args.canonical_prefix}_{m}.npy")
        n_path = os.path.join(args.dir, f"{args.native_prefix}_{m}.npy")

        if not os.path.exists(c_path) or not os.path.exists(n_path):
            print(f"[WARN] skip {m}, missing file")
            continue

        c = load(c_path, args.max_n)
        n = load(n_path, args.max_n)

        print_and_save_summary(m, c, n, rows)

        plot_hist(c, n, m, args.out_dir)
        plot_boxplot(c, n, m, args.out_dir)
        plot_log_boxplot(c, n, m, args.out_dir)
        plot_violin(c, n, m, args.out_dir)

    c_res_path = os.path.join(args.dir, f"{args.canonical_prefix}_residual.npy")
    n_res_path = os.path.join(args.dir, f"{args.native_prefix}_residual.npy")

    if os.path.exists(c_res_path) and os.path.exists(n_res_path):
        c_res = load(c_res_path, args.max_residual_n)
        n_res = load(n_res_path, args.max_residual_n)
        plot_profile(c_res, n_res, args.out_dir)
    else:
        print("[WARN] residual.npy not found, skip residual profile")

    summary_path = os.path.join(args.out_dir, "quick_eval_summary.csv")
    save_summary_csv(rows, summary_path)

    print("=" * 70)
    print("[DONE]")
    print(f"summary csv: {summary_path}")
    print(f"figures saved to: {args.out_dir}")


if __name__ == "__main__":
    main()