#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_source_map(path):
    if path is None or (not os.path.exists(path)):
        return None
    with open(path, "r") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def summarize_array(name, arr):
    return {
        "name": name,
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def save_dual_hist(canonical, native, out_path, xlabel, title, bins=100, density=False):
    plt.figure(figsize=(6, 4))
    plt.hist(canonical, bins=bins, alpha=0.6, label="canonical", density=density)
    plt.hist(native, bins=bins, alpha=0.6, label="native", density=density)
    plt.xlabel(xlabel)
    plt.ylabel("density" if density else "count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_dual_boxplot(canonical, native, out_path, ylabel, title, showfliers=False):
    plt.figure(figsize=(5, 4))
    plt.boxplot([canonical, native], tick_labels=["canonical", "native"], showfliers=showfliers)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_domain_pca(canonical_arr, native_arr, out_path, title):
    x = np.concatenate([canonical_arr, native_arr], axis=0)
    domain = np.array([0] * len(canonical_arr) + [1] * len(native_arr))

    pca = PCA(n_components=2)
    x2 = pca.fit_transform(x)

    plt.figure(figsize=(6, 5))
    plt.scatter(x2[domain == 0, 0], x2[domain == 0, 1], s=2, alpha=0.5, label="canonical")
    plt.scatter(x2[domain == 1, 0], x2[domain == 1, 1], s=2, alpha=0.5, label="native")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return pca.explained_variance_ratio_.tolist(), x2, domain


def save_source_pca(arr, source_idx, out_path, title):
    pca = PCA(n_components=2)
    arr2 = pca.fit_transform(arr)

    plt.figure(figsize=(6, 5))
    plt.scatter(arr2[:, 0], arr2[:, 1], c=source_idx, s=2, alpha=0.5)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return pca.explained_variance_ratio_.tolist()


def compute_source_stats(values, source_idx, source_map=None):
    results = []
    uniq = sorted(np.unique(source_idx).tolist())
    for k in uniq:
        cur = values[source_idx == k]
        item = {
            "source_index": int(k),
            "source_name": source_map.get(int(k), f"source_{k}") if source_map else f"source_{k}",
            "n": int(len(cur)),
            "mean": float(np.mean(cur)),
            "median": float(np.median(cur)),
            "p95": float(np.percentile(cur, 95)),
            "p99": float(np.percentile(cur, 99)),
            "max": float(np.max(cur)),
        }
        results.append(item)
    return results


def residual_energy(residual):
    return np.mean(residual ** 2, axis=1)


def residual_abs_mean(residual):
    return np.mean(np.abs(residual), axis=1)


def write_summary(
    out_path,
    canonical_mse_stats,
    native_mse_stats,
    canonical_res_energy_stats,
    native_res_energy_stats,
    canonical_res_abs_stats,
    native_res_abs_stats,
    latent_domain_pca_evr,
    residual_domain_pca_evr,
    canonical_latent_source_stats,
    native_latent_source_stats,
    canonical_mse_by_source,
    native_mse_by_source,
):
    with open(out_path, "w") as f:
        f.write("[Recon MSE summary]\n")
        for item in [canonical_mse_stats, native_mse_stats]:
            f.write(
                f"{item['name']}\t"
                f"n={item['n']}\tmean={item['mean']:.8f}\tmedian={item['median']:.8f}\t"
                f"p95={item['p95']:.8f}\tp99={item['p99']:.8f}\tmax={item['max']:.8f}\n"
            )
        f.write("\n")

        f.write("[Residual energy summary]\n")
        for item in [canonical_res_energy_stats, native_res_energy_stats]:
            f.write(
                f"{item['name']}\t"
                f"n={item['n']}\tmean={item['mean']:.8f}\tmedian={item['median']:.8f}\t"
                f"p95={item['p95']:.8f}\tp99={item['p99']:.8f}\tmax={item['max']:.8f}\n"
            )
        f.write("\n")

        f.write("[Residual abs-mean summary]\n")
        for item in [canonical_res_abs_stats, native_res_abs_stats]:
            f.write(
                f"{item['name']}\t"
                f"n={item['n']}\tmean={item['mean']:.8f}\tmedian={item['median']:.8f}\t"
                f"p95={item['p95']:.8f}\tp99={item['p99']:.8f}\tmax={item['max']:.8f}\n"
            )
        f.write("\n")

        f.write("[Latent PCA explained variance ratio: canonical+native]\n")
        f.write(",".join([f"{x:.8f}" for x in latent_domain_pca_evr]) + "\n\n")

        f.write("[Residual PCA explained variance ratio: canonical+native]\n")
        f.write(",".join([f"{x:.8f}" for x in residual_domain_pca_evr]) + "\n\n")

        f.write("[Canonical latent source sizes]\n")
        for item in canonical_latent_source_stats:
            f.write(f"{item['source_index']}\t{item['source_name']}\tn={item['n']}\n")
        f.write("\n")

        f.write("[Native latent source sizes]\n")
        for item in native_latent_source_stats:
            f.write(f"{item['source_index']}\t{item['source_name']}\tn={item['n']}\n")
        f.write("\n")

        f.write("[Canonical recon MSE by source]\n")
        for item in canonical_mse_by_source:
            f.write(
                f"{item['source_index']}\t{item['source_name']}\t"
                f"n={item['n']}\tmean={item['mean']:.8f}\tmedian={item['median']:.8f}\t"
                f"p95={item['p95']:.8f}\tp99={item['p99']:.8f}\tmax={item['max']:.8f}\n"
            )
        f.write("\n")

        f.write("[Native recon MSE by source]\n")
        for item in native_mse_by_source:
            f.write(
                f"{item['source_index']}\t{item['source_name']}\t"
                f"n={item['n']}\tmean={item['mean']:.8f}\tmedian={item['median']:.8f}\t"
                f"p95={item['p95']:.8f}\tp99={item['p99']:.8f}\tmax={item['max']:.8f}\n"
            )


def main():
    parser = argparse.ArgumentParser(description="Compare canonical vs native model1 outputs")

    parser.add_argument("--canonical_latent", required=True)
    parser.add_argument("--canonical_residual", required=True)
    parser.add_argument("--canonical_mse", required=True)
    parser.add_argument("--canonical_source_index", required=True)
    parser.add_argument("--canonical_source_map", default=None)

    parser.add_argument("--native_latent", required=True)
    parser.add_argument("--native_residual", required=True)
    parser.add_argument("--native_mse", required=True)
    parser.add_argument("--native_source_index", required=True)
    parser.add_argument("--native_source_map", default=None)

    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    # load
    c_latent = np.load(args.canonical_latent)
    c_residual = np.load(args.canonical_residual)
    c_mse = np.load(args.canonical_mse)
    c_source_idx = np.load(args.canonical_source_index)
    c_source_map = load_source_map(args.canonical_source_map)

    n_latent = np.load(args.native_latent)
    n_residual = np.load(args.native_residual)
    n_mse = np.load(args.native_mse)
    n_source_idx = np.load(args.native_source_index)
    n_source_map = load_source_map(args.native_source_map)

    # basic stats
    c_mse_stats = summarize_array("canonical_mse", c_mse)
    n_mse_stats = summarize_array("native_mse", n_mse)

    c_res_energy = residual_energy(c_residual)
    n_res_energy = residual_energy(n_residual)
    c_res_energy_stats = summarize_array("canonical_residual_energy", c_res_energy)
    n_res_energy_stats = summarize_array("native_residual_energy", n_res_energy)

    c_res_abs = residual_abs_mean(c_residual)
    n_res_abs = residual_abs_mean(n_residual)
    c_res_abs_stats = summarize_array("canonical_residual_abs_mean", c_res_abs)
    n_res_abs_stats = summarize_array("native_residual_abs_mean", n_res_abs)

    # plots: mse
    save_dual_hist(
        c_mse, n_mse,
        out_path=os.path.join(args.out_dir, "compare_recon_mse_hist.png"),
        xlabel="reconstruction MSE",
        title="Canonical vs Native recon MSE",
        bins=100,
        density=False
    )

    save_dual_boxplot(
        c_mse, n_mse,
        out_path=os.path.join(args.out_dir, "compare_recon_mse_boxplot.png"),
        ylabel="reconstruction MSE",
        title="Canonical vs Native recon MSE"
    )

    # plots: residual energy
    save_dual_hist(
        c_res_energy, n_res_energy,
        out_path=os.path.join(args.out_dir, "compare_residual_energy_hist.png"),
        xlabel="residual energy",
        title="Canonical vs Native residual energy",
        bins=100,
        density=False
    )

    save_dual_boxplot(
        c_res_energy, n_res_energy,
        out_path=os.path.join(args.out_dir, "compare_residual_energy_boxplot.png"),
        ylabel="residual energy",
        title="Canonical vs Native residual energy"
    )

    # plots: residual abs mean
    save_dual_hist(
        c_res_abs, n_res_abs,
        out_path=os.path.join(args.out_dir, "compare_residual_abs_mean_hist.png"),
        xlabel="mean(abs(residual))",
        title="Canonical vs Native residual abs mean",
        bins=100,
        density=False
    )

    save_dual_boxplot(
        c_res_abs, n_res_abs,
        out_path=os.path.join(args.out_dir, "compare_residual_abs_mean_boxplot.png"),
        ylabel="mean(abs(residual))",
        title="Canonical vs Native residual abs mean"
    )

    # PCA: canonical + native
    latent_domain_pca_evr, _, _ = save_domain_pca(
        c_latent, n_latent,
        out_path=os.path.join(args.out_dir, "compare_latent_pca_domain.png"),
        title="PCA of latent: canonical vs native"
    )

    residual_domain_pca_evr, _, _ = save_domain_pca(
        c_residual, n_residual,
        out_path=os.path.join(args.out_dir, "compare_residual_pca_domain.png"),
        title="PCA of residual: canonical vs native"
    )

    # source PCA inside each domain
    save_source_pca(
        c_latent, c_source_idx,
        out_path=os.path.join(args.out_dir, "canonical_latent_pca_by_source.png"),
        title="Canonical latent PCA by source"
    )

    save_source_pca(
        n_latent, n_source_idx,
        out_path=os.path.join(args.out_dir, "native_latent_pca_by_source.png"),
        title="Native latent PCA by source"
    )

    save_source_pca(
        c_residual, c_source_idx,
        out_path=os.path.join(args.out_dir, "canonical_residual_pca_by_source.png"),
        title="Canonical residual PCA by source"
    )

    save_source_pca(
        n_residual, n_source_idx,
        out_path=os.path.join(args.out_dir, "native_residual_pca_by_source.png"),
        title="Native residual PCA by source"
    )

    # source stats
    canonical_latent_source_stats = compute_source_stats(
        np.zeros(len(c_source_idx)), c_source_idx, c_source_map
    )
    native_latent_source_stats = compute_source_stats(
        np.zeros(len(n_source_idx)), n_source_idx, n_source_map
    )
    canonical_mse_by_source = compute_source_stats(c_mse, c_source_idx, c_source_map)
    native_mse_by_source = compute_source_stats(n_mse, n_source_idx, n_source_map)

    # summary
    write_summary(
        out_path=os.path.join(args.out_dir, "compare_summary.txt"),
        canonical_mse_stats=c_mse_stats,
        native_mse_stats=n_mse_stats,
        canonical_res_energy_stats=c_res_energy_stats,
        native_res_energy_stats=n_res_energy_stats,
        canonical_res_abs_stats=c_res_abs_stats,
        native_res_abs_stats=n_res_abs_stats,
        latent_domain_pca_evr=latent_domain_pca_evr,
        residual_domain_pca_evr=residual_domain_pca_evr,
        canonical_latent_source_stats=canonical_latent_source_stats,
        native_latent_source_stats=native_latent_source_stats,
        canonical_mse_by_source=canonical_mse_by_source,
        native_mse_by_source=native_mse_by_source,
    )

    print("[DONE] Saved comparison outputs to:", args.out_dir)
    print("Key files:")
    print("  compare_recon_mse_hist.png")
    print("  compare_recon_mse_boxplot.png")
    print("  compare_residual_energy_hist.png")
    print("  compare_residual_energy_boxplot.png")
    print("  compare_residual_abs_mean_hist.png")
    print("  compare_residual_abs_mean_boxplot.png")
    print("  compare_latent_pca_domain.png")
    print("  compare_residual_pca_domain.png")
    print("  canonical_latent_pca_by_source.png")
    print("  native_latent_pca_by_source.png")
    print("  canonical_residual_pca_by_source.png")
    print("  native_residual_pca_by_source.png")
    print("  compare_summary.txt")


if __name__ == "__main__":
    main()