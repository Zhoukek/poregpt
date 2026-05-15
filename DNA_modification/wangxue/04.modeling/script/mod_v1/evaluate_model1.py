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


def load_train_history(path):
    if path is None or (not os.path.exists(path)):
        return None
    with open(path, "r") as f:
        hist = json.load(f)
    return hist


def save_loss_curve(train_history, out_dir):
    if train_history is None:
        return None

    epochs = [x["epoch"] for x in train_history]
    train_loss = [x["train_loss"] for x in train_history]
    val_loss = [x["val_loss"] for x in train_history]

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_loss, label="train_loss", linewidth=2)
    plt.plot(epochs, val_loss, label="val_loss", linewidth=2)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Model1 training curve")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(out_dir, "loss_curve.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def save_mse_histogram(mse, out_dir, bins=100):
    plt.figure(figsize=(6, 4))
    plt.hist(mse, bins=bins)
    plt.xlim(0, 0.5)
    plt.xlabel("reconstruction MSE")
    plt.ylabel("count")
    plt.title("Model1 reconstruction MSE")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "recon_mse_hist.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path



def save_triplet_plot(x, xhat, r, mse, idx, out_path, title_prefix="sample"):
    fig, axes = plt.subplots(3, 1, figsize=(8, 5), sharex=True)

    axes[0].plot(x[idx], linewidth=1)
    axes[0].set_title(f"{title_prefix} input | idx={idx} | mse={mse[idx]:.6f}")

    axes[1].plot(xhat[idx], linewidth=1)
    axes[1].set_title("reconstructed")

    axes[2].plot(r[idx], linewidth=1)
    axes[2].set_title("residual = input - recon")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_random_triplets(x, xhat, r, mse, out_dir, n_random=12, seed=42):
    rng = np.random.default_rng(seed)
    n = len(x)
    n_random = min(n_random, n)
    idxs = rng.choice(n, size=n_random, replace=False)

    triplet_dir = os.path.join(out_dir, "random_triplets")
    ensure_dir(triplet_dir)

    for idx in idxs:
        out_path = os.path.join(triplet_dir, f"random_idx_{idx}.png")
        save_triplet_plot(x, xhat, r, mse, idx, out_path, title_prefix="random")

    return triplet_dir, idxs.tolist()


def save_worst_triplets(x, xhat, r, mse, out_dir, n_worst=12):
    n = len(x)
    n_worst = min(n_worst, n)
    idxs = np.argsort(mse)[-n_worst:]

    triplet_dir = os.path.join(out_dir, "worst_triplets")
    ensure_dir(triplet_dir)

    for idx in idxs:
        out_path = os.path.join(triplet_dir, f"worst_idx_{idx}.png")
        save_triplet_plot(x, xhat, r, mse, idx, out_path, title_prefix="worst")

    return triplet_dir, idxs.tolist()


def compute_source_mse_stats(mse, source_idx, source_map=None):
    results = []
    uniq = sorted(np.unique(source_idx).tolist())
    for k in uniq:
        cur = mse[source_idx == k]
        item = {
            "source_index": int(k),
            "source_name": source_map.get(int(k), f"source_{k}") if source_map else f"source_{k}",
            "n": int(len(cur)),
            "mean": float(cur.mean()),
            "median": float(np.median(cur)),
            "p90": float(np.percentile(cur, 90)),
            "p95": float(np.percentile(cur, 95)),
            "p99": float(np.percentile(cur, 99)),
            "max": float(cur.max()),
        }
        results.append(item)
    return results


def save_source_mse_boxplot(mse, source_idx, out_dir, source_map=None):
    uniq = sorted(np.unique(source_idx).tolist())
    data = [mse[source_idx == k] for k in uniq]
    labels = [os.path.basename(source_map[k]) if (source_map and k in source_map) else str(k) for k in uniq]

    plt.figure(figsize=(max(6, len(uniq) * 1.2), 4.5))
    plt.boxplot(data, tick_labels=labels, showfliers=False)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("reconstruction MSE")
    plt.title("Reconstruction MSE by source")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "recon_mse_by_source_boxplot.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path





def save_pca_scatter(arr, source_idx, out_dir, title, out_name):
    pca = PCA(n_components=2)
    arr2 = pca.fit_transform(arr)

    plt.figure(figsize=(6, 5))
    plt.scatter(arr2[:, 0], arr2[:, 1], c=source_idx, s=2, alpha=0.5)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.tight_layout()
    out_path = os.path.join(out_dir, out_name)
    plt.savefig(out_path, dpi=200)
    plt.close()

    return out_path, pca.explained_variance_ratio_.tolist(), arr2


def compute_source_centers(z, source_idx, source_map=None):
    centers = {}
    uniq = sorted(np.unique(source_idx).tolist())
    for k in uniq:
        cur = z[source_idx == k]
        centers[int(k)] = cur.mean(axis=0)

    pairwise = []
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            a = uniq[i]
            b = uniq[j]
            d = float(np.linalg.norm(centers[a] - centers[b]))
            pairwise.append({
                "source_a": int(a),
                "source_a_name": source_map.get(a, f"source_{a}") if source_map else f"source_{a}",
                "source_b": int(b),
                "source_b_name": source_map.get(b, f"source_{b}") if source_map else f"source_{b}",
                "distance": d,
            })
    return pairwise


def write_summary(
    out_dir,
    mse,
    random_idxs,
    worst_idxs,
    source_mse_stats,
    latent_pca_evr,
    residual_pca_evr,
    center_distances,
    train_history=None,
):
    summary_path = os.path.join(out_dir, "evaluation_summary.txt")
    with open(summary_path, "w") as f:
        if train_history is not None and len(train_history) > 0:
            last = train_history[-1]
            best_val = min([x["val_loss"] for x in train_history])
            f.write("[Training curve]\n")
            f.write(f"epochs\t{len(train_history)}\n")
            f.write(f"last_train_loss\t{last['train_loss']:.8f}\n")
            f.write(f"last_val_loss\t{last['val_loss']:.8f}\n")
            f.write(f"best_val_loss\t{best_val:.8f}\n\n")

        f.write("[Overall reconstruction MSE]\n")
        f.write(f"num_chunks\t{len(mse)}\n")
        f.write(f"mean\t{mse.mean():.8f}\n")
        f.write(f"median\t{np.median(mse):.8f}\n")
        f.write(f"p90\t{np.percentile(mse, 90):.8f}\n")
        f.write(f"p95\t{np.percentile(mse, 95):.8f}\n")
        f.write(f"p99\t{np.percentile(mse, 99):.8f}\n")
        f.write(f"max\t{mse.max():.8f}\n\n")

        f.write("[Random chunk indices]\n")
        f.write(",".join(map(str, random_idxs)) + "\n\n")

        f.write("[Worst chunk indices]\n")
        f.write(",".join(map(str, worst_idxs)) + "\n\n")

        f.write("[Reconstruction MSE by source]\n")
        for item in source_mse_stats:
            f.write(
                f"{item['source_index']}\t{item['source_name']}\t"
                f"n={item['n']}\tmean={item['mean']:.8f}\tmedian={item['median']:.8f}\t"
                f"p95={item['p95']:.8f}\tp99={item['p99']:.8f}\tmax={item['max']:.8f}\n"
            )
        f.write("\n")

        f.write("[Latent PCA explained variance ratio]\n")
        f.write(",".join([f"{x:.8f}" for x in latent_pca_evr]) + "\n\n")

        f.write("[Residual PCA explained variance ratio]\n")
        f.write(",".join([f"{x:.8f}" for x in residual_pca_evr]) + "\n\n")

        f.write("[Pairwise latent center distances]\n")
        for item in center_distances:
            f.write(
                f"{item['source_a']}:{item['source_a_name']}\t"
                f"{item['source_b']}:{item['source_b_name']}\t"
                f"{item['distance']:.8f}\n"
            )

    return summary_path



def main():
    parser = argparse.ArgumentParser(description="Evaluate model1 outputs on canonical data")
    parser.add_argument("--input", required=True, help="Path to canonical_model1_input.npy")
    parser.add_argument("--recon", required=True, help="Path to canonical_model1_recon.npy")
    parser.add_argument("--residual", required=True, help="Path to canonical_model1_residual.npy")
    parser.add_argument("--latent", required=True, help="Path to canonical_model1_latent.npy")
    parser.add_argument("--mse", required=True, help="Path to canonical_model1_recon_mse.npy")
    parser.add_argument("--source_index", required=True, help="Path to canonical_model1_source_index.npy")
    parser.add_argument("--source_map", default=None, help="Optional path to source_map.json")
    parser.add_argument("--train_history", default=None, help="Optional path to train_history.json")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--n_random", type=int, default=12, help="Number of random triplets")
    parser.add_argument("--n_worst", type=int, default=12, help="Number of worst triplets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    x = np.load(args.input)
    xhat = np.load(args.recon)
    r = np.load(args.residual)
    z = np.load(args.latent)
    mse = np.load(args.mse)
    source_idx = np.load(args.source_index)

    source_map = load_source_map(args.source_map)
    train_history = load_train_history(args.train_history)

    n = len(x)
    if not (len(xhat) == len(r) == len(z) == len(mse) == len(source_idx) == n):
        raise ValueError("Input arrays have inconsistent first dimension.")

    # 0) loss curve
    loss_curve_path = save_loss_curve(train_history, args.out_dir)

    # 1) reconstruction quality
    save_mse_histogram(mse, args.out_dir)
    _, random_idxs = save_random_triplets(
        x, xhat, r, mse, args.out_dir, n_random=args.n_random, seed=args.seed
    )
    _, worst_idxs = save_worst_triplets(
        x, xhat, r, mse, args.out_dir, n_worst=args.n_worst
    )

    # 2) batch effect
    source_mse_stats = compute_source_mse_stats(mse, source_idx, source_map=source_map)
    save_source_mse_boxplot(mse, source_idx, args.out_dir, source_map=source_map)

    _, latent_pca_evr, _ = save_pca_scatter(
        z, source_idx, args.out_dir,
        title="PCA of canonical latent",
        out_name="latent_pca_by_source.png"
    )

    _, residual_pca_evr, _ = save_pca_scatter(
        r, source_idx, args.out_dir,
        title="PCA of canonical residual",
        out_name="residual_pca_by_source.png"
    )

    center_distances = compute_source_centers(z, source_idx, source_map=source_map)

    summary_path = write_summary(
        out_dir=args.out_dir,
        mse=mse,
        random_idxs=random_idxs,
        worst_idxs=worst_idxs,
        source_mse_stats=source_mse_stats,
        latent_pca_evr=latent_pca_evr,
        residual_pca_evr=residual_pca_evr,
        center_distances=center_distances,
        train_history=train_history,
    )

    print("[DONE] Evaluation outputs saved to:", args.out_dir)
    print("Key files:")
    if loss_curve_path is not None:
        print("  loss_curve.png")
    print("  recon_mse_hist.png")
    print("  random_triplets/")
    print("  worst_triplets/")
    print("  recon_mse_by_source_boxplot.png")
    print("  latent_pca_by_source.png")
    print("  residual_pca_by_source.png")
    print("  evaluation_summary.txt")
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
