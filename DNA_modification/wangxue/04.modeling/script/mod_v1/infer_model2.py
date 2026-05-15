#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import joblib


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_pca_state_plot(pca_arr, state_ids, out_path, title, max_points=50000, seed=42):
    rng = np.random.default_rng(seed)

    if len(pca_arr) > max_points:
        idx = rng.choice(len(pca_arr), size=max_points, replace=False)
        pca_arr = pca_arr[idx]
        state_ids = state_ids[idx]

    plt.figure(figsize=(6, 5))
    plt.scatter(pca_arr[:, 0], pca_arr[:, 1], c=state_ids, s=2, alpha=0.5)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def compute_state_frequency(state_ids, n_states):
    counts = np.bincount(state_ids, minlength=n_states)
    freqs = counts.astype(np.float64) / max(np.sum(counts), 1)
    return counts, freqs


def plot_state_mean_waveforms(residuals, state_ids, n_states, out_dir, prefix, top_n=None):
    ensure_dir(out_dir)

    counts = np.bincount(state_ids, minlength=n_states)
    order = np.argsort(counts)[::-1]
    if top_n is not None:
        order = order[:top_n]

    for s in order:
        cur = residuals[state_ids == s]
        if len(cur) == 0:
            continue

        mean_wave = cur.mean(axis=0)
        std_wave = cur.std(axis=0)
        x = np.arange(mean_wave.shape[0])

        plt.figure(figsize=(8, 4))
        plt.plot(x, mean_wave, label=f"state {s} mean")
        plt.fill_between(x, mean_wave - std_wave, mean_wave + std_wave, alpha=0.25, label="±1 std")
        plt.xlabel("position in chunk")
        plt.ylabel("residual")
        plt.title(f"{prefix} state {s} | n={len(cur)}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_state_{s}_mean_waveform.png"), dpi=200)
        plt.close()


def save_state_examples(residuals, state_ids, n_states, out_dir, prefix, n_examples=12, seed=42):
    ensure_dir(out_dir)
    rng = np.random.default_rng(seed)

    for s in range(n_states):
        cur_idx = np.where(state_ids == s)[0]
        if len(cur_idx) == 0:
            continue

        chosen = cur_idx if len(cur_idx) <= n_examples else rng.choice(cur_idx, size=n_examples, replace=False)

        plt.figure(figsize=(8, 4))
        for idx in chosen:
            plt.plot(residuals[idx], alpha=0.7, linewidth=1)
        plt.xlabel("position in chunk")
        plt.ylabel("residual")
        plt.title(f"{prefix} state {s} examples | n_show={len(chosen)}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_state_{s}_examples.png"), dpi=200)
        plt.close()



def main():
    parser = argparse.ArgumentParser(description="Infer model2 states for new residual data")
    parser.add_argument("--residual", required=True, help="Path to new model1 residual, e.g. synthetic_model1_residual.npy")
    parser.add_argument("--model2_pca", required=True, help="Path to model2_pca.joblib")
    parser.add_argument("--model2_kmeans", required=True, help="Path to model2_kmeans.joblib")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--prefix", default="synthetic_model2", help="Prefix for output files")
    parser.add_argument("--top_n_waveforms", type=int, default=16, help="How many most frequent states to plot")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    # -----------------------------------------------------
    # Load residual and models
    # -----------------------------------------------------
    residual = np.load(args.residual).astype(np.float32)
    if residual.ndim != 2:
        raise ValueError(f"Residual must have shape (N, L), got {residual.shape}")

    pca = joblib.load(args.model2_pca)
    kmeans = joblib.load(args.model2_kmeans)

    # -----------------------------------------------------
    # Transform and predict
    # -----------------------------------------------------
    residual_pca = pca.transform(residual)
    state_ids = kmeans.predict(residual_pca)

    n_states = int(kmeans.n_clusters)

    # -----------------------------------------------------
    # Save arrays
    # -----------------------------------------------------
    np.save(os.path.join(args.out_dir, f"{args.prefix}_pca.npy"), residual_pca.astype(np.float32))
    np.save(os.path.join(args.out_dir, f"{args.prefix}_state_id.npy"), state_ids.astype(np.int32))

    counts, freqs = compute_state_frequency(state_ids, n_states)
    np.save(os.path.join(args.out_dir, f"{args.prefix}_state_counts.npy"), counts.astype(np.int64))
    np.save(os.path.join(args.out_dir, f"{args.prefix}_state_freq.npy"), freqs.astype(np.float32))

    # -----------------------------------------------------
    # Plots
    # -----------------------------------------------------
    save_pca_state_plot(
        residual_pca[:, :2],
        state_ids,
        out_path=os.path.join(args.out_dir, f"{args.prefix}_pca_by_state.png"),
        title=f"{args.prefix} PCA colored by state",
        seed=args.seed
    )

    plot_state_mean_waveforms(
        residuals=residual,
        state_ids=state_ids,
        n_states=n_states,
        out_dir=os.path.join(args.out_dir, f"{args.prefix}_state_mean_waveforms"),
        prefix=args.prefix,
        top_n=args.top_n_waveforms
    )

    save_state_examples(
        residuals=residual,
        state_ids=state_ids,
        n_states=n_states,
        out_dir=os.path.join(args.out_dir, f"{args.prefix}_state_examples"),
        prefix=args.prefix,
        seed=args.seed
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------
    summary = {
        "residual": args.residual,
        "model2_pca": args.model2_pca,
        "model2_kmeans": args.model2_kmeans,
        "n_chunks": int(len(residual)),
        "chunk_len": int(residual.shape[1]),
        "n_states": n_states,
        "state_counts": counts.tolist(),
        "state_freq": freqs.tolist(),
        "prefix": args.prefix,
    }

    with open(os.path.join(args.out_dir, f"{args.prefix}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("[DONE] Saved files:")
    print(f"  {args.prefix}_pca.npy")
    print(f"  {args.prefix}_state_id.npy")
    print(f"  {args.prefix}_state_counts.npy")
    print(f"  {args.prefix}_state_freq.npy")
    print(f"  {args.prefix}_pca_by_state.png")
    print(f"  {args.prefix}_state_mean_waveforms/")
    print(f"  {args.prefix}_state_examples/")
    print(f"  {args.prefix}_summary.json")


if __name__ == "__main__":
    main()
