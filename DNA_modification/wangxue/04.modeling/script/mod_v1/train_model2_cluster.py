#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def normalize_rows_to_freq(counts):
    total = np.sum(counts)
    if total == 0:
        return np.zeros_like(counts, dtype=np.float64)
    return counts.astype(np.float64) / total


def compute_state_frequency(state_ids, n_states):
    counts = np.bincount(state_ids, minlength=n_states)
    freqs = normalize_rows_to_freq(counts)
    return counts, freqs


def save_dual_bar(freq_native, freq_canonical, out_path, title):
    x = np.arange(len(freq_native))
    width = 0.4

    plt.figure(figsize=(max(8, len(x) * 0.35), 4.5))
    plt.bar(x - width / 2, freq_native, width=width, label="native")
    plt.bar(x + width / 2, freq_canonical, width=width, label="canonical")
    plt.xlabel("state id")
    plt.ylabel("frequency")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_enrichment_bar(enrichment, out_path, title):
    x = np.arange(len(enrichment))
    plt.figure(figsize=(max(8, len(x) * 0.35), 4.5))
    plt.bar(x, enrichment)
    plt.xlabel("state id")
    plt.ylabel("log2(native_freq / canonical_freq)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_pca_domain_plot(native_pca, canonical_pca, out_path, title, max_points=50000, seed=42):
    rng = np.random.default_rng(seed)

    def subsample(arr, n):
        if len(arr) <= n:
            return arr
        idx = rng.choice(len(arr), size=n, replace=False)
        return arr[idx]

    n_show = max_points // 2
    native_show = subsample(native_pca, n_show)
    canonical_show = subsample(canonical_pca, n_show)

    plt.figure(figsize=(6, 5))
    plt.scatter(canonical_show[:, 0], canonical_show[:, 1], s=2, alpha=0.4, label="canonical")
    plt.scatter(native_show[:, 0], native_show[:, 1], s=2, alpha=0.4, label="native")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_pca_state_plot(native_pca, native_state_ids, out_path, title, max_points=50000, seed=42):
    rng = np.random.default_rng(seed)
    if len(native_pca) > max_points:
        idx = rng.choice(len(native_pca), size=max_points, replace=False)
        native_pca = native_pca[idx]
        native_state_ids = native_state_ids[idx]

    plt.figure(figsize=(6, 5))
    plt.scatter(native_pca[:, 0], native_pca[:, 1], c=native_state_ids, s=2, alpha=0.5)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


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
    parser = argparse.ArgumentParser(description="Model2 V1: residual state discovery by PCA + KMeans")
    parser.add_argument("--native_residual", required=True, help="Path to native_model1_residual.npy")
    parser.add_argument("--canonical_residual", required=True, help="Path to canonical_model1_residual.npy")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--n_states", type=int, default=16, help="Number of KMeans clusters/states")
    parser.add_argument("--pca_dim", type=int, default=16, help="PCA dimension before clustering")
    parser.add_argument("--max_train_native", type=int, default=200000, help="Max native chunks used to fit PCA/KMeans")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--top_n_waveforms", type=int, default=16, help="How many most frequent states to plot mean waveforms for")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    rng = np.random.default_rng(args.seed)

    # -----------------------------------------------------
    # Load residuals
    # -----------------------------------------------------
    native_residual = np.load(args.native_residual).astype(np.float32)      # (Nn, L)
    canonical_residual = np.load(args.canonical_residual).astype(np.float32) # (Nc, L)

    if native_residual.ndim != 2 or canonical_residual.ndim != 2:
        raise ValueError("Residual arrays must have shape (N, L)")
    if native_residual.shape[1] != canonical_residual.shape[1]:
        raise ValueError(
            f"Residual chunk length mismatch: native {native_residual.shape[1]} vs canonical {canonical_residual.shape[1]}"
        )

    chunk_len = native_residual.shape[1]

    # -----------------------------------------------------
    # Subsample native for fitting
    # -----------------------------------------------------
    if len(native_residual) > args.max_train_native:
        fit_idx = rng.choice(len(native_residual), size=args.max_train_native, replace=False)
        native_fit = native_residual[fit_idx]
    else:
        fit_idx = np.arange(len(native_residual))
        native_fit = native_residual

    # -----------------------------------------------------
    # PCA fit on native residual
    # -----------------------------------------------------
    pca = PCA(n_components=args.pca_dim, random_state=args.seed)
    native_fit_pca = pca.fit_transform(native_fit)

    # transform full native / canonical
    native_pca = pca.transform(native_residual)
    canonical_pca = pca.transform(canonical_residual)

    # -----------------------------------------------------
    # KMeans fit on native residual PCA
    # -----------------------------------------------------
    kmeans = KMeans(
        n_clusters=args.n_states,
        random_state=args.seed,
        n_init=10
    )
    kmeans.fit(native_fit_pca)
    joblib.dump(pca, os.path.join(args.out_dir, "model2_pca.joblib"))
    joblib.dump(kmeans, os.path.join(args.out_dir, "model2_kmeans.joblib"))
    native_state_ids = kmeans.predict(native_pca)
    canonical_state_ids = kmeans.predict(canonical_pca)

    # -----------------------------------------------------
    # Save raw outputs
    # -----------------------------------------------------
    np.save(os.path.join(args.out_dir, "native_model2_state_id.npy"), native_state_ids.astype(np.int32))
    np.save(os.path.join(args.out_dir, "canonical_model2_state_id.npy"), canonical_state_ids.astype(np.int32))
    np.save(os.path.join(args.out_dir, "native_model2_pca.npy"), native_pca.astype(np.float32))
    np.save(os.path.join(args.out_dir, "canonical_model2_pca.npy"), canonical_pca.astype(np.float32))
    np.save(os.path.join(args.out_dir, "model2_cluster_centers_pca.npy"), kmeans.cluster_centers_.astype(np.float32))

    # -----------------------------------------------------
    # Frequency / enrichment
    # -----------------------------------------------------
    native_counts, native_freq = compute_state_frequency(native_state_ids, args.n_states)
    canonical_counts, canonical_freq = compute_state_frequency(canonical_state_ids, args.n_states)

    eps = 1e-8
    enrichment = np.log2((native_freq + eps) / (canonical_freq + eps))

    np.save(os.path.join(args.out_dir, "native_model2_state_counts.npy"), native_counts.astype(np.int64))
    np.save(os.path.join(args.out_dir, "canonical_model2_state_counts.npy"), canonical_counts.astype(np.int64))
    np.save(os.path.join(args.out_dir, "native_model2_state_freq.npy"), native_freq.astype(np.float32))
    np.save(os.path.join(args.out_dir, "canonical_model2_state_freq.npy"), canonical_freq.astype(np.float32))
    np.save(os.path.join(args.out_dir, "model2_state_enrichment_log2.npy"), enrichment.astype(np.float32))

    # -----------------------------------------------------
    # Plots
    # -----------------------------------------------------
    save_dual_bar(
        native_freq,
        canonical_freq,
        out_path=os.path.join(args.out_dir, "state_frequency_native_vs_canonical.png"),
        title="State frequency: native vs canonical"
    )

    save_enrichment_bar(
        enrichment,
        out_path=os.path.join(args.out_dir, "state_enrichment_log2.png"),
        title="State enrichment log2(native/canonical)"
    )

    save_pca_domain_plot(
        native_pca[:, :2],
        canonical_pca[:, :2],
        out_path=os.path.join(args.out_dir, "residual_pca_domain_native_vs_canonical.png"),
        title="Residual PCA: native vs canonical",
        seed=args.seed
    )

    save_pca_state_plot(
        native_pca[:, :2],
        native_state_ids,
        out_path=os.path.join(args.out_dir, "native_residual_pca_by_state.png"),
        title="Native residual PCA colored by state",
        seed=args.seed
    )

    # -----------------------------------------------------
    # Waveform summaries
    # -----------------------------------------------------
    plot_state_mean_waveforms(
        residuals=native_residual,
        state_ids=native_state_ids,
        n_states=args.n_states,
        out_dir=os.path.join(args.out_dir, "native_state_mean_waveforms"),
        prefix="native",
        top_n=args.top_n_waveforms
    )

    plot_state_mean_waveforms(
        residuals=canonical_residual,
        state_ids=canonical_state_ids,
        n_states=args.n_states,
        out_dir=os.path.join(args.out_dir, "canonical_state_mean_waveforms"),
        prefix="canonical",
        top_n=args.top_n_waveforms
    )

    save_state_examples(
        residuals=native_residual,
        state_ids=native_state_ids,
        n_states=args.n_states,
        out_dir=os.path.join(args.out_dir, "native_state_examples"),
        prefix="native",
        seed=args.seed
    )

    save_state_examples(
        residuals=canonical_residual,
        state_ids=canonical_state_ids,
        n_states=args.n_states,
        out_dir=os.path.join(args.out_dir, "canonical_state_examples"),
        prefix="canonical",
        seed=args.seed
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------
    summary = {
        "native_residual": args.native_residual,
        "canonical_residual": args.canonical_residual,
        "chunk_len": int(chunk_len),
        "n_native_chunks": int(len(native_residual)),
        "n_canonical_chunks": int(len(canonical_residual)),
        "n_states": int(args.n_states),
        "pca_dim": int(args.pca_dim),
        "max_train_native": int(args.max_train_native),
        "seed": int(args.seed),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "native_state_counts": native_counts.tolist(),
        "canonical_state_counts": canonical_counts.tolist(),
        "native_state_freq": native_freq.tolist(),
        "canonical_state_freq": canonical_freq.tolist(),
        "state_enrichment_log2": enrichment.tolist(),
        "top_native_enriched_states": np.argsort(enrichment)[::-1].tolist(),
        "top_canonical_enriched_states": np.argsort(enrichment).tolist(),
    }
    save_json(summary, os.path.join(args.out_dir, "model2_summary.json"))

    print("[DONE] Model2 clustering outputs saved to:", args.out_dir)
    print("Key files:")
    print("  native_model2_state_id.npy")
    print("  canonical_model2_state_id.npy")
    print("  native_model2_pca.npy")
    print("  canonical_model2_pca.npy")
    print("  state_frequency_native_vs_canonical.png")
    print("  state_enrichment_log2.png")
    print("  residual_pca_domain_native_vs_canonical.png")
    print("  native_residual_pca_by_state.png")
    print("  native_state_mean_waveforms/")
    print("  canonical_state_mean_waveforms/")
    print("  native_state_examples/")
    print("  canonical_state_examples/")
    print("  model2_summary.json")
    print("  model2_pca.joblib")
    print("  model2_kmeans.joblib")

if __name__ == "__main__":
    main()
