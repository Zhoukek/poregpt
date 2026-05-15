#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
import umap


def parse_args():
    ap = argparse.ArgumentParser(
        description="Advanced comparison toolkit for two groups of motif/base embeddings."
    )
    ap.add_argument("--emb_a", required=True, help="Group A embedding .npy, shape [N,L,D]")
    ap.add_argument("--emb_b", required=True, help="Group B embedding .npy, shape [M,L,D]")
    ap.add_argument("--label_a", default="A")
    ap.add_argument("--label_b", default="B")
    ap.add_argument("--pattern", default="ATAACAGGT")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--seed", type=int, default=0)

    # pair sampling
    ap.add_argument("--n_pairs", type=int, default=20000,
                    help="Random pairs for cross-group distance estimation")

    # embedding view
    ap.add_argument("--view_pos", type=int, default=4,
                    help="Position used for UMAP/PCA scatter")
    ap.add_argument("--view_n", type=int, default=2000,
                    help="Max points per group for UMAP/PCA")

    # classifier
    ap.add_argument("--clf_kfold", type=int, default=5)
    ap.add_argument("--clf_max_iter", type=int, default=500)

    return ap.parse_args()


# -------------------------
# helpers
# -------------------------
def l2norm(x, axis=1, keepdims=True, eps=1e-12):
    return np.sqrt(np.sum(x * x, axis=axis, keepdims=keepdims) + eps)


def cosine_similarity_rows(x1, x2):
    num = np.sum(x1 * x2, axis=1)
    den = l2norm(x1, axis=1, keepdims=False) * l2norm(x2, axis=1, keepdims=False)
    return num / (den + 1e-12)


def cosine_distance_pairs(x1, x2, n_pairs=20000, seed=0):
    rng = np.random.default_rng(seed)
    n1 = len(x1)
    n2 = len(x2)
    i = rng.integers(0, n1, size=n_pairs)
    j = rng.integers(0, n2, size=n_pairs)
    a = x1[i].astype(np.float32)
    b = x2[j].astype(np.float32)
    sim = cosine_similarity_rows(a, b)
    return 1.0 - sim


def dot_product_pairs(x1, x2, n_pairs=20000, seed=0):
    rng = np.random.default_rng(seed)
    n1 = len(x1)
    n2 = len(x2)
    i = rng.integers(0, n1, size=n_pairs)
    j = rng.integers(0, n2, size=n_pairs)
    a = x1[i].astype(np.float32)
    b = x2[j].astype(np.float32)
    return np.sum(a * b, axis=1)


def finite_rows(x):
    return np.isfinite(x).all(axis=1)


def cohen_d_from_scores(a, b, eps=1e-12):
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    ma, mb = np.mean(a), np.mean(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((len(a)-1)*va + (len(b)-1)*vb) / (len(a)+len(b)-2) + eps)
    return (ma - mb) / pooled


def subsample_rows(x, n, seed=0):
    if len(x) <= n:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=n, replace=False)
    return x[idx]


def ensure_pattern_len(pattern, L):
    if len(pattern) == L:
        return pattern
    if len(pattern) > L:
        return pattern[:L]
    return pattern + "?" * (L - len(pattern))


def save_lineplot(y, pattern, title, ylabel, out_path):
    L = len(y)
    plt.figure(figsize=(9, 4), dpi=150)
    plt.plot(range(L), y, marker="o", linewidth=2)
    plt.xticks(range(L), list(pattern))
    plt.xlabel("Base position")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_heatmap(mat, pattern, title, out_path, cmap="viridis", center=None):
    plt.figure(figsize=(6.8, 5.8), dpi=150)
    if center is None:
        im = plt.imshow(mat, aspect="auto", cmap=cmap)
    else:
        vmax = np.nanmax(np.abs(mat))
        im = plt.imshow(mat, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    plt.xticks(range(len(pattern)), list(pattern))
    plt.yticks(range(len(pattern)), list(pattern))
    plt.xlabel("Position (Group B)")
    plt.ylabel("Position (Group A)")
    plt.title(title)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# -------------------------
# main
# -------------------------
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    A = np.load(args.emb_a)  # [N,L,D]
    B = np.load(args.emb_b)  # [M,L,D]

    assert A.ndim == 3 and B.ndim == 3, "Expected embeddings of shape [N,L,D]"
    assert A.shape[1] == B.shape[1] and A.shape[2] == B.shape[2], "A/B shape mismatch on L,D"

    N, L, D = A.shape
    M = B.shape[0]
    pattern = ensure_pattern_len(args.pattern, L)

    # remove rows that are fully nan for some later views only when needed
    summary_rows = []

    # --------------------------------------------------
    # 1) per-position mean cross cosine distance
    # --------------------------------------------------
    pos_cross_cos = np.full(L, np.nan, dtype=np.float32)
    for pos in range(L):
        x1 = A[:, pos, :]
        x2 = B[:, pos, :]
        ok1 = finite_rows(x1)
        ok2 = finite_rows(x2)
        x1 = x1[ok1]
        x2 = x2[ok2]
        if len(x1) == 0 or len(x2) == 0:
            continue
        d = cosine_distance_pairs(x1, x2, n_pairs=args.n_pairs, seed=args.seed + pos)
        pos_cross_cos[pos] = float(np.mean(d))

    save_lineplot(
        pos_cross_cos, pattern,
        title="Per-position mean cross cosine distance",
        ylabel="Mean cosine distance (A vs B)",
        out_path=os.path.join(args.out_dir, "01_pos_mean_cross_cosine.png")
    )

    # --------------------------------------------------
    # 2) pos x pos cross cosine distance heatmap
    # --------------------------------------------------
    pos_pos_cross_cos = np.full((L, L), np.nan, dtype=np.float32)
    for i in range(L):
        xi = A[:, i, :]
        ok_i = finite_rows(xi)
        xi = xi[ok_i]
        if len(xi) == 0:
            continue

        for j in range(L):
            xj = B[:, j, :]
            ok_j = finite_rows(xj)
            xj = xj[ok_j]
            if len(xj) == 0:
                continue

            d = cosine_distance_pairs(
                xi, xj,
                n_pairs=args.n_pairs,
                seed=args.seed + i * 1000 + j
            )
            pos_pos_cross_cos[i, j] = float(np.mean(d))

    save_heatmap(
        pos_pos_cross_cos, pattern,
        title="Cross-position cosine distance",
        out_path=os.path.join(args.out_dir, "02_pos_pos_cross_cosine_heatmap.png"),
        cmap="viridis"
    )

    # --------------------------------------------------
    # 3) per-position mean cross dot product
    # --------------------------------------------------
    pos_cross_dot = np.full(L, np.nan, dtype=np.float32)
    for pos in range(L):
        x1 = A[:, pos, :]
        x2 = B[:, pos, :]
        ok1 = finite_rows(x1)
        ok2 = finite_rows(x2)
        x1 = x1[ok1]
        x2 = x2[ok2]
        if len(x1) == 0 or len(x2) == 0:
            continue
        dp = dot_product_pairs(x1, x2, n_pairs=args.n_pairs, seed=args.seed + 100 + pos)
        pos_cross_dot[pos] = float(np.mean(dp))

    save_lineplot(
        pos_cross_dot, pattern,
        title="Per-position mean cross dot product",
        ylabel="Mean dot product (A vs B)",
        out_path=os.path.join(args.out_dir, "03_pos_mean_dot.png")
    )

    # --------------------------------------------------
    # 4) pos x pos cross dot heatmap
    # --------------------------------------------------
    pos_pos_cross_dot = np.full((L, L), np.nan, dtype=np.float32)
    for i in range(L):
        xi = A[:, i, :]
        ok_i = finite_rows(xi)
        xi = xi[ok_i]
        if len(xi) == 0:
            continue

        for j in range(L):
            xj = B[:, j, :]
            ok_j = finite_rows(xj)
            xj = xj[ok_j]
            if len(xj) == 0:
                continue

            dp = dot_product_pairs(
                xi, xj,
                n_pairs=args.n_pairs,
                seed=args.seed + 2000 + i * 1000 + j
            )
            pos_pos_cross_dot[i, j] = float(np.mean(dp))

    save_heatmap(
        pos_pos_cross_dot, pattern,
        title="Cross-position dot product",
        out_path=os.path.join(args.out_dir, "04_pos_pos_cross_dot_heatmap.png"),
        cmap="viridis"
    )

    # --------------------------------------------------
    # 5) embedding norm per position
    # --------------------------------------------------
    pos_norm_a = np.full(L, np.nan, dtype=np.float32)
    pos_norm_b = np.full(L, np.nan, dtype=np.float32)

    for pos in range(L):
        xa = A[:, pos, :]
        xb = B[:, pos, :]
        oka = finite_rows(xa)
        okb = finite_rows(xb)
        xa = xa[oka]
        xb = xb[okb]

        if len(xa) > 0:
            pos_norm_a[pos] = float(np.mean(np.linalg.norm(xa, axis=1)))
        if len(xb) > 0:
            pos_norm_b[pos] = float(np.mean(np.linalg.norm(xb, axis=1)))

    plt.figure(figsize=(9, 4), dpi=150)
    plt.plot(range(L), pos_norm_a, marker="o", linewidth=2, label=args.label_a)
    plt.plot(range(L), pos_norm_b, marker="o", linewidth=2, label=args.label_b)
    plt.xticks(range(L), list(pattern))
    plt.xlabel("Base position")
    plt.ylabel("Mean embedding norm")
    plt.title("Per-position embedding norm")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "05_pos_norm_mean.png"))
    plt.close()

    # --------------------------------------------------
    # 6) pos x pos norm difference heatmap
    # matrix[i,j] = mean_norm(A_i) - mean_norm(B_j)
    # --------------------------------------------------
    pos_pos_normdiff = np.full((L, L), np.nan, dtype=np.float32)
    for i in range(L):
        for j in range(L):
            if np.isfinite(pos_norm_a[i]) and np.isfinite(pos_norm_b[j]):
                pos_pos_normdiff[i, j] = pos_norm_a[i] - pos_norm_b[j]

    save_heatmap(
        pos_pos_normdiff, pattern,
        title="Cross-position mean norm difference (A - B)",
        out_path=os.path.join(args.out_dir, "06_pos_pos_normdiff_heatmap.png"),
        cmap="coolwarm",
        center=0.0
    )

    # --------------------------------------------------
    # 7) effect size per position (Cohen's d on norm)
    # --------------------------------------------------
    pos_effect_d = np.full(L, np.nan, dtype=np.float32)
    for pos in range(L):
        xa = A[:, pos, :]
        xb = B[:, pos, :]
        oka = finite_rows(xa)
        okb = finite_rows(xb)
        xa = xa[oka]
        xb = xb[okb]
        if len(xa) == 0 or len(xb) == 0:
            continue

        na = np.linalg.norm(xa, axis=1)
        nb = np.linalg.norm(xb, axis=1)
        pos_effect_d[pos] = float(cohen_d_from_scores(na, nb))

    save_lineplot(
        pos_effect_d, pattern,
        title="Per-position effect size on embedding norm",
        ylabel="Cohen's d (norm A vs B)",
        out_path=os.path.join(args.out_dir, "07_pos_effect_size_cohen_d.png")
    )

    # --------------------------------------------------
    # 8) UMAP at one position
    # --------------------------------------------------
    pos = args.view_pos
    assert 0 <= pos < L, f"view_pos must be in [0,{L-1}]"

    xa = A[:, pos, :]
    xb = B[:, pos, :]
    xa = xa[finite_rows(xa)]
    xb = xb[finite_rows(xb)]

    xa = subsample_rows(xa, args.view_n, seed=args.seed + 11)
    xb = subsample_rows(xb, args.view_n, seed=args.seed + 22)

    X = np.vstack([xa, xb]).astype(np.float32)
    y = np.array([0] * len(xa) + [1] * len(xb))

    reducer = umap.UMAP(
        n_neighbors=20,
        min_dist=0.1,
        metric="cosine",
        random_state=args.seed
    )
    Z = reducer.fit_transform(X)

    plt.figure(figsize=(6.2, 5.8), dpi=150)
    plt.scatter(Z[y == 0, 0], Z[y == 0, 1], s=10, alpha=0.6, label=args.label_a)
    plt.scatter(Z[y == 1, 0], Z[y == 1, 1], s=10, alpha=0.6, label=args.label_b)
    plt.legend()
    plt.title(f"UMAP at pos={pos} ({pattern[pos]})")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"08_umap_pos{pos}.png"))
    plt.close()

    # --------------------------------------------------
    # 9) PCA at one position
    # --------------------------------------------------
    pca = PCA(n_components=2, random_state=args.seed)
    Zp = pca.fit_transform(X)

    plt.figure(figsize=(6.2, 5.8), dpi=150)
    plt.scatter(Zp[y == 0, 0], Zp[y == 0, 1], s=10, alpha=0.6, label=args.label_a)
    plt.scatter(Zp[y == 1, 0], Zp[y == 1, 1], s=10, alpha=0.6, label=args.label_b)
    plt.legend()
    evr = pca.explained_variance_ratio_
    plt.title(f"PCA at pos={pos} ({pattern[pos]}) | EVR={evr[0]:.2f},{evr[1]:.2f}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"09_pca_pos{pos}.png"))
    plt.close()

    # --------------------------------------------------
    # 10) linear AUC per position
    # --------------------------------------------------
    pos_auc = np.full(L, np.nan, dtype=np.float32)
    pos_auc_std = np.full(L, np.nan, dtype=np.float32)

    for pos in range(L):
        xa = A[:, pos, :]
        xb = B[:, pos, :]
        Xp = np.vstack([xa, xb]).astype(np.float32)
        yp = np.array([0] * len(xa) + [1] * len(xb))

        ok = np.isfinite(Xp).all(axis=1)
        Xp = Xp[ok]
        yp = yp[ok]

        if len(np.unique(yp)) < 2 or len(Xp) < args.clf_kfold:
            continue

        skf = StratifiedKFold(
            n_splits=args.clf_kfold,
            shuffle=True,
            random_state=args.seed
        )

        aucs = []
        for tr, te in skf.split(Xp, yp):
            clf = LogisticRegression(
                max_iter=args.clf_max_iter,
                n_jobs=1
            )
            clf.fit(Xp[tr], yp[tr])
            prob = clf.predict_proba(Xp[te])[:, 1]
            aucs.append(roc_auc_score(yp[te], prob))

        pos_auc[pos] = float(np.mean(aucs))
        pos_auc_std[pos] = float(np.std(aucs))

    plt.figure(figsize=(9, 4), dpi=150)
    plt.plot(range(L), pos_auc, marker="o", linewidth=2)
    plt.fill_between(
        range(L),
        pos_auc - pos_auc_std,
        pos_auc + pos_auc_std,
        alpha=0.2
    )
    plt.xticks(range(L), list(pattern))
    plt.ylim(0.0, 1.0)
    plt.xlabel("Base position")
    plt.ylabel("ROC AUC")
    plt.title("Linear separability per position")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "10_linear_auc_per_pos.png"))
    plt.close()

    # --------------------------------------------------
    # summary files
    # --------------------------------------------------
    summary_path = os.path.join(args.out_dir, "summary_metrics.tsv")
    with open(summary_path, "w", encoding="utf-8") as fw:
        fw.write("metric\tvalue\n")
        fw.write(f"N_A\t{N}\n")
        fw.write(f"N_B\t{M}\n")
        fw.write(f"L\t{L}\n")
        fw.write(f"D\t{D}\n")
        fw.write(f"view_pos\t{pos}\n")
        fw.write(f"mean_cross_cosine_all_pos\t{np.nanmean(pos_cross_cos):.6f}\n")
        fw.write(f"max_cross_cosine_pos\t{int(np.nanargmax(pos_cross_cos)) if np.isfinite(pos_cross_cos).any() else -1}\n")
        fw.write(f"max_auc_pos\t{int(np.nanargmax(pos_auc)) if np.isfinite(pos_auc).any() else -1}\n")
        fw.write(f"max_auc\t{np.nanmax(pos_auc):.6f}\n")

    auc_tsv = os.path.join(args.out_dir, "linear_auc_per_pos.tsv")
    with open(auc_tsv, "w", encoding="utf-8") as fw:
        fw.write("pos\tbase\tauc_mean\tauc_std\n")
        for i in range(L):
            fw.write(f"{i}\t{pattern[i]}\t{pos_auc[i]:.6f}\t{pos_auc_std[i]:.6f}\n")

    d_tsv = os.path.join(args.out_dir, "effect_size_per_pos.tsv")
    with open(d_tsv, "w", encoding="utf-8") as fw:
        fw.write("pos\tbase\tcohen_d_norm\n")
        for i in range(L):
            fw.write(f"{i}\t{pattern[i]}\t{pos_effect_d[i]:.6f}\n")

    print("[DONE]")
    print("Saved figures to:", args.out_dir)
    print("Summary:", summary_path)
    print("AUC TSV:", auc_tsv)
    print("Effect-size TSV:", d_tsv)


if __name__ == "__main__":
    args = parse_args()
    main(args)