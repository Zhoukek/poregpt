#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import mannwhitneyu


# =========================
# 参数
# =========================
def parse_args():
    p = argparse.ArgumentParser(
        description="Comprehensive comparison of two embedding tensors (N, L, D)"
    )
    p.add_argument("--input1", required=True, help="npy file, shape=(N,L,D)")
    p.add_argument("--input2", required=True, help="npy file, shape=(N,L,D)")
    p.add_argument("--label1", default="Group1")
    p.add_argument("--label2", default="Group2")
    p.add_argument("--out_prefix", required=True)
    p.add_argument("--mod_positions", nargs="+", type=int, required=True)
    p.add_argument("--window_size", type=int, default=1)
    p.add_argument("--tick_step", type=int, default=1)
    p.add_argument("--max_box_positions", type=int, default=150,
                   help="Subsample positions for boxplot if too many")
    return p.parse_args()


# =========================
# 基础
# =========================
def load_npy(path):
    arr = np.load(path)
    if arr.ndim != 3:
        raise ValueError(f"{path} must be shape (N,L,D), got {arr.shape}")
    return arr


def validate_inputs(a, b):
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"Sequence length mismatch: {a.shape[1]} vs {b.shape[1]}")
    if a.shape[2] != b.shape[2]:
        raise ValueError(f"Embedding dim mismatch: {a.shape[2]} vs {b.shape[2]}")


def ensure_outdir(prefix):
    outdir = os.path.dirname(prefix)
    if outdir:
        os.makedirs(outdir, exist_ok=True)


# =========================
# bin
# =========================
def bin_tensor(arr, window_size):
    """
    arr: (N,L,D) -> (N,B,D)
    """
    if window_size <= 1:
        return arr.copy()

    N, L, D = arr.shape
    B = int(np.ceil(L / window_size))
    out = np.zeros((N, B, D), dtype=np.float32)

    for i in range(B):
        s = i * window_size
        e = min((i + 1) * window_size, L)
        out[:, i, :] = arr[:, s:e, :].mean(axis=1)

    return out


def get_bins(L, w):
    B = int(np.ceil(L / w))
    start = np.arange(B) * w
    end = np.array([min((i + 1) * w - 1, L - 1) for i in range(B)])
    return start, end


def is_mod_bin(s, e, mods):
    return any(s <= m <= e for m in mods)


def dist_to_mod(s, e, mods):
    d = []
    for m in mods:
        if s <= m <= e:
            d.append(0)
        elif m < s:
            d.append(s - m)
        else:
            d.append(m - e)
    return min(d) if len(d) > 0 else np.nan


# =========================
# 统计相关
# =========================
def cosine_distance_to_vector(X, v):
    """
    X: (N,D)
    v: (D,)
    """
    dot = np.sum(X * v[None, :], axis=1)
    nx = np.linalg.norm(X, axis=1)
    nv = np.linalg.norm(v)
    return 1.0 - dot / (nx * nv + 1e-8)


def euclidean_distance_to_vector(X, v):
    return np.linalg.norm(X - v[None, :], axis=1)


def cosine_distance_between_vectors(a, b):
    dot = np.sum(a * b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    return 1.0 - dot / (na * nb + 1e-8)


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


def safe_mwu(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    try:
        return mannwhitneyu(x, y, alternative="two-sided").pvalue
    except Exception:
        return np.nan


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


# =========================
# 主统计
# =========================
def compute_all_stats(a, b, mods, label1, label2, window_size):
    """
    a,b: (N,L,D)
    """
    a_bin = bin_tensor(a, window_size)  # (N,B,D)
    b_bin = bin_tensor(b, window_size)

    N1, B, D = a_bin.shape
    N2, B2, D2 = b_bin.shape
    assert B == B2 and D == D2

    L = a.shape[1]
    bs, be = get_bins(L, window_size)

    rows = []
    sample_rows = []

    for i in range(B):
        A = a_bin[:, i, :]   # (N1,D)
        Bx = b_bin[:, i, :]  # (N2,D)

        centroid_a = A.mean(axis=0)
        centroid_b = Bx.mean(axis=0)

        # norm
        norm_a = np.linalg.norm(A, axis=1)
        norm_b = np.linalg.norm(Bx, axis=1)

        # 到对方中心
        a_to_b_cos = cosine_distance_to_vector(A, centroid_b)
        b_to_a_cos = cosine_distance_to_vector(Bx, centroid_a)
        a_to_b_euc = euclidean_distance_to_vector(A, centroid_b)
        b_to_a_euc = euclidean_distance_to_vector(Bx, centroid_a)

        # 到自己中心（组内离散程度）
        a_to_a_cos = cosine_distance_to_vector(A, centroid_a)
        b_to_b_cos = cosine_distance_to_vector(Bx, centroid_b)
        a_to_a_euc = euclidean_distance_to_vector(A, centroid_a)
        b_to_b_euc = euclidean_distance_to_vector(Bx, centroid_b)

        # 两组中心距离
        centroid_cos = cosine_distance_between_vectors(centroid_a, centroid_b)
        centroid_euc = np.linalg.norm(centroid_a - centroid_b)

        # 检验
        p_cos = safe_mwu(a_to_b_cos, b_to_a_cos)
        p_euc = safe_mwu(a_to_b_euc, b_to_a_euc)

        delta_cos = cliffs_delta(a_to_b_cos, b_to_a_cos)
        delta_euc = cliffs_delta(a_to_b_euc, b_to_a_euc)

        row = {
            "bin": i,
            "start": bs[i],
            "end": be[i],
            "is_mod": is_mod_bin(bs[i], be[i], mods),
            "dist_to_mod": dist_to_mod(bs[i], be[i], mods),

            f"{label1}_norm_mean": np.mean(norm_a),
            f"{label1}_norm_median": np.median(norm_a),
            f"{label1}_norm_std": np.std(norm_a, ddof=1) if len(norm_a) > 1 else 0.0,

            f"{label2}_norm_mean": np.mean(norm_b),
            f"{label2}_norm_median": np.median(norm_b),
            f"{label2}_norm_std": np.std(norm_b, ddof=1) if len(norm_b) > 1 else 0.0,

            f"{label1}_self_cos_mean": np.mean(a_to_a_cos),
            f"{label2}_self_cos_mean": np.mean(b_to_b_cos),
            f"{label1}_self_euc_mean": np.mean(a_to_a_euc),
            f"{label2}_self_euc_mean": np.mean(b_to_b_euc),

            f"{label1}_to_{label2}_cos_mean": np.mean(a_to_b_cos),
            f"{label2}_to_{label1}_cos_mean": np.mean(b_to_a_cos),
            f"{label1}_to_{label2}_euc_mean": np.mean(a_to_b_euc),
            f"{label2}_to_{label1}_euc_mean": np.mean(b_to_a_euc),

            "centroid_cosine": centroid_cos,
            "centroid_euclidean": centroid_euc,

            "p_cosine": p_cos,
            "p_euclidean": p_euc,
            "cliffs_delta_cosine": delta_cos,
            "cliffs_delta_euclidean": delta_euc,
        }
        rows.append(row)

        # 样本级长表
        for v in norm_a:
            sample_rows.append({
                "bin": i, "start": bs[i], "end": be[i],
                "is_mod": row["is_mod"], "dist_to_mod": row["dist_to_mod"],
                "group": label1, "metric_type": "norm",
                "metric_name": "norm", "value": v
            })

        for v in norm_b:
            sample_rows.append({
                "bin": i, "start": bs[i], "end": be[i],
                "is_mod": row["is_mod"], "dist_to_mod": row["dist_to_mod"],
                "group": label2, "metric_type": "norm",
                "metric_name": "norm", "value": v
            })

        for v in a_to_b_cos:
            sample_rows.append({
                "bin": i, "start": bs[i], "end": be[i],
                "is_mod": row["is_mod"], "dist_to_mod": row["dist_to_mod"],
                "group": label1, "metric_type": "cross",
                "metric_name": "cosine", "value": v
            })

        for v in b_to_a_cos:
            sample_rows.append({
                "bin": i, "start": bs[i], "end": be[i],
                "is_mod": row["is_mod"], "dist_to_mod": row["dist_to_mod"],
                "group": label2, "metric_type": "cross",
                "metric_name": "cosine", "value": v
            })

        for v in a_to_b_euc:
            sample_rows.append({
                "bin": i, "start": bs[i], "end": be[i],
                "is_mod": row["is_mod"], "dist_to_mod": row["dist_to_mod"],
                "group": label1, "metric_type": "cross",
                "metric_name": "euclidean", "value": v
            })

        for v in b_to_a_euc:
            sample_rows.append({
                "bin": i, "start": bs[i], "end": be[i],
                "is_mod": row["is_mod"], "dist_to_mod": row["dist_to_mod"],
                "group": label2, "metric_type": "cross",
                "metric_name": "euclidean", "value": v
            })

    pos_df = pd.DataFrame(rows)
    sample_df = pd.DataFrame(sample_rows)

    # FDR
    for col_p, col_q in [
        ("p_cosine", "fdr_cosine"),
        ("p_euclidean", "fdr_euclidean")
    ]:
        qvals = benjamini_hochberg(pos_df[col_p].values)
        pos_df[col_q] = qvals

    pos_df["minus_log10_fdr_cosine"] = -np.log10(pos_df["fdr_cosine"].fillna(1.0) + 1e-300)
    pos_df["minus_log10_fdr_euclidean"] = -np.log10(pos_df["fdr_euclidean"].fillna(1.0) + 1e-300)

    return pos_df, sample_df


# =========================
# 绘图辅助
# =========================
def add_mod_shading(ax, pos_df):
    for _, r in pos_df.iterrows():
        if r["is_mod"]:
            ax.axvspan(r["bin"] - 0.5, r["bin"] + 0.5, color="#f04e4e", alpha=0.10, lw=0)


def set_xticks(ax, pos_df, tick_step):
    bins = pos_df["bin"].values
    tick_idx = np.arange(0, len(bins), max(1, tick_step))
    labels = [f"{int(pos_df.iloc[i]['start'])}-{int(pos_df.iloc[i]['end'])}" for i in tick_idx]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(labels, rotation=45, ha="right")


def sample_positions_for_boxplot(pos_df, max_positions):
    bins = pos_df["bin"].values
    if len(bins) <= max_positions:
        return bins
    idx = np.linspace(0, len(bins) - 1, max_positions).astype(int)
    return bins[idx]


def make_boxplot(ax, df, positions, group_col="group", value_col="value",
                 label_order=None, title=None, ylabel=None):
    if label_order is None:
        label_order = sorted(df[group_col].unique())

    width = 0.35
    offsets = np.linspace(-width / 2, width / 2, len(label_order))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]

    for gi, g in enumerate(label_order):
        data = []
        xs = []
        sub = df[df[group_col] == g]
        for p in positions:
            vals = sub.loc[sub["bin"] == p, value_col].values
            data.append(vals)
            xs.append(p + offsets[gi])

        bp = ax.boxplot(
            data,
            positions=xs,
            widths=width / max(1, len(label_order)),
            patch_artist=True,
            showfliers=False,
            manage_ticks=False
        )

        for box in bp["boxes"]:
            box.set(facecolor=colors[gi % len(colors)], alpha=0.6, edgecolor="black", linewidth=0.8)
        for med in bp["medians"]:
            med.set(color="black", linewidth=1.0)
        for w in bp["whiskers"]:
            w.set(color="black", linewidth=0.8)
        for c in bp["caps"]:
            c.set(color="black", linewidth=0.8)

    if title:
        ax.set_title(title, fontsize=12, weight="bold")
    if ylabel:
        ax.set_ylabel(ylabel)

    ax.grid(axis="y", alpha=0.25)


# =========================
# 出图
# =========================
def plot_comprehensive_pdf(pos_df, sample_df, out_pdf, label1, label2, tick_step, max_box_positions):
    with PdfPages(out_pdf) as pdf:
        # Page 1
        fig, axes = plt.subplots(3, 1, figsize=(18, 12), constrained_layout=True)

        ax = axes[0]
        ax.plot(pos_df["bin"], pos_df["centroid_cosine"], lw=2, label="Centroid cosine distance")
        add_mod_shading(ax, pos_df)
        set_xticks(ax, pos_df, tick_step)
        ax.set_ylabel("Cosine distance")
        ax.set_title("Group centroid distance across positions", fontsize=14, weight="bold")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)

        ax = axes[1]
        ax.plot(pos_df["bin"], pos_df["centroid_euclidean"], lw=2, label="Centroid euclidean distance")
        add_mod_shading(ax, pos_df)
        set_xticks(ax, pos_df, tick_step)
        ax.set_ylabel("Euclidean distance")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)

        ax = axes[2]
        ax.plot(pos_df["bin"], pos_df["minus_log10_fdr_cosine"], lw=2, label="-log10(FDR), cosine")
        ax.plot(pos_df["bin"], pos_df["minus_log10_fdr_euclidean"], lw=2, label="-log10(FDR), euclidean")
        ax.axhline(-np.log10(0.05), ls="--", lw=1, color="red", alpha=0.8, label="FDR=0.05")
        add_mod_shading(ax, pos_df)
        set_xticks(ax, pos_df, tick_step)
        ax.set_ylabel("-log10(FDR)")
        ax.set_xlabel("Position bin")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)

        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        # Page 2
        fig, axes = plt.subplots(3, 1, figsize=(18, 12), constrained_layout=True)

        ax = axes[0]
        ax.plot(pos_df["bin"], pos_df["cliffs_delta_cosine"], lw=2, label="Cliff's delta (cosine)")
        ax.plot(pos_df["bin"], pos_df["cliffs_delta_euclidean"], lw=2, label="Cliff's delta (euclidean)")
        ax.axhline(0, color="black", lw=1)
        add_mod_shading(ax, pos_df)
        set_xticks(ax, pos_df, tick_step)
        ax.set_ylabel("Effect size")
        ax.set_title("Effect size across positions", fontsize=14, weight="bold")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)

        ax = axes[1]
        ax.plot(pos_df["bin"], pos_df[f"{label1}_self_cos_mean"], lw=2, label=f"{label1} self cosine")
        ax.plot(pos_df["bin"], pos_df[f"{label2}_self_cos_mean"], lw=2, label=f"{label2} self cosine")
        add_mod_shading(ax, pos_df)
        set_xticks(ax, pos_df, tick_step)
        ax.set_ylabel("Self cosine distance")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)

        ax = axes[2]
        ax.plot(pos_df["bin"], pos_df[f"{label1}_self_euc_mean"], lw=2, label=f"{label1} self euclidean")
        ax.plot(pos_df["bin"], pos_df[f"{label2}_self_euc_mean"], lw=2, label=f"{label2} self euclidean")
        add_mod_shading(ax, pos_df)
        set_xticks(ax, pos_df, tick_step)
        ax.set_ylabel("Self euclidean distance")
        ax.set_xlabel("Position bin")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)

        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        # Page 3: norm boxplot
        positions = sample_positions_for_boxplot(pos_df, max_box_positions)
        norm_df = sample_df[(sample_df["metric_type"] == "norm") & (sample_df["bin"].isin(positions))]

        fig, ax = plt.subplots(figsize=(20, 6), constrained_layout=True)
        make_boxplot(
            ax, norm_df, positions,
            group_col="group", value_col="value",
            label_order=[label1, label2],
            title="Position-wise embedding norm distribution",
            ylabel="Embedding norm"
        )
        add_mod_shading(ax, pos_df[pos_df["bin"].isin(positions)].reset_index(drop=True))
        ax.set_xticks(positions)
        ax.set_xticklabels([str(int(p)) for p in positions], rotation=45, ha="right")
        ax.set_xlabel("Position bin")

        handles = [
            plt.Line2D([0], [0], color="#4C78A8", lw=8, alpha=0.6),
            plt.Line2D([0], [0], color="#F58518", lw=8, alpha=0.6),
        ]
        ax.legend(handles=handles, labels=[label1, label2], frameon=False, loc="upper right")

        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        # Page 4: cross distance boxplots
        for metric_name, ylabel in [("cosine", "Cross-group cosine distance"),
                                    ("euclidean", "Cross-group euclidean distance")]:

            cross_df = sample_df[
                (sample_df["metric_type"] == "cross") &
                (sample_df["metric_name"] == metric_name) &
                (sample_df["bin"].isin(positions))
            ]

            fig, ax = plt.subplots(figsize=(20, 6), constrained_layout=True)
            make_boxplot(
                ax, cross_df, positions,
                group_col="group", value_col="value",
                label_order=[label1, label2],
                title=f"Position-wise {ylabel} distribution",
                ylabel=ylabel
            )
            add_mod_shading(ax, pos_df[pos_df["bin"].isin(positions)].reset_index(drop=True))
            ax.set_xticks(positions)
            ax.set_xticklabels([str(int(p)) for p in positions], rotation=45, ha="right")
            ax.set_xlabel("Position bin")

            handles = [
                plt.Line2D([0], [0], color="#4C78A8", lw=8, alpha=0.6),
                plt.Line2D([0], [0], color="#F58518", lw=8, alpha=0.6),
            ]
            ax.legend(handles=handles, labels=[label1, label2], frameon=False, loc="upper right")

            pdf.savefig(fig, dpi=300)
            plt.close(fig)

        # Page 5: histogram
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

        cross_cos = sample_df[(sample_df["metric_type"] == "cross") & (sample_df["metric_name"] == "cosine")]
        cross_euc = sample_df[(sample_df["metric_type"] == "cross") & (sample_df["metric_name"] == "euclidean")]
        norm_all = sample_df[sample_df["metric_type"] == "norm"]

        hist_configs = [
            (axes[0, 0], cross_cos, "Global histogram: cross cosine", "Cosine distance"),
            (axes[0, 1], cross_euc, "Global histogram: cross euclidean", "Euclidean distance"),
            (axes[1, 0], norm_all[norm_all["group"] == label1], f"Global histogram: norm ({label1})", "Norm"),
            (axes[1, 1], norm_all[norm_all["group"] == label2], f"Global histogram: norm ({label2})", "Norm"),
        ]

        for ax, df_, title, xlabel in hist_configs:
            ax.hist(df_["value"], bins=60, alpha=0.80, edgecolor="white")
            ax.set_title(title, fontsize=12, weight="bold")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Count")
            ax.grid(alpha=0.25)

        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        # Page 6: mod vs background
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

        tmp_df = pos_df.copy()
        tmp_df["region"] = np.where(tmp_df["is_mod"], "mod", "background")

        configs = [
            (axes[0, 0], "centroid_cosine", "Centroid cosine: mod vs background"),
            (axes[0, 1], "centroid_euclidean", "Centroid euclidean: mod vs background"),
            (axes[1, 0], "minus_log10_fdr_cosine", "-log10(FDR) cosine: mod vs background"),
            (axes[1, 1], "minus_log10_fdr_euclidean", "-log10(FDR) euclidean: mod vs background"),
        ]

        for ax, col, title in configs:
            data = [
                tmp_df.loc[tmp_df["region"] == "background", col].dropna().values,
                tmp_df.loc[tmp_df["region"] == "mod", col].dropna().values
            ]
            bp = ax.boxplot(data, patch_artist=True, showfliers=False)
            colors = ["#9ecae1", "#fb6a4a"]
            for box, c in zip(bp["boxes"], colors):
                box.set(facecolor=c, alpha=0.7, edgecolor="black")
            for med in bp["medians"]:
                med.set(color="black")
            ax.set_xticks([1, 2])
            ax.set_xticklabels(["background", "mod"])
            ax.set_title(title, fontsize=12, weight="bold")
            ax.grid(axis="y", alpha=0.25)

        pdf.savefig(fig, dpi=300)
        plt.close(fig)


# =========================
# summary
# =========================
def write_summary(pos_df, out_txt):
    lines = []
    lines.append("=== Overall summary ===")

    for metric in [
        "centroid_cosine",
        "centroid_euclidean",
        "minus_log10_fdr_cosine",
        "minus_log10_fdr_euclidean"
    ]:
        mod_vals = pos_df.loc[pos_df["is_mod"], metric].dropna().values
        bg_vals = pos_df.loc[~pos_df["is_mod"], metric].dropna().values

        lines.append(f"\n[{metric}]")
        lines.append(f"mod_mean={np.mean(mod_vals) if len(mod_vals) else np.nan:.6f}")
        lines.append(f"bg_mean={np.mean(bg_vals) if len(bg_vals) else np.nan:.6f}")
        lines.append(f"mod_median={np.median(mod_vals) if len(mod_vals) else np.nan:.6f}")
        lines.append(f"bg_median={np.median(bg_vals) if len(bg_vals) else np.nan:.6f}")

    lines.append("\n=== Top positions by centroid cosine ===")
    tmp = pos_df.sort_values("centroid_cosine", ascending=False).head(20)
    lines.append(tmp[["bin", "start", "end", "is_mod", "centroid_cosine", "fdr_cosine"]].to_string(index=False))

    lines.append("\n=== Top positions by centroid euclidean ===")
    tmp = pos_df.sort_values("centroid_euclidean", ascending=False).head(20)
    lines.append(tmp[["bin", "start", "end", "is_mod", "centroid_euclidean", "fdr_euclidean"]].to_string(index=False))

    lines.append("\n=== Top significant positions (cosine) ===")
    tmp = pos_df.sort_values("fdr_cosine", ascending=True).head(20)
    lines.append(tmp[["bin", "start", "end", "is_mod", "centroid_cosine", "fdr_cosine", "cliffs_delta_cosine"]].to_string(index=False))

    lines.append("\n=== Top significant positions (euclidean) ===")
    tmp = pos_df.sort_values("fdr_euclidean", ascending=True).head(20)
    lines.append(tmp[["bin", "start", "end", "is_mod", "centroid_euclidean", "fdr_euclidean", "cliffs_delta_euclidean"]].to_string(index=False))

    with open(out_txt, "w") as f:
        f.write("\n".join(lines))


# =========================
# main
# =========================
def main():
    args = parse_args()
    ensure_outdir(args.out_prefix)

    a = load_npy(args.input1)
    b = load_npy(args.input2)
    validate_inputs(a, b)

    pos_df, sample_df = compute_all_stats(
        a, b,
        mods=args.mod_positions,
        label1=args.label1,
        label2=args.label2,
        window_size=args.window_size
    )

    pos_csv = args.out_prefix + ".position_stats.csv"
    sample_csv = args.out_prefix + ".sample_level_stats.csv"
    pdf_file = args.out_prefix + ".comprehensive_plots.pdf"
    txt_file = args.out_prefix + ".summary.txt"

    pos_df.to_csv(pos_csv, index=False)
    sample_df.to_csv(sample_csv, index=False)

    plot_comprehensive_pdf(
        pos_df, sample_df, pdf_file,
        label1=args.label1,
        label2=args.label2,
        tick_step=args.tick_step,
        max_box_positions=args.max_box_positions
    )

    write_summary(pos_df, txt_file)

    print("Done.")
    print("Position stats :", pos_csv)
    print("Sample stats   :", sample_csv)
    print("Plots PDF      :", pdf_file)
    print("Summary text   :", txt_file)


if __name__ == "__main__":
    main()