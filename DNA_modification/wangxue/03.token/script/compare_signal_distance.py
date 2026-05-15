#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import mannwhitneyu, kruskal

# -----------------------------
# Global plot style
# -----------------------------
plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COLOR1 = "#C44E52"        # muted red
COLOR2 = "#4C72B0"        # muted blue
HIGHLIGHT_COLOR = "#D9EDF7"
DELTA_COLOR = "#2E8B57"

# -----------------------------
# Utilities
# -----------------------------
def ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

def metric_label(metric: str) -> str:
    mapping = {
        "mean": "Mean current",
        "std": "Current std",
        "skewness": "Current skewness",
        "dwell_time": "Dwell time (samples)",
        "dwell_time_ms": "Dwell time (ms)",
        "median": "Median current",
        "range": "Current range",
        "q25": "Current Q25",
        "q75": "Current Q75",
    }
    return mapping.get(metric, metric)

def p_to_star(p):
    if p is None or not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"

def sem(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) <= 1:
        return np.nan
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))

def cohens_d(x1: np.ndarray, x2: np.ndarray) -> float:
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    x1 = x1[np.isfinite(x1)]
    x2 = x2[np.isfinite(x2)]

    if len(x1) < 2 or len(x2) < 2:
        return np.nan

    s1 = np.std(x1, ddof=1)
    s2 = np.std(x2, ddof=1)
    n1 = len(x1)
    n2 = len(x2)

    pooled_var = ((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2)
    if pooled_var <= 0:
        return np.nan

    pooled_sd = np.sqrt(pooled_var)
    return float((np.mean(x1) - np.mean(x2)) / pooled_sd)

def eta_squared_kruskal(H: float, k: int, n: int) -> float:
    if n <= k or not np.isfinite(H):
        return np.nan
    val = (H - k + 1) / (n - k)
    return float(max(0.0, val))

def bh_adjust(pvals: List[float]) -> List[float]:
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)

    out = np.full(n, np.nan, dtype=float)
    valid = np.isfinite(pvals)
    if valid.sum() == 0:
        return out.tolist()

    pv = pvals[valid]
    order = np.argsort(pv)
    ranked = pv[order]

    adj = np.empty_like(ranked)
    prev = 1.0
    m = len(ranked)

    for i in range(m - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * m / rank
        prev = min(prev, val)
        adj[i] = min(prev, 1.0)

    restored = np.empty_like(adj)
    restored[order] = adj
    out[valid] = restored
    return out.tolist()

def summarize_array(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "sem": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
        "sem": sem(x),
        "q25": float(np.percentile(x, 25)),
        "q75": float(np.percentile(x, 75)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }

def load_csv(path: str, metric: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = ["base_idx_in_pattern", metric]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"{path} 缺少列: {col}")

    df["base_idx_in_pattern"] = pd.to_numeric(df["base_idx_in_pattern"], errors="coerce")
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=["base_idx_in_pattern", metric]).copy()
    df["base_idx_in_pattern"] = df["base_idx_in_pattern"].astype(int)

    if "base" in df.columns:
        df["base"] = df["base"].astype(str)

    return df

def extract_positions(df1: pd.DataFrame, df2: pd.DataFrame) -> List[int]:
    pos = sorted(set(df1["base_idx_in_pattern"].unique()) | set(df2["base_idx_in_pattern"].unique()))
    return [int(x) for x in pos]

# -----------------------------
# Statistics
# -----------------------------
def compute_stats_table(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    metric: str,
    label1: str,
    label2: str,
    highlight: List[int],
) -> pd.DataFrame:
    positions = extract_positions(df1, df2)
    rows = []

    for p in positions:
        x1 = df1.loc[df1["base_idx_in_pattern"] == p, metric].dropna().values
        x2 = df2.loc[df2["base_idx_in_pattern"] == p, metric].dropna().values

        s1 = summarize_array(x1)
        s2 = summarize_array(x2)

        if len(x1) > 0 and len(x2) > 0:
            try:
                stat, pval = mannwhitneyu(x1, x2, alternative="two-sided")
            except Exception:
                stat, pval = np.nan, np.nan
        else:
            stat, pval = np.nan, np.nan

        delta_mean = (
            (s1["mean"] - s2["mean"])
            if np.isfinite(s1["mean"]) and np.isfinite(s2["mean"])
            else np.nan
        )

        effect = cohens_d(x1, x2)

        row = {
            "metric": metric,
            "position": p,
            "is_highlight": int(p in highlight),

            f"{label1}_n": s1["n"],
            f"{label1}_mean": s1["mean"],
            f"{label1}_median": s1["median"],
            f"{label1}_std": s1["std"],
            f"{label1}_sem": s1["sem"],
            f"{label1}_q25": s1["q25"],
            f"{label1}_q75": s1["q75"],
            f"{label1}_min": s1["min"],
            f"{label1}_max": s1["max"],

            f"{label2}_n": s2["n"],
            f"{label2}_mean": s2["mean"],
            f"{label2}_median": s2["median"],
            f"{label2}_std": s2["std"],
            f"{label2}_sem": s2["sem"],
            f"{label2}_q25": s2["q25"],
            f"{label2}_q75": s2["q75"],
            f"{label2}_min": s2["min"],
            f"{label2}_max": s2["max"],

            "delta_mean": delta_mean,
            "effect_size_cohens_d": effect,
            "mw_stat": stat,
            "p_value": pval,
            "test_name": "mannwhitneyu",
        }
        rows.append(row)

    stats_df = pd.DataFrame(rows)
    stats_df["adj_p_value_bh"] = bh_adjust(stats_df["p_value"].values.tolist())
    return stats_df

def save_stats_table(stats_df: pd.DataFrame, output_pdf: str, metric: str):
    if output_pdf.endswith(".pdf"):
        out_csv = output_pdf.replace(".pdf", f"_{metric}.stats.csv")
    else:
        out_csv = output_pdf + f"_{metric}.stats.csv"
    stats_df.to_csv(out_csv, index=False)
    print(f"[OK] saved stats: {out_csv}")

def global_within_group_test(df: pd.DataFrame, metric: str, label: str):
    positions = sorted(df["base_idx_in_pattern"].dropna().unique().astype(int).tolist())
    arrays = []
    ns = []

    for p in positions:
        vals = df.loc[df["base_idx_in_pattern"] == p, metric].dropna().values.astype(float)
        if len(vals) > 0:
            arrays.append(vals)
            ns.append(len(vals))

    if len(arrays) >= 2:
        try:
            stat, pval = kruskal(*arrays)
        except Exception:
            stat, pval = np.nan, np.nan
    else:
        stat, pval = np.nan, np.nan

    total_n = int(np.sum(ns)) if len(ns) > 0 else 0
    effect = eta_squared_kruskal(stat, len(arrays), total_n) if len(arrays) >= 2 else np.nan

    pooled = df[metric].dropna().values.astype(float)

    return {
        "comparison": f"{label}_within_group",
        "test_name": "kruskal",
        "statistic": stat,
        "p_value": pval,
        "effect_size": effect,
        "num_positions": len(arrays),
        "total_n": total_n,
        "pooled_values": pooled,
        "summary": summarize_array(pooled),
    }

def global_between_group_test(df1: pd.DataFrame, df2: pd.DataFrame, metric: str, label1: str, label2: str):
    x1 = df1[metric].dropna().values.astype(float)
    x2 = df2[metric].dropna().values.astype(float)

    if len(x1) > 0 and len(x2) > 0:
        try:
            stat, pval = mannwhitneyu(x1, x2, alternative="two-sided")
        except Exception:
            stat, pval = np.nan, np.nan
    else:
        stat, pval = np.nan, np.nan

    effect = cohens_d(x1, x2)

    return {
        "comparison": f"{label1}_vs_{label2}",
        "test_name": "mannwhitneyu",
        "statistic": stat,
        "p_value": pval,
        "effect_size": effect,
        f"{label1}_n": len(x1),
        f"{label2}_n": len(x2),
        f"{label1}_summary": summarize_array(x1),
        f"{label2}_summary": summarize_array(x2),
        "values1": x1,
        "values2": x2,
    }

def save_global_stats_table(
    output_pdf: str,
    metric: str,
    label1: str,
    label2: str,
    within_a: dict,
    within_b: dict,
    between_ab: dict,
):
    if output_pdf.endswith(".pdf"):
        out_csv = output_pdf.replace(".pdf", f"_{metric}.global_stats.csv")
    else:
        out_csv = output_pdf + f"_{metric}.global_stats.csv"

    rows = [
        {
            "metric": metric,
            "comparison": f"{label1}_within",
            "test_name": within_a["test_name"],
            "statistic": within_a["statistic"],
            "p_value": within_a["p_value"],
            "effect_size": within_a["effect_size"],
            "num_positions": within_a["num_positions"],
            "total_n": within_a["total_n"],
            "mean": within_a["summary"]["mean"],
            "median": within_a["summary"]["median"],
            "std": within_a["summary"]["std"],
            "q25": within_a["summary"]["q25"],
            "q75": within_a["summary"]["q75"],
        },
        {
            "metric": metric,
            "comparison": f"{label2}_within",
            "test_name": within_b["test_name"],
            "statistic": within_b["statistic"],
            "p_value": within_b["p_value"],
            "effect_size": within_b["effect_size"],
            "num_positions": within_b["num_positions"],
            "total_n": within_b["total_n"],
            "mean": within_b["summary"]["mean"],
            "median": within_b["summary"]["median"],
            "std": within_b["summary"]["std"],
            "q25": within_b["summary"]["q25"],
            "q75": within_b["summary"]["q75"],
        },
        {
            "metric": metric,
            "comparison": f"{label1}_vs_{label2}",
            "test_name": between_ab["test_name"],
            "statistic": between_ab["statistic"],
            "p_value": between_ab["p_value"],
            "effect_size": between_ab["effect_size"],
            f"{label1}_n": between_ab[f"{label1}_n"],
            f"{label2}_n": between_ab[f"{label2}_n"],
            f"{label1}_mean": between_ab[f"{label1}_summary"]["mean"],
            f"{label2}_mean": between_ab[f"{label2}_summary"]["mean"],
            f"{label1}_median": between_ab[f"{label1}_summary"]["median"],
            f"{label2}_median": between_ab[f"{label2}_summary"]["median"],
            f"{label1}_std": between_ab[f"{label1}_summary"]["std"],
            f"{label2}_std": between_ab[f"{label2}_summary"]["std"],
        },
    ]

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[OK] saved global stats: {out_csv}")

# -----------------------------
# Plot helpers
# -----------------------------
def style_boxplot(bp, color: str, alpha: float = 0.38):
    for box in bp["boxes"]:
        box.set_facecolor(color)
        box.set_edgecolor(color)
        box.set_alpha(alpha)
    for whisker in bp["whiskers"]:
        whisker.set_color(color)
    for cap in bp["caps"]:
        cap.set_color(color)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.2)

def add_stat_box(ax, lines: List[str]):
    ax.text(
        0.98, 0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="lightgray",
            alpha=0.92
        )
    )

# -----------------------------
# Plot 1: original per-position main figure (NO stats)
# -----------------------------
def boxplot_two_groups_figure(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    metric: str,
    label1: str,
    label2: str,
    highlight: List[int],
    title: str = None,
):
    positions = extract_positions(df1, df2)
    if len(positions) == 0:
        raise ValueError("没有可画的 position。")

    data1 = [
        df1.loc[df1["base_idx_in_pattern"] == p, metric].dropna().values
        for p in positions
    ]
    data2 = [
        df2.loc[df2["base_idx_in_pattern"] == p, metric].dropna().values
        for p in positions
    ]

    fig_w = max(10, len(positions) * 0.42)
    fig, ax = plt.subplots(figsize=(fig_w, 5.8))

    for p in positions:
        if p in highlight:
            ax.axvspan(p - 0.5, p + 0.5, color=HIGHLIGHT_COLOR, alpha=0.6, zorder=0)

    offset = 0.18
    width = 0.28

    bp1 = ax.boxplot(
        data1,
        positions=[p - offset for p in positions],
        widths=width,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
        boxprops=dict(linewidth=1.0),
        zorder=3,
    )
    bp2 = ax.boxplot(
        data2,
        positions=[p + offset for p in positions],
        widths=width,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
        boxprops=dict(linewidth=1.0),
        zorder=3,
    )

    style_boxplot(bp1, COLOR1, alpha=0.35)
    style_boxplot(bp2, COLOR2, alpha=0.35)

    mean1 = [np.nanmean(x) if len(x) > 0 else np.nan for x in data1]
    mean2 = [np.nanmean(x) if len(x) > 0 else np.nan for x in data2]

    ax.plot(
        [p - offset for p in positions],
        mean1,
        color=COLOR1,
        marker="o",
        linewidth=1.6,
        markersize=4.5,
        zorder=4,
    )
    ax.plot(
        [p + offset for p in positions],
        mean2,
        color=COLOR2,
        marker="o",
        linewidth=1.6,
        markersize=4.5,
        zorder=4,
    )

    if metric == "skewness":
        ax.axhline(0, linestyle="--", linewidth=1.0, color="gray", zorder=1)

    ax.set_xlim(min(positions) - 1, max(positions) + 1)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(p) for p in positions], rotation=0)
    ax.set_xlabel("Position in pattern")
    ax.set_ylabel(metric_label(metric))
    ax.set_title(title if title else metric_label(metric))

    handles = [
        Line2D([0], [0], color=COLOR1, lw=2, marker="s", markersize=8, label=label1),
        Line2D([0], [0], color=COLOR2, lw=2, marker="s", markersize=8, label=label2),
    ]
    if highlight:
        handles.append(Line2D([0], [0], color=HIGHLIGHT_COLOR, lw=8, label="Highlighted positions"))
    ax.legend(handles=handles, frameon=False, loc="best")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")

    fig.tight_layout()
    return fig

# -----------------------------
# Plot 2: original global summary (KEEP)
# -----------------------------
def global_summary_figure(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    stats_df: pd.DataFrame,
    metric: str,
    label1: str,
    label2: str,
    highlight: List[int],
    title: str = None,
):
    positions = extract_positions(df1, df2)
    global_stats = global_between_group_test(df1, df2, metric, label1, label2)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8))
    ax1, ax2, ax3, ax4 = axes

    x1 = df1[metric].dropna().values.astype(float)
    x2 = df2[metric].dropna().values.astype(float)

    parts = ax1.violinplot(
        [x1, x2],
        positions=[1, 2],
        widths=0.7,
        showmeans=False,
        showmedians=True,
        showextrema=False
    )
    for i, body in enumerate(parts["bodies"]):
        body.set_alpha(0.30)
        body.set_facecolor(COLOR1 if i == 0 else COLOR2)
        body.set_edgecolor(COLOR1 if i == 0 else COLOR2)

    if "cmedians" in parts:
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.2)

    bp = ax1.boxplot(
        [x1, x2],
        positions=[1, 2],
        widths=0.22,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
        boxprops=dict(linewidth=1.0),
    )
    for i, box in enumerate(bp["boxes"]):
        c = COLOR1 if i == 0 else COLOR2
        box.set_facecolor(c)
        box.set_edgecolor(c)
        box.set_alpha(0.45)

    ax1.set_xticks([1, 2])
    ax1.set_xticklabels([label1, label2])
    ax1.set_ylabel(metric_label(metric))
    ax1.set_title("Overall distribution")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    star = p_to_star(global_stats["p_value"])
    if np.isfinite(global_stats["p_value"]):
        text = f"{star}\np={global_stats['p_value']:.2e}"
    else:
        text = "p=NA"
    if np.isfinite(global_stats["effect_size"]):
        text += f"\nd={global_stats['effect_size']:.2f}"
    text += f"\nn={global_stats[f'{label1}_n']} | {global_stats[f'{label2}_n']}"

    y_top = max(np.nanmax(x1) if len(x1) else np.nan, np.nanmax(x2) if len(x2) else np.nan)
    y_bottom = min(np.nanmin(x1) if len(x1) else np.nan, np.nanmin(x2) if len(x2) else np.nan)
    if np.isfinite(y_top) and np.isfinite(y_bottom):
        yr = y_top - y_bottom if y_top > y_bottom else 1.0
        ax1.text(1.5, y_top + 0.08 * yr, text, ha="center", va="bottom", fontsize=9)
        ax1.set_ylim(y_bottom - 0.08 * yr, y_top + 0.28 * yr)

    if metric == "skewness":
        ax1.axhline(0, linestyle="--", linewidth=1.0, color="gray")

    plot_df = stats_df.copy().sort_values("position")
    x = plot_df["position"].values
    fdr = plot_df["adj_p_value_bh"].values.astype(float)
    delta = plot_df["delta_mean"].values.astype(float)

    neglog10_fdr = np.where(np.isfinite(fdr) & (fdr > 0), -np.log10(fdr), np.nan)

    for p in x:
        if p in highlight:
            ax2.axvspan(p - 0.5, p + 0.5, color=HIGHLIGHT_COLOR, alpha=0.6, zorder=0)

    ax2.bar(x, neglog10_fdr, color="#7A7A7A", alpha=0.7, width=0.75, label="-log10(FDR)")
    ax2.axhline(-np.log10(0.05), linestyle="--", linewidth=1.0, color="gray")

    ax2_t = ax2.twinx()
    ax2_t.plot(x, delta, color="#2E8B57", marker="o", linewidth=1.5, markersize=4, label="delta_mean")

    ax2.set_xlabel("Position in pattern")
    ax2.set_ylabel("-log10(FDR)")
    ax2_t.set_ylabel("Delta mean")
    ax2.set_title("Between-group difference")

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2_t.spines["top"].set_visible(False)

    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_t.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, frameon=False, loc="best")

    def agg_group(df):
        g = (
            df.groupby("base_idx_in_pattern")[metric]
            .agg(["mean", "count"])
            .reset_index()
            .sort_values("base_idx_in_pattern")
        )
        sems = []
        for p in g["base_idx_in_pattern"].values:
            vals = df.loc[df["base_idx_in_pattern"] == p, metric].dropna().values
            sems.append(sem(vals))
        g["sem"] = sems
        return g

    g1 = agg_group(df1)
    g2 = agg_group(df2)

    for p in positions:
        if p in highlight:
            ax3.axvspan(p - 0.5, p + 0.5, color=HIGHLIGHT_COLOR, alpha=0.6, zorder=0)

    ax3.plot(g1["base_idx_in_pattern"], g1["mean"], color=COLOR1, marker="o", linewidth=1.8, label=label1)
    ax3.plot(g2["base_idx_in_pattern"], g2["mean"], color=COLOR2, marker="o", linewidth=1.8, label=label2)

    v1 = np.isfinite(g1["sem"].values)
    if v1.any():
        ax3.fill_between(
            g1["base_idx_in_pattern"].values[v1],
            (g1["mean"] - g1["sem"]).values[v1],
            (g1["mean"] + g1["sem"]).values[v1],
            color=COLOR1, alpha=0.18
        )

    v2 = np.isfinite(g2["sem"].values)
    if v2.any():
        ax3.fill_between(
            g2["base_idx_in_pattern"].values[v2],
            (g2["mean"] - g2["sem"]).values[v2],
            (g2["mean"] + g2["sem"]).values[v2],
            color=COLOR2, alpha=0.18
        )

    if metric == "skewness":
        ax3.axhline(0, linestyle="--", linewidth=1.0, color="gray")

    ax3.set_xlabel("Position in pattern")
    ax3.set_ylabel(metric_label(metric))
    ax3.set_title("Within-group structure")
    ax3.legend(frameon=False, loc="best")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    for p in x:
        if p in highlight:
            ax4.axvspan(p - 0.5, p + 0.5, color=HIGHLIGHT_COLOR, alpha=0.6, zorder=0)

    bar_colors = [
        "#2E8B57" if np.isfinite(v) and v >= 0 else "#B04A5A"
        for v in delta
    ]

    ax4.bar(
        x,
        delta,
        color=bar_colors,
        alpha=0.80,
        width=0.75
    )
    ax4.axhline(0, linestyle="--", linewidth=1.0, color="gray")

    ax4.set_xlabel("Position in pattern")
    ax4.set_ylabel("Delta mean")
    ax4.set_title("Delta mean across positions")
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    delta_valid = delta[np.isfinite(delta)]
    if len(delta_valid) > 0:
        s = summarize_array(delta_valid)
        txt = (
            f"n={s['n']}\n"
            f"mean={s['mean']:.3f}\n"
            f"median={s['median']:.3f}\n"
            f"std={s['std']:.3f}"
        )
        ax4.text(
            0.98, 0.98,
            txt,
            transform=ax4.transAxes,
            ha="right", va="top",
            fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="lightgray",
                alpha=0.92
            )
        )

    fig.suptitle(title if title else f"{metric_label(metric)} summary", y=1.02, fontsize=13)
    fig.tight_layout()
    return fig

# -----------------------------
# Plot 3: new global overview figure
# -----------------------------
def plot_global_overview_figure(
    metric: str,
    label1: str,
    label2: str,
    within_a: dict,
    within_b: dict,
    between_ab: dict,
    title: str = None,
):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    ax1, ax2, ax3 = axes

    xA = within_a["pooled_values"]
    bp1 = ax1.boxplot(
        [xA],
        positions=[1],
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
        boxprops=dict(linewidth=1.0),
    )
    style_boxplot(bp1, COLOR1, alpha=0.40)
    ax1.set_xticks([1])
    ax1.set_xticklabels([label1])
    ax1.set_ylabel(metric_label(metric))
    ax1.set_title(f"{label1}: overall")

    if metric == "skewness":
        ax1.axhline(0, linestyle="--", linewidth=1.0, color="gray")

    starA = p_to_star(within_a["p_value"])
    linesA = [
        f"Kruskal p = {within_a['p_value']:.2e} ({starA})" if np.isfinite(within_a["p_value"]) else "Kruskal p = NA",
        f"effect size = {within_a['effect_size']:.2f}" if np.isfinite(within_a["effect_size"]) else "effect size = NA",
        f"positions = {within_a['num_positions']}",
        f"n = {within_a['total_n']}",
    ]
    add_stat_box(ax1, linesA)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    xB = within_b["pooled_values"]
    bp2 = ax2.boxplot(
        [xB],
        positions=[1],
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
        boxprops=dict(linewidth=1.0),
    )
    style_boxplot(bp2, COLOR2, alpha=0.40)
    ax2.set_xticks([1])
    ax2.set_xticklabels([label2])
    ax2.set_ylabel(metric_label(metric))
    ax2.set_title(f"{label2}: overall")

    if metric == "skewness":
        ax2.axhline(0, linestyle="--", linewidth=1.0, color="gray")

    starB = p_to_star(within_b["p_value"])
    linesB = [
        f"Kruskal p = {within_b['p_value']:.2e} ({starB})" if np.isfinite(within_b["p_value"]) else "Kruskal p = NA",
        f"effect size = {within_b['effect_size']:.2f}" if np.isfinite(within_b["effect_size"]) else "effect size = NA",
        f"positions = {within_b['num_positions']}",
        f"n = {within_b['total_n']}",
    ]
    add_stat_box(ax2, linesB)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    x1 = between_ab["values1"]
    x2 = between_ab["values2"]

    bp3 = ax3.boxplot(
        [x1, x2],
        positions=[1, 2],
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
        boxprops=dict(linewidth=1.0),
    )

    style_boxplot(
        {
            "boxes": [bp3["boxes"][0]],
            "whiskers": [bp3["whiskers"][0], bp3["whiskers"][1]],
            "caps": [bp3["caps"][0], bp3["caps"][1]],
            "medians": [bp3["medians"][0]],
        },
        COLOR1,
        alpha=0.40
    )
    style_boxplot(
        {
            "boxes": [bp3["boxes"][1]],
            "whiskers": [bp3["whiskers"][2], bp3["whiskers"][3]],
            "caps": [bp3["caps"][2], bp3["caps"][3]],
            "medians": [bp3["medians"][1]],
        },
        COLOR2,
        alpha=0.40
    )

    ax3.set_xticks([1, 2])
    ax3.set_xticklabels([label1, label2])
    ax3.set_ylabel(metric_label(metric))
    ax3.set_title(f"{label1} vs {label2}: overall")

    if metric == "skewness":
        ax3.axhline(0, linestyle="--", linewidth=1.0, color="gray")

    starAB = p_to_star(between_ab["p_value"])
    linesAB = [
        f"Mann–Whitney p = {between_ab['p_value']:.2e} ({starAB})" if np.isfinite(between_ab["p_value"]) else "Mann–Whitney p = NA",
        f"Cohen's d = {between_ab['effect_size']:.2f}" if np.isfinite(between_ab["effect_size"]) else "Cohen's d = NA",
        f"n = {between_ab[f'{label1}_n']} | {between_ab[f'{label2}_n']}",
    ]
    add_stat_box(ax3, linesAB)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    fig.suptitle(
        title if title else f"{metric_label(metric)} global overview",
        y=1.02,
        fontsize=13
    )
    fig.tight_layout()
    return fig

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Per-position main plot + global summary + global overview into one multi-page PDF."
    )
    ap.add_argument("--input1", required=True, help="CSV for group1")
    ap.add_argument("--input2", required=True, help="CSV for group2")
    ap.add_argument("--output", required=True, help="Output PDF path")
    ap.add_argument(
        "--metric",
        nargs="+",
        default=["mean"],
        help="Metrics to plot, e.g. mean std skewness dwell_time"
    )
    ap.add_argument(
        "--highlight",
        nargs="*",
        type=int,
        default=[],
        help="Highlighted positions, e.g. --highlight 13 32 51"
    )
    ap.add_argument("--label1", default="Group1", help="Legend label for input1")
    ap.add_argument("--label2", default="Group2", help="Legend label for input2")
    ap.add_argument("--title", default=None, help="Optional custom title")
    args = ap.parse_args()

    if not args.output.endswith(".pdf"):
        raise ValueError("--output 必须是 .pdf 文件，例如 results/compare.pdf")

    ensure_parent_dir(args.output)

    valid_metrics = [
        "mean", "std", "skewness", "dwell_time", "dwell_time_ms",
        "median", "range", "q25", "q75"
    ]

    with PdfPages(args.output) as pdf:
        for metric in args.metric:
            if metric not in valid_metrics:
                print(f"[WARN] skip invalid metric: {metric}")
                continue

            print(f"[INFO] plotting metric: {metric}")

            df1 = load_csv(args.input1, metric)
            df2 = load_csv(args.input2, metric)

            between_stats_df = compute_stats_table(
                df1=df1,
                df2=df2,
                metric=metric,
                label1=args.label1,
                label2=args.label2,
                highlight=args.highlight,
            )
            save_stats_table(between_stats_df, args.output, metric)

            within_a = global_within_group_test(df1, metric, args.label1)
            within_b = global_within_group_test(df2, metric, args.label2)
            between_ab = global_between_group_test(df1, df2, metric, args.label1, args.label2)

            save_global_stats_table(
                output_pdf=args.output,
                metric=metric,
                label1=args.label1,
                label2=args.label2,
                within_a=within_a,
                within_b=within_b,
                between_ab=between_ab,
            )

            fig1 = boxplot_two_groups_figure(
                df1=df1,
                df2=df2,
                metric=metric,
                label1=args.label1,
                label2=args.label2,
                highlight=args.highlight,
                title=args.title if args.title else metric_label(metric),
            )
            pdf.savefig(fig1, bbox_inches="tight")
            plt.close(fig1)

            fig2 = global_summary_figure(
                df1=df1,
                df2=df2,
                stats_df=between_stats_df,
                metric=metric,
                label1=args.label1,
                label2=args.label2,
                highlight=args.highlight,
                title=f"{metric_label(metric)}: global summary",
            )
            pdf.savefig(fig2, bbox_inches="tight")
            plt.close(fig2)

            fig3 = plot_global_overview_figure(
                metric=metric,
                label1=args.label1,
                label2=args.label2,
                within_a=within_a,
                within_b=within_b,
                between_ab=between_ab,
                title=f"{metric_label(metric)}: global overview",
            )
            pdf.savefig(fig3, bbox_inches="tight")
            plt.close(fig3)

    print(f"[OK] saved PDF: {args.output}")

if __name__ == "__main__":
    main()