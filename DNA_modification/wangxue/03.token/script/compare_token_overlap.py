#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import ast
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare token overlap between two jsonl files for window sizes 1,3,5,7,9 and export one PDF"
    )
    parser.add_argument("--input1", required=True, help="First jsonl file")
    parser.add_argument("--input2", required=True, help="Second jsonl file")
    parser.add_argument("--label1", default="file1", help="Label for first file")
    parser.add_argument("--label2", default="file2", help="Label for second file")
    parser.add_argument(
        "--output_prefix",
        default="token_overlap_compare",
        help="Output prefix for csv/pdf"
    )
    parser.add_argument(
        "--highlight",
        nargs="+",
        type=int,
        default=[13, 32, 51, 70, 89, 108, 127],
        help="Positions to highlight"
    )
    parser.add_argument(
        "--fig_width",
        type=float,
        default=18,
        help="Figure width"
    )
    parser.add_argument(
        "--fig_height",
        type=float,
        default=4,
        help="Figure height"
    )
    parser.add_argument(
        "--tick_step",
        type=int,
        default=None,
        help="X tick step; default auto"
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        type=int,
        default=[1, 3, 5, 7, 9],
        help="Window sizes to compare, default: 1 3 5 7 9"
    )
    return parser.parse_args()


def load_jsonl(path: str) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)

    if "base_token_ids" not in df.columns:
        raise ValueError(f"{path} missing required column: base_token_ids")

    if len(df) == 0:
        raise ValueError(f"{path} is empty")

    if isinstance(df["base_token_ids"].iloc[0], str):
        df["base_token_ids"] = df["base_token_ids"].apply(ast.literal_eval)

    return df


def validate_base_token_ids(df: pd.DataFrame, file_label: str) -> int:
    lengths = df["base_token_ids"].apply(len).unique()
    if len(lengths) != 1:
        raise ValueError(
            f"{file_label}: inconsistent base_token_ids lengths found: {lengths}"
        )
    seq_len = int(lengths[0])
    return seq_len


def collect_tokens_by_region(df: pd.DataFrame, start: int, end: int) -> set:
    """
    Collect union of tokens across all reads within [start, end).
    For window=1, this is exactly one position.
    For window>1, this merges all positions in the window.
    """
    token_set = set()

    for row in df["base_token_ids"]:
        for pos in range(start, end):
            vals = row[pos]
            if vals is None:
                continue
            if isinstance(vals, list):
                token_set.update(vals)
            else:
                token_set.add(vals)

    return token_set


def compute_overlap_between_files(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    window_size: int = 1
) -> pd.DataFrame:
    seq_len1 = validate_base_token_ids(df1, "input1")
    seq_len2 = validate_base_token_ids(df2, "input2")

    if seq_len1 != seq_len2:
        raise ValueError(f"Sequence length mismatch: {seq_len1} vs {seq_len2}")

    if window_size <= 0:
        raise ValueError("window must be >= 1")

    results = []

    for start in range(0, seq_len1, window_size):
        end = min(start + window_size, seq_len1)

        s1 = collect_tokens_by_region(df1, start, end)
        s2 = collect_tokens_by_region(df2, start, end)

        inter = s1 & s2
        union = s1 | s2

        inter_size = len(inter)
        union_size = len(union)
        jaccard = inter_size / union_size if union_size > 0 else 0.0

        overlap_ratio_file1 = inter_size / len(s1) if len(s1) > 0 else 0.0
        overlap_ratio_file2 = inter_size / len(s2) if len(s2) > 0 else 0.0

        results.append({
            "window_size": window_size,
            "start": start,
            "end": end - 1,
            "n_tokens_file1": len(s1),
            "n_tokens_file2": len(s2),
            "n_intersection": inter_size,
            "n_union": union_size,
            "has_overlap": inter_size > 0,
            "jaccard_overlap": jaccard,
            "overlap_ratio_file1": overlap_ratio_file1,
            "overlap_ratio_file2": overlap_ratio_file2,
            "shared_tokens": ",".join(map(str, sorted(inter))) if inter_size <= 30 else ""
        })

    return pd.DataFrame(results)


def build_colors(res_df: pd.DataFrame, window_size: int, highlight_positions: set):
    colors = []

    if window_size == 1:
        for pos in res_df["start"]:
            if pos in highlight_positions:
                colors.append("tomato")
            else:
                colors.append("steelblue")
    else:
        for s, e in zip(res_df["start"], res_df["end"]):
            if any(p >= s and p <= e for p in highlight_positions):
                colors.append("tomato")
            else:
                colors.append("steelblue")

    return colors


def make_bar_figure(
    res_df: pd.DataFrame,
    window_size: int,
    highlight_positions,
    fig_width: float = 18,
    fig_height: float = 4,
    tick_step: int = None,
    label1: str = "file1",
    label2: str = "file2",
):
    x = np.arange(len(res_df))
    y = res_df["jaccard_overlap"].values

    highlight_positions = set(highlight_positions)
    colors = build_colors(res_df, window_size, highlight_positions)

    if window_size == 1:
        x_labels = res_df["start"].astype(str).tolist()
        xlabel = "Position"
        if tick_step is None:
            tick_step_use = 10
        else:
            tick_step_use = tick_step
    else:
        x_labels = [f"{s}-{e}" for s, e in zip(res_df["start"], res_df["end"])]
        xlabel = f"{window_size}-bp window"
        if tick_step is None:
            tick_step_use = 1
        else:
            tick_step_use = tick_step

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.bar(x, y, color=colors)

    ticks = np.arange(0, len(x), tick_step_use)
    tick_labels = [x_labels[i] for i in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    ax.set_ylim(0, 1.05)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Jaccard overlap")
    ax.set_title(
        f"Token overlap: {label1} vs {label2} (window={window_size})"
    )

    if window_size == 1:
        for i, pos in enumerate(res_df["start"]):
            if pos in highlight_positions:
                ax.axvline(i, linestyle="--", linewidth=0.6, alpha=0.25, color="gray")
    else:
        for i, (s, e) in enumerate(zip(res_df["start"], res_df["end"])):
            if any(p >= s and p <= e for p in highlight_positions):
                ax.axvline(i, linestyle="--", linewidth=0.6, alpha=0.2, color="gray")

    # simple summary box
    top_mean = res_df["jaccard_overlap"].mean()
    top_med = res_df["jaccard_overlap"].median()
    top_max = res_df["jaccard_overlap"].max()
    top_min = res_df["jaccard_overlap"].min()

    txt = (
        f"windows = {len(res_df)}\n"
        f"mean = {top_mean:.3f}\n"
        f"median = {top_med:.3f}\n"
        f"min = {top_min:.3f}\n"
        f"max = {top_max:.3f}"
    )
    ax.text(
        0.985, 0.98, txt,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="lightgray", alpha=0.92)
    )

    plt.tight_layout()
    return fig


def save_top_tables(res_df: pd.DataFrame, window_size: int):
    print(f"\n[Window={window_size}] Top 10 highest-overlap positions/windows")
    print(
        res_df.sort_values("jaccard_overlap", ascending=False)
        .head(10)[["start", "end", "n_intersection", "n_union", "jaccard_overlap"]]
        .to_string(index=False)
    )

    print(f"\n[Window={window_size}] Top 10 lowest-overlap positions/windows")
    print(
        res_df.sort_values("jaccard_overlap", ascending=True)
        .head(10)[["start", "end", "n_intersection", "n_union", "jaccard_overlap"]]
        .to_string(index=False)
    )


def main():
    args = parse_args()

    df1 = load_jsonl(args.input1)
    df2 = load_jsonl(args.input2)

    output_pdf = f"{args.output_prefix}.pdf"
    os.makedirs(os.path.dirname(output_pdf) if os.path.dirname(output_pdf) else ".", exist_ok=True)

    all_results = []

    with PdfPages(output_pdf) as pdf:
        for w in args.windows:
            if w <= 0:
                print(f"[WARN] skip invalid window: {w}")
                continue

            res_df = compute_overlap_between_files(
                df1=df1,
                df2=df2,
                window_size=w
            )

            output_csv = f"{args.output_prefix}.window{w}.csv"
            res_df.to_csv(output_csv, index=False)
            print(f"[INFO] saved csv: {output_csv}")

            fig = make_bar_figure(
                res_df=res_df,
                window_size=w,
                highlight_positions=args.highlight,
                fig_width=args.fig_width,
                fig_height=args.fig_height,
                tick_step=args.tick_step,
                label1=args.label1,
                label2=args.label2,
            )
            pdf.savefig(fig, dpi=300, bbox_inches="tight")
            plt.close(fig)

            save_top_tables(res_df, w)

            all_results.append(res_df)

    if len(all_results) > 0:
        merged_df = pd.concat(all_results, axis=0, ignore_index=True)
        merged_csv = f"{args.output_prefix}.all_windows.csv"
        merged_df.to_csv(merged_csv, index=False)
        print(f"[INFO] saved merged csv: {merged_csv}")

    print(f"[INFO] saved pdf: {output_pdf}")


if __name__ == "__main__":
    main()