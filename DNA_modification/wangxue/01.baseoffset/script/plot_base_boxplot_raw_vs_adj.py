#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot per-position signal-mean boxplots for fixed-length reads.

This script visualizes signal mean distribution at each base position.

For each read and each base position i:
    span = spans[i]
    signal_mean_i = mean(signal[span_start:span_end])

Then values are grouped by base position:
    position 0 -> all reads' signal mean at position 0
    position 1 -> all reads' signal mean at position 1
    ...

It compares two span fields:
    1. base_sample_spans_rel
    2. base_sample_spans_rel_adj

Supported input:
    - .jsonl
    - .jsonl.gz

Outputs:
    - boxplot_by_position_base_sample_spans_rel.png
    - boxplot_by_position_base_sample_spans_rel_adj.png
    - boxplot_by_position_compare_raw_vs_adj.png
    - per_position_signal_mean_stats.tsv

Example:
python plot_position_boxplot_raw_vs_adj.py \
  --data-jsonl signal_none.adjusted.jsonl.gz \
  --out-dir position_boxplot \
  --limit 1000
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, TextIO

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
(Path(os.environ["XDG_CACHE_HOME"]) / "fontconfig").mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DATA_JSONL = Path("signal_none.adjusted.jsonl")
DEFAULT_OUT_DIR = Path("position_boxplot")
DEFAULT_LIMIT = 1000

DEFAULT_PATTERN_FIELD = "pattern"
DEFAULT_SIGNAL_FIELD = "signal"
DEFAULT_READ_ID_FIELD = "read_id"

DEFAULT_SPAN_FIELDS = [
    "base_sample_spans_rel",
    "base_sample_spans_rel_adj",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-position signal-mean boxplots for fixed-length JSONL reads."
        )
    )

    parser.add_argument(
        "--data-jsonl",
        type=Path,
        default=DEFAULT_DATA_JSONL,
        help="Input JSONL or JSONL.GZ file.",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Maximum records to load. Use 0 for no limit.",
    )

    parser.add_argument(
        "--pattern-field",
        default=DEFAULT_PATTERN_FIELD,
        help="Sequence field name.",
    )

    parser.add_argument(
        "--signal-field",
        default=DEFAULT_SIGNAL_FIELD,
        help="Signal field name.",
    )

    parser.add_argument(
        "--read-id-field",
        default=DEFAULT_READ_ID_FIELD,
        help="Read ID field name.",
    )

    parser.add_argument(
        "--span-fields",
        nargs="+",
        default=DEFAULT_SPAN_FIELDS,
        help=(
            "Span fields to compare. Default: "
            "base_sample_spans_rel base_sample_spans_rel_adj"
        ),
    )

    parser.add_argument(
        "--min-segment-len",
        type=int,
        default=1,
        help="Minimum signal samples required for each base interval.",
    )

    parser.add_argument(
        "--showfliers",
        action="store_true",
        help="Show outliers in boxplot. Default: hide outliers.",
    )

    parser.add_argument(
        "--max-positions",
        type=int,
        default=0,
        help=(
            "Maximum number of base positions to plot. "
            "Use 0 to plot all positions."
        ),
    )

    parser.add_argument(
        "--position-start",
        type=int,
        default=0,
        help="Start base position for plotting, 0-based inclusive.",
    )

    parser.add_argument(
        "--position-end",
        type=int,
        default=0,
        help=(
            "End base position for plotting, 0-based exclusive. "
            "Use 0 to plot until sequence end."
        ),
    )

    parser.add_argument(
        "--fig-width",
        type=float,
        default=18.0,
        help="Figure width.",
    )

    parser.add_argument(
        "--fig-height",
        type=float,
        default=6.0,
        help="Figure height.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output figure DPI.",
    )

    return parser.parse_args()


def open_text_auto(path: Path) -> TextIO:
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def sanitize_name(text: str) -> str:
    keep = []
    for ch in text:
        if ch.isalnum() or ch in ("_", "-", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def safe_float_array(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def validate_spans(spans: Any, signal_len: int) -> bool:
    if not isinstance(spans, list) or len(spans) == 0:
        return False

    if signal_len <= 0:
        return False

    prev_start = -math.inf

    for item in spans:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            return False

        try:
            start = float(item[0])
            end = float(item[1])
        except Exception:
            return False

        if not math.isfinite(start) or not math.isfinite(end):
            return False

        if start >= end:
            return False

        if start < prev_start:
            return False

        prev_start = start

    return True


def clip_interval(start: int, end: int, signal_len: int) -> tuple[int, int]:
    start = max(0, min(start, signal_len))
    end = max(0, min(end, signal_len))
    return start, end


def summarize_array(arr: np.ndarray) -> dict[str, Any]:
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "p05": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "p95": float("nan"),
        }

    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
    }


def init_position_values(length: int) -> list[list[float]]:
    return [[] for _ in range(length)]


def ensure_length(values: list[list[float]], length: int) -> list[list[float]]:
    if len(values) < length:
        values.extend([[] for _ in range(length - len(values))])
    return values


def collect_position_values_for_record(
    record: dict[str, Any],
    span_field: str,
    pattern_field: str,
    signal_field: str,
    min_segment_len: int,
) -> list[list[float]]:
    seq = str(record[pattern_field]).upper()
    signal = safe_float_array(record[signal_field])
    spans = record[span_field]

    n = min(len(seq), len(spans))

    values_by_position = init_position_values(n)

    if signal.size == 0:
        return values_by_position

    if not validate_spans(spans, signal.size):
        return values_by_position

    for pos in range(n):
        try:
            start = int(math.floor(float(spans[pos][0])))
            end = int(math.ceil(float(spans[pos][1])))
        except Exception:
            continue

        start, end = clip_interval(start, end, signal.size)

        if end - start < min_segment_len:
            continue

        segment = signal[start:end]

        if segment.size < min_segment_len:
            continue

        mean_value = float(np.mean(segment))

        if math.isfinite(mean_value):
            values_by_position[pos].append(mean_value)

    return values_by_position


def collect_all_position_values(
    path: Path,
    limit: int,
    span_fields: list[str],
    pattern_field: str,
    signal_field: str,
    read_id_field: str,
    min_segment_len: int,
) -> tuple[dict[str, list[list[float]]], dict[str, Any]]:
    values_by_field: dict[str, list[list[float]]] = {
        span_field: []
        for span_field in span_fields
    }

    stats = {
        "records_seen": 0,
        "records_loaded": 0,
        "records_processed_any": 0,
        "records_skipped_missing_required": 0,
        "records_skipped_no_value": 0,
        "records_failed_exception": 0,
        "observed_lengths": {},
    }

    required_fields = set([pattern_field, signal_field])
    required_fields.update(span_fields)

    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    with open_text_auto(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if limit > 0 and stats["records_loaded"] >= limit:
                break

            line = line.strip()
            if not line:
                continue

            stats["records_seen"] += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"[WARN] invalid JSON at line {line_number}: {exc}",
                    file=sys.stderr,
                )
                continue

            missing = [field for field in required_fields if field not in record]

            if missing:
                stats["records_skipped_missing_required"] += 1
                continue

            seq_len = len(str(record[pattern_field]))
            stats["observed_lengths"][str(seq_len)] = (
                stats["observed_lengths"].get(str(seq_len), 0) + 1
            )

            stats["records_loaded"] += 1

            try:
                record_has_value = False

                for span_field in span_fields:
                    per_record_values = collect_position_values_for_record(
                        record=record,
                        span_field=span_field,
                        pattern_field=pattern_field,
                        signal_field=signal_field,
                        min_segment_len=min_segment_len,
                    )

                    values_by_field[span_field] = ensure_length(
                        values_by_field[span_field],
                        len(per_record_values),
                    )

                    for pos, vals in enumerate(per_record_values):
                        if vals:
                            record_has_value = True
                            values_by_field[span_field][pos].extend(vals)

                if record_has_value:
                    stats["records_processed_any"] += 1
                else:
                    stats["records_skipped_no_value"] += 1

            except Exception as exc:
                stats["records_failed_exception"] += 1
                read_id = record.get(read_id_field, "?")
                print(
                    f"[WARN] failed line={line_number}, read_id={read_id}: {exc}",
                    file=sys.stderr,
                )

            if stats["records_loaded"] % 200 == 0:
                print(
                    f"[INFO] loaded={stats['records_loaded']}, "
                    f"processed={stats['records_processed_any']}",
                    file=sys.stderr,
                )

    return values_by_field, stats


def slice_positions(
    values_by_position: list[list[float]],
    position_start: int,
    position_end: int,
    max_positions: int,
) -> tuple[list[list[float]], list[int]]:
    n = len(values_by_position)

    start = max(0, position_start)

    if position_end > 0:
        end = min(position_end, n)
    else:
        end = n

    if max_positions > 0:
        end = min(end, start + max_positions)

    sliced_values = values_by_position[start:end]
    positions = list(range(start, end))

    return sliced_values, positions


def plot_one_field_position_boxplot(
    values_by_position: list[list[float]],
    span_field: str,
    output_path: Path,
    position_start: int,
    position_end: int,
    max_positions: int,
    showfliers: bool,
    fig_width: float,
    fig_height: float,
    dpi: int,
) -> None:
    data, positions = slice_positions(
        values_by_position=values_by_position,
        position_start=position_start,
        position_end=position_end,
        max_positions=max_positions,
    )

    if not data:
        raise ValueError(f"No position values to plot for {span_field}")

    clean_data = []

    for vals in data:
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        clean_data.append(arr)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

    ax.boxplot(
        clean_data,
        positions=positions,
        widths=0.6,
        showfliers=showfliers,
        patch_artist=True,
        medianprops={"linewidth": 1.5},
        boxprops={"linewidth": 1.0, "facecolor": "#4c78a8", "alpha": 0.35},
        whiskerprops={"linewidth": 0.9},
        capprops={"linewidth": 0.9},
    )

    ax.set_title(f"Per-position signal mean boxplot | {span_field}")
    ax.set_xlabel("Base position")
    ax.set_ylabel("Mean signal per base interval")
    ax.grid(axis="y", alpha=0.25)

    if len(positions) > 60:
        step = max(1, len(positions) // 20)
        xticks = positions[::step]
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(x) for x in xticks], rotation=45, ha="right")
    else:
        ax.set_xticks(positions)
        ax.set_xticklabels([str(x) for x in positions], rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    print(f"[INFO] saved: {output_path}")


def plot_compare_position_boxplot(
    values_by_field: dict[str, list[list[float]]],
    span_fields: list[str],
    output_path: Path,
    position_start: int,
    position_end: int,
    max_positions: int,
    showfliers: bool,
    fig_width: float,
    fig_height: float,
    dpi: int,
) -> None:
    if len(span_fields) != 2:
        print("[WARN] compare plot requires exactly 2 span fields; skipped.")
        return

    field_raw, field_adj = span_fields

    raw_data, positions = slice_positions(
        values_by_position=values_by_field[field_raw],
        position_start=position_start,
        position_end=position_end,
        max_positions=max_positions,
    )

    adj_data, positions_adj = slice_positions(
        values_by_position=values_by_field[field_adj],
        position_start=position_start,
        position_end=position_end,
        max_positions=max_positions,
    )

    n = min(len(raw_data), len(adj_data), len(positions), len(positions_adj))
    raw_data = raw_data[:n]
    adj_data = adj_data[:n]
    positions = positions[:n]

    data = []
    plot_positions = []

    shift = 0.18

    for pos, raw_vals, adj_vals in zip(positions, raw_data, adj_data):
        raw_arr = np.asarray(raw_vals, dtype=np.float64)
        adj_arr = np.asarray(adj_vals, dtype=np.float64)

        raw_arr = raw_arr[np.isfinite(raw_arr)]
        adj_arr = adj_arr[np.isfinite(adj_arr)]

        data.extend([raw_arr, adj_arr])
        plot_positions.extend([pos - shift, pos + shift])

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

    box = ax.boxplot(
        data,
        positions=plot_positions,
        widths=0.28,
        showfliers=showfliers,
        patch_artist=True,
        medianprops={"linewidth": 1.4},
        boxprops={"linewidth": 0.9},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
    )

    for idx, patch in enumerate(box["boxes"]):
        if idx % 2 == 0:
            patch.set_facecolor("#999999")
            patch.set_alpha(0.35)
        else:
            patch.set_facecolor("#4c78a8")
            patch.set_alpha(0.35)

    ax.set_title("Per-position signal mean boxplot | raw span vs adjusted span")
    ax.set_xlabel("Base position")
    ax.set_ylabel("Mean signal per base interval")
    ax.grid(axis="y", alpha=0.25)

    if len(positions) > 60:
        step = max(1, len(positions) // 20)
        xticks = positions[::step]
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(x) for x in xticks], rotation=45, ha="right")
    else:
        ax.set_xticks(positions)
        ax.set_xticklabels([str(x) for x in positions], rotation=45, ha="right")

    legend_handles = [
        plt.Line2D([0], [0], color="#999999", lw=8, alpha=0.5, label=field_raw),
        plt.Line2D([0], [0], color="#4c78a8", lw=8, alpha=0.5, label=field_adj),
    ]

    ax.legend(handles=legend_handles, frameon=False)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    print(f"[INFO] saved: {output_path}")


def write_position_stats_tsv(
    values_by_field: dict[str, list[list[float]]],
    output_path: Path,
) -> None:
    fieldnames = [
        "span_field",
        "position",
        "count",
        "mean",
        "std",
        "median",
        "min",
        "max",
        "p05",
        "p25",
        "p75",
        "p95",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for span_field, values_by_position in values_by_field.items():
            for pos, vals in enumerate(values_by_position):
                arr = np.asarray(vals, dtype=np.float64)
                row = {
                    "span_field": span_field,
                    "position": pos,
                    **summarize_array(arr),
                }
                writer.writerow(row)

    print(f"[INFO] saved: {output_path}")


def main() -> int:
    args = parse_args()

    if args.limit < 0:
        raise ValueError("--limit must be >= 0. Use 0 for no limit.")

    if args.min_segment_len <= 0:
        raise ValueError("--min-segment-len must be positive.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] ===== Config =====")
    print(f"[INFO] data_jsonl: {args.data_jsonl}")
    print(f"[INFO] out_dir: {args.out_dir}")
    print(f"[INFO] limit: {args.limit}")
    print(f"[INFO] span_fields: {args.span_fields}")
    print(f"[INFO] pattern_field: {args.pattern_field}")
    print(f"[INFO] signal_field: {args.signal_field}")
    print(f"[INFO] position_start: {args.position_start}")
    print(f"[INFO] position_end: {args.position_end}")
    print(f"[INFO] max_positions: {args.max_positions}")

    values_by_field, stats = collect_all_position_values(
        path=args.data_jsonl,
        limit=args.limit,
        span_fields=args.span_fields,
        pattern_field=args.pattern_field,
        signal_field=args.signal_field,
        read_id_field=args.read_id_field,
        min_segment_len=args.min_segment_len,
    )

    print("\n[INFO] ===== Run stats =====")
    for key, value in stats.items():
        print(f"[INFO] {key}: {value}")

    print("\n[INFO] ===== Position counts =====")
    for span_field in args.span_fields:
        n_pos = len(values_by_field[span_field])
        print(f"[INFO] {span_field}: {n_pos} positions")

        if n_pos > 0:
            counts = [len(x) for x in values_by_field[span_field]]
            print(
                f"[INFO] {span_field}: "
                f"min_count={min(counts)}, "
                f"max_count={max(counts)}, "
                f"mean_count={np.mean(counts):.2f}"
            )

    for span_field in args.span_fields:
        output_path = args.out_dir / f"boxplot_by_position_{sanitize_name(span_field)}.png"

        plot_one_field_position_boxplot(
            values_by_position=values_by_field[span_field],
            span_field=span_field,
            output_path=output_path,
            position_start=args.position_start,
            position_end=args.position_end,
            max_positions=args.max_positions,
            showfliers=args.showfliers,
            fig_width=args.fig_width,
            fig_height=args.fig_height,
            dpi=args.dpi,
        )

    if len(args.span_fields) == 2:
        compare_path = args.out_dir / "boxplot_by_position_compare_raw_vs_adj.png"

        plot_compare_position_boxplot(
            values_by_field=values_by_field,
            span_fields=args.span_fields,
            output_path=compare_path,
            position_start=args.position_start,
            position_end=args.position_end,
            max_positions=args.max_positions,
            showfliers=args.showfliers,
            fig_width=args.fig_width,
            fig_height=args.fig_height,
            dpi=args.dpi,
        )

    stats_path = args.out_dir / "per_position_signal_mean_stats.tsv"

    write_position_stats_tsv(
        values_by_field=values_by_field,
        output_path=stats_path,
    )

    print("\n[INFO] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())