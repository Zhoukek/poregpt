#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate ATGC signal-mean distributions under different span fields and offsets.

This script is a generalized and production-style version of the previous
LB07 unadjusted / left3.7 / left4 scripts.

Main features
-------------
1. Supports arbitrary span field:
   - base_sample_spans_rel
   - base_sample_spans_rel_adj
   - any other compatible span field

2. Supports arbitrary offsets:
   - 0
   - -3.5
   - -3.7
   - -4.0
   - etc.

3. Supports fractional offsets by interpolating span boundaries.

4. Supports plain JSONL and gzipped JSONL:
   - xxx.jsonl
   - xxx.jsonl.gz
   - xxx.gz

5. Outputs:
   - distribution plots for each offset
   - summary JSON
   - pairwise separation TSV
   - per-base stats TSV

Offset convention
-----------------
For sequence base i:

    shifted_start = i - offset
    shifted_end   = shifted_start + 1

Therefore:

    offset = 0      -> base[i] uses span[i]
    offset = -4.0   -> base[i] uses span[i + 4]
    offset = -3.7   -> base[i] uses interpolated interval [i + 3.7, i + 4.7]

In this convention, negative offsets mean assigning downstream/right-side
signal intervals to the current base, which is often described as "left shift"
of the base assignment.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, TextIO

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
(Path(os.environ["XDG_CACHE_HOME"]) / "fontconfig").mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# -----------------------------
# Default config
# -----------------------------

DEFAULT_DATA_JSONL = Path("signal_none.adjusted.jsonl")
DEFAULT_OUT_DIR = Path("plot_adj")
DEFAULT_LIMIT = 1000
DEFAULT_SPAN_FIELD = "base_sample_spans_rel"
DEFAULT_OFFSETS = [0.0, -3.5, -3.7, -4.0]

BASES = ("A", "T", "G", "C")

COLORS = {
    "A": "#d62728",
    "T": "#1f77b4",
    "G": "#2ca02c",
    "C": "#9467bd",
}


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class RunStats:
    records_loaded: int = 0
    reads_processed: int = 0
    reads_skipped_missing_field: int = 0
    reads_skipped_invalid_span: int = 0
    reads_skipped_empty_signal: int = 0
    reads_skipped_no_value: int = 0
    reads_failed_exception: int = 0


@dataclass
class Config:
    data_jsonl: str
    out_dir: str
    limit: int
    span_field: str
    signal_field: str
    pattern_field: str
    read_id_field: str
    offsets: list[float]
    prefix: str
    min_segment_len: int
    plot_bins: int


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ATGC signal-mean distributions using a selected span field "
            "and multiple offsets. Supports .jsonl and .jsonl.gz."
        )
    )

    parser.add_argument(
        "--data-jsonl",
        type=Path,
        default=DEFAULT_DATA_JSONL,
        help="Input JSONL file. Supports .jsonl and .jsonl.gz.",
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
        help="Maximum number of valid records to load. Use 0 for no limit.",
    )

    parser.add_argument(
        "--span-field",
        default=DEFAULT_SPAN_FIELD,
        help=(
            "Span field to use, for example: base_sample_spans_rel or "
            "base_sample_spans_rel_adj."
        ),
    )

    parser.add_argument(
        "--signal-field",
        default="signal",
        help="Signal field name in JSONL.",
    )

    parser.add_argument(
        "--pattern-field",
        default="pattern",
        help="Sequence/pattern field name in JSONL.",
    )

    parser.add_argument(
        "--read-id-field",
        default="read_id",
        help="Read id field name in JSONL.",
    )

    parser.add_argument(
        "--offsets",
        type=float,
        nargs="+",
        default=DEFAULT_OFFSETS,
        help="Offsets to evaluate, e.g. --offsets 0 -3.5 -3.7 -4.0",
    )

    parser.add_argument(
        "--prefix",
        default=None,
        help=(
            "Output prefix. If not provided, one will be generated from span field "
            "and offsets."
        ),
    )

    parser.add_argument(
        "--min-segment-len",
        type=int,
        default=1,
        help="Minimum number of signal samples required for one extracted interval.",
    )

    parser.add_argument(
        "--plot-bins",
        type=int,
        default=90,
        help="Histogram bins for distribution plots.",
    )

    return parser.parse_args()


# -----------------------------
# Utility functions
# -----------------------------

def open_text_auto(path: Path) -> TextIO:
    """
    Open plain text JSONL or gzipped JSONL transparently.

    Supported:
        xxx.jsonl
        xxx.jsonl.gz
        xxx.gz
    """
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def sanitize_name(text: str) -> str:
    keep = []
    for char in text:
        if char.isalnum() or char in ("_", "-", "."):
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep)


def offset_label(offset: float) -> str:
    if abs(offset) < 1e-12:
        return "offset0"

    abs_value = abs(offset)

    if float(abs_value).is_integer():
        value = str(int(abs_value))
    else:
        value = str(abs_value).replace(".", "p")

    if offset < 0:
        return f"left{value}"
    return f"right{value}"


def normal_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    if std <= 0 or not np.isfinite(std):
        return np.full_like(x, np.nan, dtype=np.float64)

    z = (x - mean) / std
    return np.exp(-0.5 * z * z) / (std * math.sqrt(2.0 * math.pi))


def safe_float_array(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def validate_spans(spans: Any, signal_len: int) -> bool:
    """
    Validate span structure.

    Expected:
        spans = [[start, end], [start, end], ...]

    Requirements:
        - each span has two numeric values
        - start < end
        - starts are non-decreasing
        - signal length > 0
    """
    if not isinstance(spans, list) or len(spans) == 0:
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

    if signal_len <= 0:
        return False

    return True


# -----------------------------
# Loading
# -----------------------------

def load_records(
    path: Path,
    limit: int,
    pattern_field: str,
    span_field: str,
    signal_field: str,
    read_id_field: str,
) -> tuple[list[dict[str, Any]], RunStats]:
    records: list[dict[str, Any]] = []
    stats = RunStats()

    if not path.exists():
        raise FileNotFoundError(f"Input JSONL does not exist: {path}")

    with open_text_auto(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if limit > 0 and len(records) >= limit:
                break

            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"[WARN] skip invalid JSON at line {line_number}: {exc}",
                    file=sys.stderr,
                )
                continue

            read_id = obj.get(read_id_field)
            pattern = obj.get(pattern_field)
            spans = obj.get(span_field)
            signal = obj.get(signal_field)

            if read_id is None or pattern is None or spans is None or signal is None:
                stats.reads_skipped_missing_field += 1
                continue

            records.append(obj)

    stats.records_loaded = len(records)
    return records, stats


# -----------------------------
# Boundary interpolation
# -----------------------------

def build_boundaries(spans: list[list[int]], n: int) -> np.ndarray:
    """
    Build boundary array.

    boundary[i] = start of span[i]
    boundary[n] = end of span[n - 1]

    Length = n + 1
    """
    boundaries = np.empty(n + 1, dtype=np.float64)

    for i in range(n):
        boundaries[i] = float(spans[i][0])

    boundaries[n] = float(spans[n - 1][1])
    return boundaries


def boundary_at(boundaries: np.ndarray, position: float) -> float:
    """
    Interpolate boundary for fractional base-index position.

    Example:
        position = 3.7 means 70% between boundary[3] and boundary[4].
    """
    max_pos = boundaries.size - 1

    if abs(position - max_pos) < 1e-9:
        return float(boundaries[-1])

    left = int(math.floor(position))

    if left < 0 or left + 1 >= boundaries.size:
        raise IndexError(f"position {position} out of [0, {max_pos}]")

    fraction = position - left

    return float((1.0 - fraction) * boundaries[left] + fraction * boundaries[left + 1])


# -----------------------------
# Core extraction
# -----------------------------

def collect_interval_means_for_one_read(
    seq: str,
    spans: list[list[int]],
    signal: np.ndarray,
    offset: float,
    min_segment_len: int = 1,
) -> dict[str, list[float]]:
    """
    Extract per-base signal means for one read.

    For base i:
        shifted_start = i - offset
        shifted_end   = shifted_start + 1

    Then convert shifted boundaries to signal sample coordinates.
    """
    seq = str(seq).upper()
    values: dict[str, list[float]] = {base: [] for base in BASES}

    n = min(len(seq), len(spans))
    signal_len = signal.size

    if n <= 0 or signal_len <= 0:
        return values

    boundaries = build_boundaries(spans, n)

    for seq_index in range(n):
        base = seq[seq_index]

        if base not in values:
            continue

        shifted_start = float(seq_index) - offset
        shifted_end = shifted_start + 1.0

        if shifted_start < 0 or shifted_end > n:
            continue

        try:
            sample_start = int(math.floor(boundary_at(boundaries, shifted_start)))
            sample_end = int(math.ceil(boundary_at(boundaries, shifted_end)))
        except IndexError:
            continue

        sample_start = max(0, min(sample_start, signal_len))
        sample_end = max(0, min(sample_end, signal_len))

        if sample_end - sample_start < min_segment_len:
            continue

        segment = signal[sample_start:sample_end]

        if segment.size < min_segment_len:
            continue

        mean_value = float(np.mean(segment))

        if math.isfinite(mean_value):
            values[base].append(mean_value)

    return values


def collect_all_values(
    records: list[dict[str, Any]],
    stats: RunStats,
    pattern_field: str,
    span_field: str,
    signal_field: str,
    read_id_field: str,
    offsets: Iterable[float],
    min_segment_len: int,
) -> tuple[dict[float, dict[str, list[float]]], RunStats]:
    offsets = list(offsets)

    values_by_offset: dict[float, dict[str, list[float]]] = {
        offset: {base: [] for base in BASES}
        for offset in offsets
    }

    for index, rec in enumerate(records, start=1):
        read_id = rec.get(read_id_field, "?")

        try:
            seq = rec[pattern_field]
            spans = rec[span_field]
            signal = safe_float_array(rec[signal_field])

            if signal.size == 0:
                stats.reads_skipped_empty_signal += 1
                continue

            if not validate_spans(spans, signal.size):
                stats.reads_skipped_invalid_span += 1
                continue

            any_value = False

            for offset in offsets:
                interval_means = collect_interval_means_for_one_read(
                    seq=seq,
                    spans=spans,
                    signal=signal,
                    offset=offset,
                    min_segment_len=min_segment_len,
                )

                if any(len(interval_means[base]) > 0 for base in BASES):
                    any_value = True

                for base in BASES:
                    values_by_offset[offset][base].extend(interval_means[base])

            if any_value:
                stats.reads_processed += 1
            else:
                stats.reads_skipped_no_value += 1

        except Exception as exc:
            stats.reads_failed_exception += 1
            print(f"[WARN] failed read {read_id}: {exc}", file=sys.stderr)

        if index % 200 == 0:
            print(
                f"[INFO] scanned {index}/{len(records)} records, "
                f"processed={stats.reads_processed}, "
                f"invalid_span={stats.reads_skipped_invalid_span}, "
                f"no_value={stats.reads_skipped_no_value}",
                file=sys.stderr,
            )

    return values_by_offset, stats


# -----------------------------
# Statistics
# -----------------------------

def pooled_std(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")

    value = (
        ((a.size - 1) * np.var(a, ddof=1) + (b.size - 1) * np.var(b, ddof=1))
        / (a.size + b.size - 2)
    )

    return math.sqrt(max(float(value), 0.0))


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    sd = pooled_std(a, b)

    if not math.isfinite(sd) or sd <= 0:
        return float("nan")

    return abs(float(np.mean(a)) - float(np.mean(b))) / sd


def bhattacharyya_distance_normal(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")

    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))

    if var_a <= 0 or var_b <= 0:
        return float("nan")

    mean_var = 0.5 * (var_a + var_b)

    return (
        0.125 * ((mean_a - mean_b) ** 2 / mean_var)
        + 0.5 * math.log(mean_var / math.sqrt(var_a * var_b))
    )


def base_stats(arr: np.ndarray) -> dict[str, Any]:
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "p01": float("nan"),
            "p05": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
        }

    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p01": float(np.percentile(arr, 1)),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def compare_values(
    values_by_base: dict[str, list[float]]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    arrays = {
        base: np.asarray(values_by_base[base], dtype=np.float64)
        for base in BASES
    }

    pairwise_rows: list[dict[str, Any]] = []
    base_rows: list[dict[str, Any]] = []

    d_values: list[float] = []
    bd_values: list[float] = []

    for base in BASES:
        row = {"base": base, **base_stats(arrays[base])}
        base_rows.append(row)

    for base_a, base_b in combinations(BASES, 2):
        arr_a = arrays[base_a]
        arr_b = arrays[base_b]

        d_value = cohens_d(arr_a, arr_b)
        bd_value = bhattacharyya_distance_normal(arr_a, arr_b)

        mean_a = float(np.mean(arr_a)) if arr_a.size else float("nan")
        mean_b = float(np.mean(arr_b)) if arr_b.size else float("nan")

        mean_delta = (
            abs(mean_a - mean_b)
            if math.isfinite(mean_a) and math.isfinite(mean_b)
            else float("nan")
        )

        if math.isfinite(d_value):
            d_values.append(d_value)

        if math.isfinite(bd_value):
            bd_values.append(bd_value)

        pairwise_rows.append(
            {
                "base_pair": f"{base_a}-{base_b}",
                "base_a": base_a,
                "base_b": base_b,
                "n_a": int(arr_a.size),
                "n_b": int(arr_b.size),
                "mean_a": mean_a,
                "mean_b": mean_b,
                "mean_delta": mean_delta,
                "cohens_d": d_value,
                "bhattacharyya_distance": bd_value,
            }
        )

    summary = {
        "bases": {
            row["base"]: {k: v for k, v in row.items() if k != "base"}
            for row in base_rows
        },
        "primary_metric": "min_pairwise_cohens_d",
        "min_pairwise_cohens_d": float(np.min(d_values)) if d_values else float("nan"),
        "mean_pairwise_cohens_d": float(np.mean(d_values)) if d_values else float("nan"),
        "min_pairwise_bhattacharyya_distance": float(np.min(bd_values)) if bd_values else float("nan"),
        "mean_pairwise_bhattacharyya_distance": float(np.mean(bd_values)) if bd_values else float("nan"),
    }

    return summary, pairwise_rows, base_rows


# -----------------------------
# Plotting
# -----------------------------

def save_distribution_plot(
    values_by_base: dict[str, list[float]],
    offset: float,
    span_field: str,
    output_path: Path,
    title_suffix: str = "",
    bins: int = 90,
) -> None:
    arrays = []

    for base in BASES:
        arr = np.asarray(values_by_base[base], dtype=np.float64)
        arr = arr[np.isfinite(arr)]

        if arr.size > 1:
            arrays.append(arr)

    if not arrays:
        raise ValueError("No values available for distribution plot")

    all_values = np.concatenate(arrays)

    x_min, x_max = np.percentile(all_values, [0.5, 99.5])

    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min >= x_max:
        x_min = float(np.min(all_values))
        x_max = float(np.max(all_values))

    pad = 0.05 * (x_max - x_min) if x_max > x_min else 1.0
    x_grid = np.linspace(x_min - pad, x_max + pad, 900)

    fig, ax = plt.subplots(figsize=(11, 7), dpi=180)

    for base in BASES:
        arr = np.asarray(values_by_base[base], dtype=np.float64)
        arr = arr[np.isfinite(arr)]

        if arr.size < 2:
            continue

        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1))

        if std <= 0 or not np.isfinite(std):
            continue

        ax.hist(
            arr,
            bins=bins,
            density=True,
            histtype="stepfilled",
            alpha=0.08,
            color=COLORS.get(base, None),
            linewidth=0,
        )

        ax.plot(
            x_grid,
            normal_pdf(x_grid, mean, std),
            color=COLORS.get(base, None),
            linewidth=2.2,
            label=f"{base}: n={arr.size}, mean={mean:.3f}, std={std:.3f}",
        )

    title = f"ATGC signal mean distributions | {span_field} | {offset_label(offset)}"

    if title_suffix:
        title = f"{title} — {title_suffix}"

    ax.set_title(title)
    ax.set_xlabel("Mean signal per extracted base interval")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    print(f"[INFO] plot saved: {output_path}")


# -----------------------------
# Output
# -----------------------------

def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def build_default_prefix(span_field: str, offsets: list[float]) -> str:
    offset_part = "_".join(offset_label(x) for x in offsets)
    return f"ATGC_signal_mean_{sanitize_name(span_field)}_{offset_part}"


# -----------------------------
# Main
# -----------------------------

def main() -> int:
    args = parse_args()

    if args.limit < 0:
        raise ValueError("--limit must be >= 0. Use 0 for no limit.")

    if args.min_segment_len <= 0:
        raise ValueError("--min-segment-len must be positive.")

    if not args.offsets:
        raise ValueError("--offsets cannot be empty.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    offsets = [float(x) for x in args.offsets]

    prefix = args.prefix
    if prefix is None:
        prefix = build_default_prefix(args.span_field, offsets)

    config = Config(
        data_jsonl=str(args.data_jsonl),
        out_dir=str(args.out_dir),
        limit=int(args.limit),
        span_field=args.span_field,
        signal_field=args.signal_field,
        pattern_field=args.pattern_field,
        read_id_field=args.read_id_field,
        offsets=offsets,
        prefix=prefix,
        min_segment_len=int(args.min_segment_len),
        plot_bins=int(args.plot_bins),
    )

    print("[INFO] ===== Config =====")
    for key, value in asdict(config).items():
        print(f"[INFO] {key}: {value}")

    records, stats = load_records(
        path=args.data_jsonl,
        limit=args.limit,
        pattern_field=args.pattern_field,
        span_field=args.span_field,
        signal_field=args.signal_field,
        read_id_field=args.read_id_field,
    )

    if not records:
        raise ValueError(
            f"No valid records loaded from {args.data_jsonl}. "
            f"Please check span field: {args.span_field}"
        )

    values_by_offset, stats = collect_all_values(
        records=records,
        stats=stats,
        pattern_field=args.pattern_field,
        span_field=args.span_field,
        signal_field=args.signal_field,
        read_id_field=args.read_id_field,
        offsets=offsets,
        min_segment_len=args.min_segment_len,
    )

    summary: dict[str, Any] = {
        "config": asdict(config),
        "run_stats": asdict(stats),
        "ranking_metric": "min_pairwise_cohens_d",
        "offsets": {},
    }

    pairwise_rows_all: list[dict[str, Any]] = []
    base_stats_rows_all: list[dict[str, Any]] = []

    for offset in offsets:
        label = offset_label(offset)

        total_intervals = sum(
            len(values_by_offset[offset][base])
            for base in BASES
        )

        plot_path = args.out_dir / f"{prefix}_{label}_distribution.png"

        save_distribution_plot(
            values_by_base=values_by_offset[offset],
            offset=offset,
            span_field=args.span_field,
            output_path=plot_path,
            title_suffix=f"{total_intervals} intervals from {stats.reads_processed} reads",
            bins=args.plot_bins,
        )

        offset_summary, pairwise_rows, base_rows = compare_values(
            values_by_offset[offset]
        )

        summary["offsets"][label] = {
            "offset": float(offset),
            "plot_path": str(plot_path),
            "total_intervals": int(total_intervals),
            **offset_summary,
        }

        for row in pairwise_rows:
            pairwise_rows_all.append(
                {
                    "offset_label": label,
                    "offset": float(offset),
                    **row,
                }
            )

        for row in base_rows:
            base_stats_rows_all.append(
                {
                    "offset_label": label,
                    "offset": float(offset),
                    **row,
                }
            )

    ranked = sorted(
        summary["offsets"].items(),
        key=lambda item: (
            item[1]["min_pairwise_cohens_d"]
            if math.isfinite(item[1]["min_pairwise_cohens_d"])
            else -math.inf
        ),
        reverse=True,
    )

    summary["ranking"] = [
        {
            "rank": rank,
            "offset_label": label,
            "offset": data["offset"],
            "total_intervals": data["total_intervals"],
            "min_pairwise_cohens_d": data["min_pairwise_cohens_d"],
            "mean_pairwise_cohens_d": data["mean_pairwise_cohens_d"],
            "min_pairwise_bhattacharyya_distance": data[
                "min_pairwise_bhattacharyya_distance"
            ],
            "mean_pairwise_bhattacharyya_distance": data[
                "mean_pairwise_bhattacharyya_distance"
            ],
            "plot_path": data["plot_path"],
        }
        for rank, (label, data) in enumerate(ranked, start=1)
    ]

    summary["best_offset_by_min_pairwise_cohens_d"] = (
        summary["ranking"][0] if summary["ranking"] else None
    )

    summary_path = args.out_dir / f"{prefix}_separation_summary.json"
    pairwise_path = args.out_dir / f"{prefix}_pairwise_separation.tsv"
    base_stats_path = args.out_dir / f"{prefix}_base_stats.tsv"

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    write_tsv(
        path=pairwise_path,
        rows=pairwise_rows_all,
        fieldnames=[
            "offset_label",
            "offset",
            "base_pair",
            "base_a",
            "base_b",
            "n_a",
            "n_b",
            "mean_a",
            "mean_b",
            "mean_delta",
            "cohens_d",
            "bhattacharyya_distance",
        ],
    )

    write_tsv(
        path=base_stats_path,
        rows=base_stats_rows_all,
        fieldnames=[
            "offset_label",
            "offset",
            "base",
            "count",
            "mean",
            "std",
            "median",
            "min",
            "max",
            "p01",
            "p05",
            "p25",
            "p75",
            "p95",
            "p99",
        ],
    )

    print("\n[INFO] ===== Ranking by min pairwise Cohen's d =====")
    for item in summary["ranking"]:
        print(
            f"[INFO] #{item['rank']} {item['offset_label']}: "
            f"min_d={item['min_pairwise_cohens_d']:.6f}, "
            f"mean_d={item['mean_pairwise_cohens_d']:.6f}, "
            f"mean_BD={item['mean_pairwise_bhattacharyya_distance']:.6f}, "
            f"intervals={item['total_intervals']}"
        )

    best = summary["best_offset_by_min_pairwise_cohens_d"]

    if best is not None:
        print(f"\n[INFO] ===== Best offset: {best['offset_label']} =====")
        print(f"[INFO] min_pairwise_cohens_d: {best['min_pairwise_cohens_d']:.6f}")
        print(f"[INFO] mean_pairwise_cohens_d: {best['mean_pairwise_cohens_d']:.6f}")
        print(f"[INFO] plot_path: {best['plot_path']}")

    print("\n[INFO] ===== Output files =====")
    print(f"[INFO] summary: {summary_path}")
    print(f"[INFO] pairwise: {pairwise_path}")
    print(f"[INFO] base_stats: {base_stats_path}")

    print("\n[INFO] ===== Run stats =====")
    for key, value in asdict(stats).items():
        print(f"[INFO] {key}: {value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())