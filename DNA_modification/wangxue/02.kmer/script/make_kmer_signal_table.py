#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate k-mer signal summary table from JSONL / JSONL.GZ.

For each read:
    - use current span field, e.g. base_sample_spans_rel
    - each span corresponds to a shifted base position
    - kmer = shifted base + following 4 bases

For each span index i:
    base_index = i + base_offset
    kmer = sequence[base_index : base_index + kmer_size]

Default:
    base_offset = 4
    kmer_size = 5

This means:
    span[0] -> sequence[4:9]
    span[1] -> sequence[5:10]
    ...

Tail positions with incomplete k-mer are skipped.

Output columns:
    kmer
    current.mean
    current.stdv
    current_sd.mean
    current_sd.stdv
    count
    dwell.mean
    dwell.stdv

Example:
python make_kmer_signal_table.py \
  --data-jsonl signal_none.adjusted.jsonl.gz \
  --out-tsv kmer_signal_table.tsv \
  --span-field base_sample_spans_rel \
  --base-offset 4 \
  --kmer-size 5
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, TextIO

import numpy as np


# -----------------------------
# Defaults
# -----------------------------

DEFAULT_DATA_JSONL = Path("signal_none.adjusted.jsonl.gz")
DEFAULT_OUT_TSV = Path("kmer_signal_table.tsv")

DEFAULT_PATTERN_FIELD = "pattern"
DEFAULT_SIGNAL_FIELD = "signal"
DEFAULT_SPAN_FIELD = "base_sample_spans_rel"
DEFAULT_READ_ID_FIELD = "read_id"

DEFAULT_BASE_OFFSET = 4
DEFAULT_KMER_SIZE = 5

VALID_BASES = set("ATGC")


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate k-mer signal summary table from JSONL/JSONL.GZ."
    )

    parser.add_argument(
        "--data-jsonl",
        type=Path,
        default=DEFAULT_DATA_JSONL,
        help="Input JSONL or JSONL.GZ file.",
    )

    parser.add_argument(
        "--out-tsv",
        type=Path,
        default=DEFAULT_OUT_TSV,
        help="Output TSV file.",
    )

    parser.add_argument(
        "--pattern-field",
        default=DEFAULT_PATTERN_FIELD,
        help="Sequence field name in JSONL.",
    )

    parser.add_argument(
        "--signal-field",
        default=DEFAULT_SIGNAL_FIELD,
        help="Signal field name in JSONL.",
    )

    parser.add_argument(
        "--span-field",
        default=DEFAULT_SPAN_FIELD,
        help=(
            "Span field name in JSONL, for example "
            "base_sample_spans_rel or base_sample_spans_rel_adj."
        ),
    )

    parser.add_argument(
        "--read-id-field",
        default=DEFAULT_READ_ID_FIELD,
        help="Read ID field name in JSONL.",
    )

    parser.add_argument(
        "--base-offset",
        type=int,
        default=DEFAULT_BASE_OFFSET,
        help=(
            "Base offset between span index and sequence index. "
            "Default 4 means span[i] maps to sequence[i + 4]."
        ),
    )

    parser.add_argument(
        "--kmer-size",
        type=int,
        default=DEFAULT_KMER_SIZE,
        help="K-mer size. Default: 5.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum records to process. Use 0 for no limit.",
    )

    parser.add_argument(
        "--min-dwell",
        type=int,
        default=1,
        help="Minimum span length required. Default: 1.",
    )

    parser.add_argument(
        "--ddof",
        type=int,
        default=1,
        choices=[0, 1],
        help=(
            "Degrees of freedom for standard deviation. "
            "1 = sample std, 0 = population std. Default: 1."
        ),
    )

    parser.add_argument(
        "--allow-n",
        action="store_true",
        help="Allow kmers containing non-ATGC bases. Default: skip non-ATGC kmers.",
    )

    return parser.parse_args()


# -----------------------------
# Utility
# -----------------------------

def open_text_auto(path: Path) -> TextIO:
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def safe_float_array(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def validate_spans(spans: Any) -> bool:
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

    return True


def clip_interval(start: int, end: int, signal_len: int) -> tuple[int, int]:
    start = max(0, min(start, signal_len))
    end = max(0, min(end, signal_len))
    return start, end


def is_valid_kmer(kmer: str, allow_n: bool = False) -> bool:
    if allow_n:
        return len(kmer) > 0
    return all(base in VALID_BASES for base in kmer)


def std_value(values: list[float], ddof: int) -> float:
    if len(values) == 0:
        return float("nan")

    if len(values) == 1:
        return 0.0

    return float(np.std(np.asarray(values, dtype=np.float64), ddof=ddof))


def mean_value(values: list[float]) -> float:
    if len(values) == 0:
        return float("nan")

    return float(np.mean(np.asarray(values, dtype=np.float64)))


# -----------------------------
# Core
# -----------------------------

def process_record(
    record: dict[str, Any],
    pattern_field: str,
    signal_field: str,
    span_field: str,
    base_offset: int,
    kmer_size: int,
    min_dwell: int,
    allow_n: bool,
    table: dict[str, dict[str, list[float]]],
) -> int:
    """
    Process one read and update k-mer aggregation table.

    Returns:
        number of valid span/kmer observations added.
    """
    seq = str(record[pattern_field]).upper()
    signal = safe_float_array(record[signal_field])
    spans = record[span_field]

    if signal.size == 0:
        return 0

    if not validate_spans(spans):
        return 0

    n_added = 0
    signal_len = signal.size

    for span_index, span in enumerate(spans):
        base_index = span_index + base_offset
        kmer_end = base_index + kmer_size

        # Tail incomplete k-mer: skip directly
        if base_index < 0 or kmer_end > len(seq):
            continue

        kmer = seq[base_index:kmer_end]

        if len(kmer) != kmer_size:
            continue

        if not is_valid_kmer(kmer, allow_n=allow_n):
            continue

        try:
            start = int(math.floor(float(span[0])))
            end = int(math.ceil(float(span[1])))
        except Exception:
            continue

        start, end = clip_interval(start, end, signal_len)

        dwell = end - start

        if dwell < min_dwell:
            continue

        segment = signal[start:end]

        if segment.size < min_dwell:
            continue

        current = float(np.mean(segment))

        if segment.size > 1:
            current_sd = float(np.std(segment, ddof=1))
        else:
            current_sd = 0.0

        if not math.isfinite(current) or not math.isfinite(current_sd):
            continue

        table[kmer]["current"].append(current)
        table[kmer]["current_sd"].append(current_sd)
        table[kmer]["dwell"].append(float(dwell))

        n_added += 1

    return n_added


def build_kmer_table(
    data_jsonl: Path,
    pattern_field: str,
    signal_field: str,
    span_field: str,
    read_id_field: str,
    base_offset: int,
    kmer_size: int,
    limit: int,
    min_dwell: int,
    allow_n: bool,
) -> tuple[dict[str, dict[str, list[float]]], dict[str, Any]]:
    table: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {
            "current": [],
            "current_sd": [],
            "dwell": [],
        }
    )

    stats = {
        "records_seen": 0,
        "records_loaded": 0,
        "records_processed": 0,
        "records_skipped_missing_field": 0,
        "records_skipped_no_observation": 0,
        "records_failed_exception": 0,
        "observations_added": 0,
    }

    required_fields = [
        pattern_field,
        signal_field,
        span_field,
    ]

    if not data_jsonl.exists():
        raise FileNotFoundError(f"Input file does not exist: {data_jsonl}")

    with open_text_auto(data_jsonl) as handle:
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
                stats["records_skipped_missing_field"] += 1
                continue

            stats["records_loaded"] += 1

            try:
                n_added = process_record(
                    record=record,
                    pattern_field=pattern_field,
                    signal_field=signal_field,
                    span_field=span_field,
                    base_offset=base_offset,
                    kmer_size=kmer_size,
                    min_dwell=min_dwell,
                    allow_n=allow_n,
                    table=table,
                )

                if n_added > 0:
                    stats["records_processed"] += 1
                    stats["observations_added"] += n_added
                else:
                    stats["records_skipped_no_observation"] += 1

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
                    f"processed={stats['records_processed']}, "
                    f"observations={stats['observations_added']}, "
                    f"kmers={len(table)}",
                    file=sys.stderr,
                )

    return table, stats


def write_kmer_summary(
    table: dict[str, dict[str, list[float]]],
    out_tsv: Path,
    ddof: int,
) -> None:
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "kmer",
        "current.mean",
        "current.stdv",
        "current_sd.mean",
        "current_sd.stdv",
        "count",
        "dwell.mean",
        "dwell.stdv",
    ]

    rows = []

    for kmer, values in table.items():
        current_values = values["current"]
        current_sd_values = values["current_sd"]
        dwell_values = values["dwell"]

        count = len(current_values)

        if count == 0:
            continue

        row = {
            "kmer": kmer,
            "current.mean": mean_value(current_values),
            "current.stdv": std_value(current_values, ddof=ddof),
            "current_sd.mean": mean_value(current_sd_values),
            "current_sd.stdv": std_value(current_sd_values, ddof=ddof),
            "count": count,
            "dwell.mean": mean_value(dwell_values),
            "dwell.stdv": std_value(dwell_values, ddof=ddof),
        }

        rows.append(row)

    rows.sort(key=lambda x: x["kmer"])

    with out_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


# -----------------------------
# Main
# -----------------------------

def main() -> int:
    args = parse_args()

    if args.limit < 0:
        raise ValueError("--limit must be >= 0. Use 0 for no limit.")

    if args.kmer_size <= 0:
        raise ValueError("--kmer-size must be positive.")

    if args.min_dwell <= 0:
        raise ValueError("--min-dwell must be positive.")

    print("[INFO] ===== Config =====")
    print(f"[INFO] data_jsonl: {args.data_jsonl}")
    print(f"[INFO] out_tsv: {args.out_tsv}")
    print(f"[INFO] pattern_field: {args.pattern_field}")
    print(f"[INFO] signal_field: {args.signal_field}")
    print(f"[INFO] span_field: {args.span_field}")
    print(f"[INFO] base_offset: {args.base_offset}")
    print(f"[INFO] kmer_size: {args.kmer_size}")
    print(f"[INFO] limit: {args.limit}")
    print(f"[INFO] min_dwell: {args.min_dwell}")
    print(f"[INFO] ddof: {args.ddof}")
    print(f"[INFO] allow_n: {args.allow_n}")

    table, stats = build_kmer_table(
        data_jsonl=args.data_jsonl,
        pattern_field=args.pattern_field,
        signal_field=args.signal_field,
        span_field=args.span_field,
        read_id_field=args.read_id_field,
        base_offset=args.base_offset,
        kmer_size=args.kmer_size,
        limit=args.limit,
        min_dwell=args.min_dwell,
        allow_n=args.allow_n,
    )

    write_kmer_summary(
        table=table,
        out_tsv=args.out_tsv,
        ddof=args.ddof,
    )

    print("\n[INFO] ===== Run stats =====")
    for key, value in stats.items():
        print(f"[INFO] {key}: {value}")

    print("\n[INFO] ===== Output =====")
    print(f"[INFO] kmer_count: {len(table)}")
    print(f"[INFO] output_tsv: {args.out_tsv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())