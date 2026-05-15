import argparse
import csv
import json
import math
import os
from typing import List, Optional, Dict, Any

import numpy as np


def safe_float(x):
    if x is None:
        return None
    if isinstance(x, (float, np.floating)) and (math.isnan(x) or math.isinf(x)):
        return None
    return float(x)


def compute_skewness(seg: np.ndarray) -> float:
    """
    计算偏度（skewness）
    使用总体三阶中心矩定义：
        skew = E[((x - mu) / sigma)^3]

    若长度太短或 std=0，则返回 nan
    """
    if seg.size < 3:
        return float("nan")

    mu = np.mean(seg)
    sigma = np.std(seg)

    if sigma == 0:
        return float("nan")

    z = (seg - mu) / sigma
    return float(np.mean(z ** 3))



def compute_base_signal_features(
    signal: np.ndarray,
    base_sample_spans_rel: List[List[int]],
    sampling_rate: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    对每个 base 对应的 signal 片段提取一组统计特征。

    Args:
        signal: 1D np.ndarray
        base_sample_spans_rel: 每个 base 的相对 signal 区间 [a, b)
        sampling_rate: 可选，采样率（Hz）。若提供，则额外输出 dwell_time_ms

    Returns:
        List[dict]，每个 base 一个 dict
    """
    feats = []
    n = len(signal)

    for span in base_sample_spans_rel:
        # 默认空特征
        empty = {
            "mean": None,
            "std": None,
            "skewness": None,
            "dwell_time": None,
            "dwell_time_ms": None,
            "median": None,
            "min": None,
            "max": None,
            "range": None,
            "q25": None,
            "q75": None,
        }

        if span is None or len(span) != 2:
            feats.append(empty)
            continue

        a, b = span
        if a < 0 or b <= a or a >= n:
            feats.append(empty)
            continue

        b = min(b, n)
        seg = signal[a:b]

        if seg.size == 0:
            feats.append(empty)
            continue

        mean_v = float(np.mean(seg))
        std_v = float(np.std(seg))
        median_v = float(np.median(seg))
        min_v = float(np.min(seg))
        max_v = float(np.max(seg))
        range_v = max_v - min_v
        q25_v = float(np.percentile(seg, 25))
        q75_v = float(np.percentile(seg, 75))
        skew_v = compute_skewness(seg)
        dwell_v = int(b - a)

        if sampling_rate is not None and sampling_rate > 0:
            dwell_ms_v = float(dwell_v / sampling_rate * 1000.0)
        else:
            dwell_ms_v = None

        feats.append({
            "mean": safe_float(mean_v),
            "std": safe_float(std_v),
            "skewness": safe_float(skew_v),
            "dwell_time": dwell_v,
            "dwell_time_ms": safe_float(dwell_ms_v),
            "median": safe_float(median_v),
            "min": safe_float(min_v),
            "max": safe_float(max_v),
            "range": safe_float(range_v),
            "q25": safe_float(q25_v),
            "q75": safe_float(q75_v),
        })

    return feats


def main():
    ap = argparse.ArgumentParser(
        description="Step2: compute per-base signal features from Step1 JSONL output."
    )
    ap.add_argument("--in_jsonl", required=True,
                    help="Step1 output JSONL, must contain signal + base_sample_spans_rel")
    ap.add_argument("--out_jsonl", required=True,
                    help="Output JSONL with per-base signal features")
    ap.add_argument("--out_csv", default=None,
                    help="Optional long-format CSV output")
    ap.add_argument("--keep_signal", action="store_true",
                    help="Keep original signal in output JSONL")
    ap.add_argument("--keep_spans", action="store_true",
                    help="Keep base_sample_spans_rel in output JSONL")
    ap.add_argument("--sampling_rate", type=float, default=None,
                    help="Optional signal sampling rate in Hz; if given, dwell_time_ms will be computed")
    args = ap.parse_args()

    out_jsonl_dir = os.path.dirname(args.out_jsonl)
    if out_jsonl_dir:
        os.makedirs(out_jsonl_dir, exist_ok=True)

    if args.out_csv:
        out_csv_dir = os.path.dirname(args.out_csv)
        if out_csv_dir:
            os.makedirs(out_csv_dir, exist_ok=True)

    n_total = 0
    n_ok = 0
    n_skip = 0

    csv_writer = None
    csv_f = None

    if args.out_csv:
        csv_f = open(args.out_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(
            csv_f,
            fieldnames=[
                "read_id",
                "label",
                "pattern",
                "occ_start_base",
                "occ_end_base",
                "base_idx_in_pattern",
                "base",
                "mean",
                "std",
                "skewness",
                "dwell_time",
                "dwell_time_ms",
                "median",
                "min",
                "max",
                "range",
                "q25",
                "q75",
            ]
        )
        csv_writer.writeheader()

    with open(args.in_jsonl, "r", encoding="utf-8") as fr, \
         open(args.out_jsonl, "w", encoding="utf-8") as fw:

        for line in fr:
            n_total += 1
            rec = json.loads(line)

            signal = rec.get("signal", None)
            spans = rec.get("base_sample_spans_rel", None)

            # 这里我保留 read，不直接丢弃；这样更利于后续和 embedding 对齐
            if signal is None or spans is None:
                out_rec = {
                    "read_id": rec.get("read_id"),
                    "label": rec.get("label"),
                    "pattern": rec.get("pattern"),
                    "occ_start_base": rec.get("occ_start_base"),
                    "occ_end_base": rec.get("occ_end_base"),
                    "base_features": None,
                }
                fw.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                n_skip += 1
                continue

            signal = np.asarray(signal, dtype=np.float32)
            base_features = compute_base_signal_features(
                signal=signal,
                base_sample_spans_rel=spans,
                sampling_rate=args.sampling_rate,
            )

            out_rec = {
                "read_id": rec.get("read_id"),
                "label": rec.get("label"),
                "pattern": rec.get("pattern"),
                "occ_start_base": rec.get("occ_start_base"),
                "occ_end_base": rec.get("occ_end_base"),
                "base_features": base_features,
            }

            seq = rec.get("seq", None)
            b0 = rec.get("occ_start_base", None)
            b1 = rec.get("occ_end_base", None)
            if seq is not None and b0 is not None and b1 is not None:
                motif_seq = seq[b0:b1]
                out_rec["bases"] = list(motif_seq)

            if args.keep_signal:
                out_rec["signal"] = rec["signal"]

            if args.keep_spans:
                out_rec["base_sample_spans_rel"] = rec["base_sample_spans_rel"]

            fw.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            n_ok += 1

            if csv_writer is not None:
                bases = out_rec.get("bases", None)
                for i, feat in enumerate(base_features):
                    csv_writer.writerow({
                        "read_id": out_rec.get("read_id"),
                        "label": out_rec.get("label"),
                        "pattern": out_rec.get("pattern"),
                        "occ_start_base": out_rec.get("occ_start_base"),
                        "occ_end_base": out_rec.get("occ_end_base"),
                        "base_idx_in_pattern": i,
                        "base": bases[i] if (bases is not None and i < len(bases)) else None,
                        "mean": feat.get("mean"),
                        "std": feat.get("std"),
                        "skewness": feat.get("skewness"),
                        "dwell_time": feat.get("dwell_time"),
                        "dwell_time_ms": feat.get("dwell_time_ms"),
                        "median": feat.get("median"),
                        "min": feat.get("min"),
                        "max": feat.get("max"),
                        "range": feat.get("range"),
                        "q25": feat.get("q25"),
                        "q75": feat.get("q75"),
                    })

    if csv_f is not None:
        csv_f.close()

    print(f"[DONE] total={n_total} ok={n_ok} skip={n_skip}")
    print(f"[DONE] out_jsonl={args.out_jsonl}")
    if args.out_csv:
        print(f"[DONE] out_csv={args.out_csv}")


if __name__ == "__main__":
    main()
