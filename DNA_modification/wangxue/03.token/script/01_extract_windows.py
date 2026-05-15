#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np
import h5py
import matplotlib.pyplot as plt


# =========================
# Normalize (Median-MAD)
# =========================
def nanopore_normalize_mad(signal: np.ndarray) -> np.ndarray:
    if signal.size == 0:
        return np.array([], dtype=np.float32)
    med = np.median(signal)
    mad = 1.4826 * np.median(np.abs(signal - med))
    if mad < 1e-6:
        return np.array([], dtype=np.float32)
    return ((signal - med) / mad).astype(np.float32)


# =========================
# MV -> base mapping
# =========================
def step_base_from_mv(mv):
    mv = np.asarray(mv, dtype=np.int32)
    step_base = np.cumsum(mv) - 1
    step_base[step_base < 0] = 0
    return step_base


def base_to_step_bounds(step_base, b0, b1):
    mask = (step_base >= b0) & (step_base < b1)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None, None
    return int(idx[0]), int(idx[-1] + 1)


def read_signal(h5, key):
    return h5["signals"][key][:].astype(np.float32)


# =========================
# Pattern occurrences
# =========================
def find_all_occurrences(seq: str, pattern: str):
    starts = []
    i = 0
    while True:
        j = seq.find(pattern, i)
        if j < 0:
            break
        starts.append(j)
        i = j + 1
    return starts


# =========================
# Build per-base raw-sample spans (relative to window start)
# =========================
def build_base_sample_spans_rel(
    step_base,
    occ_start_base: int,
    pat_len: int,
    win_from: int,
    mv_stride: int,
    signal_len: int,
):
    """
    Returns:
      base_sample_spans_rel: List[[rel0, rel1]] length=pat_len
      base_abs_spans:        List[[abs0, abs1]] length=pat_len
      ok_any:                bool (at least one valid base span)
    """
    base_sample_spans_rel = []
    base_abs_spans = []
    ok_any = False

    for pos in range(pat_len):
        bb0 = occ_start_base + pos
        bb1 = bb0 + 1

        s0, s1 = base_to_step_bounds(step_base, bb0, bb1)
        if s0 is None:
            base_sample_spans_rel.append([-1, -1])
            base_abs_spans.append([-1, -1])
            continue

        abs0 = int(s0 * mv_stride)
        abs1 = int(s1 * mv_stride)

        # clamp to signal length
        abs0 = max(0, min(abs0, signal_len))
        abs1 = max(0, min(abs1, signal_len))
        if abs0 >= abs1:
            base_sample_spans_rel.append([-1, -1])
            base_abs_spans.append([-1, -1])
            continue

        # relative to window start (clamp to [0, win_len])
        rel0 = abs0 - win_from
        rel1 = abs1 - win_from

        # If window starts inside this base span, rel0 may be <0; clamp it.
        rel0 = max(0, rel0)
        rel1 = max(0, rel1)

        if rel0 >= rel1:
            base_sample_spans_rel.append([-1, -1])
            base_abs_spans.append([abs0, abs1])
            continue

        ok_any = True
        base_sample_spans_rel.append([int(rel0), int(rel1)])
        base_abs_spans.append([int(abs0), int(abs1)])

    return base_sample_spans_rel, base_abs_spans, ok_any


# =========================
# Plot batch (one subplot per read)
# =========================
def save_batch_png_subplots(batch, batch_idx, out_dir, pattern, args):
    """
    batch: list of dict:
      {
        rid, signal, win_from, win_to, pat_from, pat_to,
        base_abs_spans (list of [abs0,abs1]) length=pat_len
      }
    """
    n = len(batch)
    if n == 0:
        return None

    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, f"{pattern}_batch_{batch_idx:04d}.png")

    # batch y-lim if requested
    y_min, y_max = None, None
    if args.y_lim_mode == "batch" or args.share_ylim:
        ymn = float("inf")
        ymx = float("-inf")
        for item in batch:
            sig = item["signal"]
            wf, wt = item["win_from"], item["win_to"]
            y = sig[wf:wt]
            if args.downsample_plot and args.downsample_plot > 1:
                y = y[::args.downsample_plot]
            if y.size == 0:
                continue
            ymn = min(ymn, float(np.min(y)))
            ymx = max(ymx, float(np.max(y)))
        if np.isfinite(ymn) and np.isfinite(ymx) and ymn < ymx:
            y_min, y_max = ymn, ymx

    fig_h = max(8, n * args.row_height_inch)
    fig, axes = plt.subplots(
        nrows=n, ncols=1,
        figsize=(args.fig_width_inch, fig_h),
        dpi=args.dpi,
        sharex=False
    )
    if n == 1:
        axes = [axes]

    pat_len = len(pattern)

    for ax, item in zip(axes, batch):
        rid = item["rid"]
        sig = item["signal"]
        wf, wt = item["win_from"], item["win_to"]
        pf, pt = item["pat_from"], item["pat_to"]
        base_abs_spans = item.get("base_abs_spans", [])

        x = np.arange(wf, wt, dtype=np.int32)
        y = sig[wf:wt]

        if args.downsample_plot and args.downsample_plot > 1:
            x = x[::args.downsample_plot]
            y = y[::args.downsample_plot]

        ax.plot(x, y, linewidth=args.linewidth)

        # highlight pattern region
        ax.axvspan(pf, pt, color="gray", alpha=0.18, linewidth=0)

        # mark each base boundary and optionally highlight one base span
        for pos in range(min(pat_len, len(base_abs_spans))):
            a, b = base_abs_spans[pos]
            if a < 0 or b <= a:
                continue

            lw = 0.5
            alpha = 0.55

            if args.mark_base_pos >= 0 and pos == args.mark_base_pos:
                lw = 1.4
                alpha = 0.9
                ax.axvspan(a, b, color="orange", alpha=0.12, linewidth=0)

            ax.axvline(a, linewidth=lw, alpha=alpha)
            ax.axvline(b, linewidth=lw, alpha=alpha)

            # base letter near top of axis (use y-lim)
            y0, y1 = ax.get_ylim()
            y_text = y1 - 0.03 * (y1 - y0)
            mid = (a + b) / 2.0
            ax.text(mid, y_text, pattern[pos], fontsize=7, ha="center", va="top", alpha=0.9)

        ax.set_title(rid, fontsize=9, loc="left")
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(False)

        # y-axis control
        if args.y_lim_mode == "manual":
            ax.set_ylim(args.y_lim[0], args.y_lim[1])
        elif args.y_lim_mode == "batch" or args.share_ylim:
            if y_min is not None and y_max is not None:
                ax.set_ylim(y_min, y_max)
        elif args.y_lim_mode == "auto":
            pass
        else:
            raise ValueError(f"Unknown y_lim_mode: {args.y_lim_mode}")

    axes[-1].set_xlabel("sample index", fontsize=10)
    fig.suptitle(
        f"pattern={pattern} | batch={batch_idx:04d} | n={n} | norm={args.normalize_mode} "
        f"| plot_ds={args.downsample_plot} | ylim={args.y_lim_mode} | mark_base={args.mark_base_pos}",
        fontsize=12, y=0.995
    )
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(out_png)
    plt.close(fig)
    return out_png


# =========================
# QC helpers
# =========================
def safe_percentiles(arr, qs=(0, 1, 5, 25, 50, 75, 95, 99, 100)):
    if len(arr) == 0:
        return None
    a = np.asarray(arr, dtype=np.float32)
    return {str(q): float(np.percentile(a, q)) for q in qs}


# =========================
# Main
# =========================
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_txt), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)

    batch = []
    batch_idx = 0
    matched = 0
    seen = 0

    # QC stats
    mv_value_counter = Counter()
    ratios_signal_steps = []
    seq_len_list = []
    max_base_in_mv_list = []
    span_len_list = []        # base span length in raw samples (abs1-abs0)
    empty_base_span_cnt = 0
    missing_signal_cnt = 0
    seq_out_of_mv_cnt = 0

    txt_out = open(args.out_txt, "w", encoding="utf-8")
    txt_out.write(
        "read_id\tocc_start_base\tocc_end_base\tpat_signal_start\tpat_signal_end\twin_signal_start\twin_signal_end\n"
    )
    jsonl_out = open(args.out_jsonl, "w", encoding="utf-8")

    with h5py.File(args.h5_path, "r") as h5, open(args.jsonl_path, "r", encoding="utf-8") as f:
        if "signals" not in h5:
            raise ValueError(f"H5 missing group '/signals': {args.h5_path}")

        for line in f:
            seen += 1
            rec = json.loads(line)

            # 1) strand filter
            if rec.get("align_strand") != args.target_strand:
                continue

            rid = rec.get("id") or rec.get("read_id")
            seq = rec.get("seq") or ""
            mv = rec.get("moves")

            if not rid or not seq or mv is None:
                continue

            # pattern occurrences
            occ_starts = find_all_occurrences(seq, args.pattern)
            if not occ_starts:
                continue

            # signal key
            key = rec.get("signal_key") or rid
            if key not in h5["signals"]:
                missing_signal_cnt += 1
                continue

            signal_raw = read_signal(h5, key)

            # normalize
            if args.normalize_mode == "mad":
                signal = nanopore_normalize_mad(signal_raw)
                if signal.size == 0:
                    continue
            elif args.normalize_mode == "none":
                signal = signal_raw
            else:
                raise ValueError(f"Unknown normalize_mode: {args.normalize_mode}")

            # QC: moves values + ratio
            mv_arr = np.asarray(mv, dtype=np.int32)
            # only count a small sample to avoid huge counter cost
            for v in np.unique(mv_arr):
                mv_value_counter[int(v)] += 1

            n_steps = int(len(mv_arr))
            if n_steps > 0:
                ratios_signal_steps.append(float(len(signal)) / float(n_steps))

            step_base = step_base_from_mv(mv_arr)
            max_base_in_mv = int(step_base.max()) if step_base.size else -1

            seq_len_list.append(len(seq))
            max_base_in_mv_list.append(max_base_in_mv)

            pat_len = len(args.pattern)

            for start in occ_starts:
                b0 = int(start)
                b1 = int(start + pat_len)

                # QC: seq index out of moves coverage (possible offset)
                if max_base_in_mv >= 0 and b1 > (max_base_in_mv + 1):
                    seq_out_of_mv_cnt += 1
                    continue

                # pattern step bounds
                s0, s1 = base_to_step_bounds(step_base, b0, b1)
                if s0 is None:
                    continue

                pat_from = int(s0 * args.stride)
                pat_to = int(s1 * args.stride)

                # clip
                pat_from = max(0, min(pat_from, len(signal)))
                pat_to = max(0, min(pat_to, len(signal)))
                if pat_from >= pat_to:
                    continue

                win_from = max(0, pat_from - args.pad_samples)
                win_to = min(len(signal), pat_to + args.pad_samples)
                if win_from >= win_to:
                    continue

                # per-base spans
                base_sample_spans_rel, base_abs_spans, ok_any = build_base_sample_spans_rel(
                    step_base=step_base,
                    occ_start_base=b0,
                    pat_len=pat_len,
                    win_from=win_from,
                    mv_stride=args.stride,
                    signal_len=len(signal),
                )
                if not ok_any:
                    continue

                # QC span lens
                for a, b in base_abs_spans:
                    if a < 0 or b <= a:
                        empty_base_span_cnt += 1
                    else:
                        span_len_list.append(int(b - a))

                # write TXT
                txt_out.write(f"{rid}\t{b0}\t{b1}\t{pat_from}\t{pat_to}\t{win_from}\t{win_to}\n")

                # write JSONL
                rec_json = {
                    "read_id": rid,
                    "label": rec.get("label", None),
                    "pattern": args.pattern,
                    "occ_start_base": b0,
                    "occ_end_base": b1,
                    "mv_stride": args.stride,
                    "pad_samples": args.pad_samples,
                    "normalize_mode": args.normalize_mode,
                    "pat_from": pat_from,
                    "pat_to": pat_to,
                    "win_from": win_from,
                    "win_to": win_to,
                    "base_sample_spans_rel": base_sample_spans_rel,
                }

                if args.write_moves:
                    rec_json["moves"] = mv

                if args.write_seq:
                    rec_json["seq"] = seq

                if args.write_window_signal:
                    rec_json["signal"] = signal[win_from:win_to].astype(np.float32).tolist()

                # extra QC context
                if args.write_debug_meta:
                    rec_json["signal_full_len"] = int(len(signal))
                    rec_json["n_steps"] = int(n_steps)
                    rec_json["max_base_in_mv"] = int(max_base_in_mv)
                    rec_json["seq_len"] = int(len(seq))
                    rec_json["signal_key"] = key

                jsonl_out.write(json.dumps(rec_json, ensure_ascii=False) + "\n")

                # plot
                matched += 1
                if not args.no_plot:
                    batch.append({
                        "rid": f"{rid} | occ={b0}",
                        "signal": signal,
                        "win_from": win_from,
                        "win_to": win_to,
                        "pat_from": pat_from,
                        "pat_to": pat_to,
                        "base_abs_spans": base_abs_spans,
                    })

                    if len(batch) >= args.batch_size:
                        batch_idx += 1
                        out_png = save_batch_png_subplots(batch, batch_idx, args.out_dir, args.pattern, args)
                        print(f"[PNG] wrote: {out_png}")
                        batch = []

                if matched >= args.max_plots_reads:
                    break

            if matched >= args.max_plots_reads:
                break

    # flush last plot batch
    if (not args.no_plot) and batch:
        batch_idx += 1
        out_png = save_batch_png_subplots(batch, batch_idx, args.out_dir, args.pattern, args)
        print(f"[PNG] wrote: {out_png}")

    txt_out.close()
    jsonl_out.close()

    # QC summary
    qc = {
        "scanned_lines": int(seen),
        "matched_records_written": int(matched),
        "missing_signal_cnt": int(missing_signal_cnt),
        "seq_out_of_mv_cnt": int(seq_out_of_mv_cnt),
        "moves_unique_values": {str(k): int(v) for k, v in sorted(mv_value_counter.items())},
        "signal_len_div_steps_percentiles": safe_percentiles(ratios_signal_steps),
        "seq_len_percentiles": safe_percentiles(seq_len_list),
        "max_base_in_mv_percentiles": safe_percentiles(max_base_in_mv_list),
        "base_span_len_samples_percentiles": safe_percentiles(span_len_list),
        "empty_or_invalid_base_span_count": int(empty_base_span_cnt),
        "notes": {
            "stride_check": "If signal_len/steps is near your STRIDE (e.g., ~5), stride is likely correct.",
            "offset_check": "If seq_out_of_mv_cnt is high, seq base indices may not align with moves base indices (offset).",
        }
    }

    qc_path = os.path.join(args.out_dir, "qc_summary.json")
    with open(qc_path, "w", encoding="utf-8") as fw:
        json.dump(qc, fw, ensure_ascii=False, indent=2)

    print(f"[DONE] scanned lines: {seen}")
    print(f"[DONE] matched (written up to {args.max_plots_reads}): {matched}")
    print(f"[DONE] PNG dir: {args.out_dir} (batches: {batch_idx})")
    print(f"[DONE] result TXT: {args.out_txt}")
    print(f"[DONE] result JSONL: {args.out_jsonl}")
    print(f"[DONE] QC summary: {qc_path}")
    print(f"[NOTE] Step1 outputs raw window signal (no downsample); tokenizer will handle stride/compression later.")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Step1: Extract motif windows from nanopore signals using movetable, "
                    "write window signal + per-base spans for later token slicing."
    )

    ap.add_argument("--jsonl_path", required=True, help="movetable jsonl (with seq, moves, align_strand, id/signal_key)")
    ap.add_argument("--h5_path", required=True, help="signal.h5 with group /signals/<key>")

    ap.add_argument("--pattern", default="ATAACAGGT")
    ap.add_argument("--target_strand", type=int, default=0)

    ap.add_argument("--stride", type=int, default=5, help="mv stride: raw_samples_per_step (often 5)")
    ap.add_argument("--pad_samples", type=int, default=0)

    ap.add_argument("--max_plots_reads", type=int, default=1000, help="max records (occurrences) to write/plot")
    ap.add_argument("--batch_size", type=int, default=100)

    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_txt", required=True)
    ap.add_argument("--out_jsonl", required=True)

    # Output control
    ap.add_argument("--normalize_mode", default="mad", choices=["mad", "none"])
    ap.add_argument("--write_window_signal", action="store_true", help="write window signal to JSONL")
    ap.add_argument("--write_moves", action="store_true", help="write moves to JSONL (file can be large)")
    ap.add_argument("--write_seq", action="store_true", help="write seq to JSONL (for alignment checks)")
    ap.add_argument("--write_debug_meta", action="store_true", help="write extra meta: signal_full_len, n_steps, ...")

    # Plot
    ap.add_argument("--no_plot", action="store_true")
    ap.add_argument("--downsample_plot", type=int, default=1, help="downsample for plotting only")
    ap.add_argument("--linewidth", type=float, default=0.8)
    ap.add_argument("--fig_width_inch", type=float, default=18)
    ap.add_argument("--row_height_inch", type=float, default=1.0)
    ap.add_argument("--dpi", type=int, default=120)

    ap.add_argument("--share_ylim", action="store_true")
    ap.add_argument("--y_lim_mode", default="manual", choices=["auto", "batch", "manual"])
    ap.add_argument("--y_lim", type=float, nargs=2, default=(-3, 3), help="manual y-lim: min max")

    ap.add_argument("--mark_base_pos", type=int, default=4,
                    help="highlight base position in pattern (0-based). -1 disables")

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)