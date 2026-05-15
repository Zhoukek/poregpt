import argparse
import json
import os
from collections import Counter

import numpy as np
import h5py
import matplotlib.pyplot as plt
from poregpt.utils import nanopore_process_signal




# =========================
# ATGCN utils
# =========================
VALID_SEQ_BASES = {"A", "T", "G", "C"}
VALID_PATTERN_BASES = {"A", "T", "G", "C", "N"}

COMP = {
    "A": "T",
    "T": "A",
    "G": "C",
    "C": "G",
    "N": "N",
}


def normalize_base_char(ch: str) -> str:
    """
    Normalize base char:
      - uppercase
      - U -> T
      - others kept as uppercase
    """
    ch = ch.upper()
    if ch == "U":
        return "T"
    return ch


def normalize_seq_atgcn(seq: str) -> str:
    return "".join(normalize_base_char(c) for c in seq)


def validate_pattern_atgcn(pattern: str):
    bad = sorted(set(c for c in pattern if c not in VALID_PATTERN_BASES))
    if bad:
        raise ValueError(
            f"--pattern contains unsupported bases: {bad}. "
            f"Only A/T/G/C/N are supported."
        )



def reverse_complement_atgcn(seq: str) -> str:
    seq = normalize_seq_atgcn(seq)
    return "".join(COMP.get(b, "N") for b in seq[::-1])


def match_atgcn(seq_sub: str, pattern: str) -> bool:
    """
    Pattern supports:
      A/T/G/C exact match
      N matches any A/T/G/C

    Sequence side must be A/T/G/C only.
    """
    if len(seq_sub) != len(pattern):
        return False

    for s, p in zip(seq_sub, pattern):
        s = normalize_base_char(s)
        p = normalize_base_char(p)

        if s not in VALID_SEQ_BASES:
            return False

        if p == "N":
            continue

        if p not in VALID_SEQ_BASES:
            return False

        if s != p:
            return False

    return True


def find_all_occurrences_atgcn(seq: str, pattern: str):
    """
    Overlapping matches are allowed.
    """
    seq = normalize_seq_atgcn(seq)
    pattern = normalize_seq_atgcn(pattern)

    starts = []
    L = len(pattern)
    if L == 0 or len(seq) < L:
        return starts

    for i in range(len(seq) - L + 1):
        if match_atgcn(seq[i:i + L], pattern):
            starts.append(i)
    return starts


# =========================
# MV -> base mapping
# =========================
def step_base_from_mv(mv):
    """
    mv: move table
    Returns:
        step_base[i] = base index (0-based) corresponding to step i
    """
    mv = np.asarray(mv, dtype=np.int32)
    step_base = np.cumsum(mv) - 1
    step_base[step_base < 0] = 0
    return step_base



def base_to_step_bounds(step_base, b0, b1):
    """
    Find step interval [s0, s1) covering base interval [b0, b1)
    """
    mask = (step_base >= b0) & (step_base < b1)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None, None
    return int(idx[0]), int(idx[-1] + 1)


def read_signal(h5, key):
    return h5["signals"][key][:].astype(np.float32)


# =========================
# Build per-base raw-sample spans
# =========================
def build_base_sample_spans_rel(
    step_base,
    occ_start_base: int,
    pat_len: int,
    win_from: int,
    win_to: int,
    mv_stride: int,
    signal_len: int,
):
    """
    Returns:
      base_sample_spans_rel: List[[rel0, rel1]] length=pat_len
      base_abs_spans:        List[[abs0, abs1]] length=pat_len
      ok_any:                bool
    """
    base_sample_spans_rel = []
    base_abs_spans = []
    ok_any = False
    win_len = win_to - win_from

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

        abs0 = max(0, min(abs0, signal_len))
        abs1 = max(0, min(abs1, signal_len))
        if abs0 >= abs1:
            base_sample_spans_rel.append([-1, -1])
            base_abs_spans.append([-1, -1])
            continue

        rel0 = abs0 - win_from
        rel1 = abs1 - win_from
        rel0 = max(0, min(rel0, win_len))
        rel1 = max(0, min(rel1, win_len))

        if rel0 >= rel1:
            base_sample_spans_rel.append([-1, -1])
            base_abs_spans.append([int(abs0), int(abs1)])
            continue

        ok_any = True
        base_sample_spans_rel.append([int(rel0), int(rel1)])
        base_abs_spans.append([int(abs0), int(abs1)])

    return base_sample_spans_rel, base_abs_spans, ok_any


def reverse_spans_rel(base_sample_spans_rel, win_len):
    """
    Reverse local spans so that reversed signal still aligns with the original motif orientation.

    Input spans are for the matched motif orientation in the current read.
    After reverse:
      - signal is reversed
      - spans are coordinate-reversed
      - span order is reversed
    """
    rev = []
    for a, b in base_sample_spans_rel[::-1]:
        if a < 0 or b <= a:
            rev.append([-1, -1])
        else:
            rev.append([int(win_len - b), int(win_len - a)])
    return rev


# =========================
# Plot batch
# =========================
def save_batch_png_subplots(batch, batch_idx, out_dir, pattern, args):
    n = len(batch)
    if n == 0:
        return None

    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, f"{pattern}_batch_{batch_idx:04d}.png")

    y_min, y_max = None, None
    if args.y_lim_mode == "batch" or args.share_ylim:
        ymn = float("inf")
        ymx = float("-inf")
        for item in batch:
            y = item["window_signal"]
            if args.downsample_plot > 1:
                y = y[::args.downsample_plot]
            if y.size == 0:
                continue
            ymn = min(ymn, float(np.min(y)))
            ymx = max(ymx, float(np.max(y)))
        if np.isfinite(ymn) and np.isfinite(ymx) and ymn < ymx:
            y_min, y_max = ymn, ymx

    fig_h = max(8, n * args.row_height_inch)
    fig, axes = plt.subplots(
        nrows=n,
        ncols=1,
        figsize=(args.fig_width_inch, fig_h),
        dpi=args.dpi,
        sharex=False
    )
    if n == 1:
        axes = [axes]

    pat_len = len(pattern)
    mark_positions = set(args.mark_base_pos or [])

    for ax, item in zip(axes, batch):
        rid = item["rid"]
        y = item["window_signal"]
        base_spans_rel = item["base_spans_rel"]
        strand = item["strand"]
        matched_pattern = item["pattern_matched"]

        x = np.arange(len(y), dtype=np.int32)

        if args.downsample_plot > 1:
            x = x[::args.downsample_plot]
            y_plot = y[::args.downsample_plot]
        else:
            y_plot = y

        ax.plot(x, y_plot, linewidth=args.linewidth)

        if args.y_lim_mode == "manual":
            ax.set_ylim(args.y_lim[0], args.y_lim[1])
        elif args.y_lim_mode == "batch" or args.share_ylim:
            if y_min is not None and y_max is not None:
                ax.set_ylim(y_min, y_max)
        elif args.y_lim_mode == "auto":
            pass
        else:
            raise ValueError(f"Unknown y_lim_mode: {args.y_lim_mode}")

        for pos in range(min(pat_len, len(base_spans_rel))):
            a, b = base_spans_rel[pos]
            if a < 0 or b <= a:
                continue

            lw = 0.5
            alpha = 0.55

            if pos in mark_positions:
                lw = 1.4
                alpha = 0.9
                ax.axvspan(a, b, color="orange", alpha=0.12, linewidth=0)

            ax.axvline(a, linewidth=lw, alpha=alpha)
            ax.axvline(b, linewidth=lw, alpha=alpha)

            mid = (a + b) / 2.0
            ax.text(
                mid, 0.97, pattern[pos],
                fontsize=7, ha="center", va="top", alpha=0.9,
                transform=ax.get_xaxis_transform()
            )

        strand_name = "forward" if strand == 0 else "reverse"
        ax.set_title(
            f"{rid} | strand={strand_name} | matched={matched_pattern}",
            fontsize=9,
            loc="left"
        )
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(False)

    mark_desc = ",".join(map(str, args.mark_base_pos)) if args.mark_base_pos else "none"
    axes[-1].set_xlabel("window sample index (orientation unified)", fontsize=10)
    fig.suptitle(
        f"pattern={pattern} | batch={batch_idx:04d} | n={n} | norm={args.normalize_mode} "
        f"| plot_ds={args.downsample_plot} | ylim={args.y_lim_mode} | mark_base={mark_desc}",
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
    if args.stride <= 0:
        raise ValueError("--stride must be > 0")
    if args.pad_samples < 0:
        raise ValueError("--pad_samples must be >= 0")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be > 0")
    if args.max_occurrences <= 0:
        raise ValueError("--max_occurrences must be > 0")
    if args.downsample_plot <= 0:
        raise ValueError("--downsample_plot must be > 0")
    if args.row_height_inch <= 0:
        raise ValueError("--row_height_inch must be > 0")
    if args.fig_width_inch <= 0:
        raise ValueError("--fig_width_inch must be > 0")
    if args.dpi <= 0:
        raise ValueError("--dpi must be > 0")
    if len(args.pattern) == 0:
        raise ValueError("--pattern must not be empty")

    pattern_norm = normalize_seq_atgcn(args.pattern)
    validate_pattern_atgcn(pattern_norm)
    pattern_rc = reverse_complement_atgcn(pattern_norm)

    os.makedirs(args.out_dir, exist_ok=True)

    out_txt_dir = os.path.dirname(args.out_txt)
    if out_txt_dir:
        os.makedirs(out_txt_dir, exist_ok=True)

    out_jsonl_dir = os.path.dirname(args.out_jsonl)
    if out_jsonl_dir:
        os.makedirs(out_jsonl_dir, exist_ok=True)

    batch = []
    batch_idx = 0
    written_occurrences = 0
    scanned_lines = 0

    mv_value_counter = Counter()
    ratios_signal_steps = []
    seq_len_list = []
    max_base_in_mv_list = []
    span_len_list = []
    empty_base_span_cnt = 0
    missing_signal_cnt = 0
    seq_out_of_mv_cnt = 0
    forward_match_cnt = 0
    reverse_match_cnt = 0

    txt_out = open(args.out_txt, "w", encoding="utf-8")
    txt_out.write(
        "read_id\tstrand\tocc_start_base\tocc_end_base\tpat_signal_start\tpat_signal_end\twin_signal_start\twin_signal_end\tpattern_query\tpattern_matched\n"
    )
    jsonl_out = open(args.out_jsonl, "w", encoding="utf-8")

    with h5py.File(args.h5_path, "r") as h5, open(args.jsonl_path, "r", encoding="utf-8") as f:
        if "signals" not in h5:
            raise ValueError(f"H5 missing group '/signals': {args.h5_path}")

        for line in f:
            scanned_lines += 1
            rec = json.loads(line)

            strand = rec.get("align_strand")
            if strand not in [0, 1]:
                continue

            rid = rec.get("id") or rec.get("read_id")
            seq = rec.get("seq") or ""
            mv = rec.get("moves")

            if not rid or not seq or mv is None:
                continue

            seq_norm = normalize_seq_atgcn(seq)

            if strand == 0:
                pattern_to_search = pattern_norm
                is_reverse = False
            else:
                pattern_to_search = pattern_rc
                is_reverse = True

            occ_starts = find_all_occurrences_atgcn(seq_norm, pattern_to_search)
            if not occ_starts:
                continue

            key = rec.get("signal_key") or rid
            if key not in h5["signals"]:
                missing_signal_cnt += 1
                continue

            signal_raw = read_signal(h5, key)

            if args.normalize_mode == "apple":
                signal = nanopore_process_signal(signal_raw, strategy="apple")
                if signal.size == 0:
                    continue
            elif args.normalize_mode == "stone":
                signal = nanopore_process_signal(signal_raw, strategy="stone")
                if signal.size == 0:
                    continue
            elif args.normalize_mode == "lemon":
                signal = nanopore_process_signal(signal_raw, strategy="lemon")
                if signal.size == 0:
                    continue
            elif args.normalize_mode == "none":
                signal = signal_raw
            elif args.normalize_mode == "mongo":
                signal = nanopore_process_signal(signal_raw, strategy="mongo")
                if signal.size == 0:
                    continue
            elif args.normalize_mode == "mongo_q30":
                signal = nanopore_process_signal(signal_raw, strategy="mongo")
                
                if signal.size == 0:
                    continue
                # 如果存在任何超出 [-3, 3] 的值，就跳过
                if (signal < -3).any() or (signal > 3).any():
                    continue
            
            else:
                raise ValueError(f"Unknown normalize_mode: {args.normalize_mode}")

            mv_arr = np.asarray(mv, dtype=np.int32)

            vals, counts = np.unique(mv_arr, return_counts=True)
            for v, c in zip(vals, counts):
                mv_value_counter[int(v)] += int(c)

            n_steps = int(len(mv_arr))
            if n_steps > 0:
                ratios_signal_steps.append(float(len(signal)) / float(n_steps))

            step_base = step_base_from_mv(mv_arr)
            max_base_in_mv = int(step_base.max()) if step_base.size else -1

            seq_len_list.append(len(seq_norm))
            max_base_in_mv_list.append(max_base_in_mv)

            pat_len = len(pattern_to_search)
            for start in occ_starts:
                b0 = int(start)
                b1 = int(start + pat_len)

                if max_base_in_mv >= 0 and b1 > (max_base_in_mv + 1):
                    seq_out_of_mv_cnt += 1
                    continue

                s0, s1 = base_to_step_bounds(step_base, b0, b1)
                if s0 is None:
                    continue

                pat_from = int(s0 * args.stride)
                pat_to = int(s1 * args.stride)

                pat_from = max(0, min(pat_from, len(signal)))
                pat_to = max(0, min(pat_to, len(signal)))
                if pat_from >= pat_to:
                    continue

                win_from = max(0, pat_from - args.pad_samples)
                win_to = min(len(signal), pat_to + args.pad_samples)
                if win_from >= win_to:
                    continue

                base_sample_spans_rel, base_abs_spans, ok_any = build_base_sample_spans_rel(
                    step_base=step_base,
                    occ_start_base=b0,
                    pat_len=pat_len,
                    win_from=win_from,
                    win_to=win_to,
                    mv_stride=args.stride,
                    signal_len=len(signal),
                )
                if not ok_any:
                    continue

                for a, b in base_abs_spans:
                    if a < 0 or b <= a:
                        empty_base_span_cnt += 1
                    else:
                        span_len_list.append(int(b - a))

                window_signal = signal[win_from:win_to].astype(np.float32)
                base_spans_out = base_sample_spans_rel

                if args.unify_motif_orientation and is_reverse:
                    win_len = len(window_signal)
                    window_signal = window_signal[::-1].copy()
                    base_spans_out = reverse_spans_rel(base_sample_spans_rel, win_len)

                txt_out.write(
                    f"{rid}\t{strand}\t{b0}\t{b1}\t{pat_from}\t{pat_to}\t{win_from}\t{win_to}\t{pattern_norm}\t{pattern_to_search}\n"
                )

                rec_json = {
                    "read_id": rid,
                    "label": rec.get("label", None),
                    "strand": int(strand),
                    "pattern_query": pattern_norm,
                    "pattern_matched": pattern_to_search,
                    "orientation_unified": bool(args.unify_motif_orientation),
                    "occ_start_base": b0,
                    "occ_end_base": b1,
                    "mv_stride": args.stride,
                    "pad_samples": args.pad_samples,
                    "normalize_mode": args.normalize_mode,
                    "pat_from": pat_from,
                    "pat_to": pat_to,
                    "win_from": win_from,
                    "win_to": win_to,
                    "base_sample_spans_rel": base_spans_out,
                }

                if args.write_moves:
                    rec_json["moves"] = mv

                if args.write_seq:
                    rec_json["seq"] = seq_norm

                if args.write_window_signal:
                    rec_json["signal"] = window_signal.tolist()

                if args.write_debug_meta:
                    rec_json["signal_full_len"] = int(len(signal))
                    rec_json["window_signal_len"] = int(len(window_signal))
                    rec_json["n_steps"] = int(n_steps)
                    rec_json["max_base_in_mv"] = int(max_base_in_mv)
                    rec_json["seq_len"] = int(len(seq_norm))
                    rec_json["signal_key"] = key
                    rec_json["is_reverse_match"] = bool(is_reverse)

                jsonl_out.write(json.dumps(rec_json, ensure_ascii=False) + "\n")

                if strand == 0:
                    forward_match_cnt += 1
                else:
                    reverse_match_cnt += 1

                written_occurrences += 1

                if not args.no_plot:
                    batch.append({
                        "rid": f"{rid} | occ={b0}",
                        "window_signal": window_signal,
                        "base_spans_rel": base_spans_out,
                        "strand": strand,
                        "pattern_matched": pattern_to_search,
                    })
                    if len(batch) >= args.batch_size:
                        batch_idx += 1
                        out_png = save_batch_png_subplots(
                            batch, batch_idx, args.out_dir, pattern_norm, args
                        )
                        print(f"[PNG] wrote: {out_png}")
                        batch = []

                if written_occurrences >= args.max_occurrences:
                    break

            if written_occurrences >= args.max_occurrences:
                break

    if (not args.no_plot) and batch:
        batch_idx += 1
        out_png = save_batch_png_subplots(batch, batch_idx, args.out_dir, pattern_norm, args)
        print(f"[PNG] wrote: {out_png}")

    txt_out.close()
    jsonl_out.close()

    qc = {
        "pattern_input": args.pattern,
        "pattern_normalized": pattern_norm,
        "pattern_reverse_complement": pattern_rc,
        "matching_mode": "ATGCN",
        "orientation_unified": bool(args.unify_motif_orientation),
        "scanned_lines": int(scanned_lines),
        "written_occurrences": int(written_occurrences),
        "forward_match_cnt": int(forward_match_cnt),
        "reverse_match_cnt": int(reverse_match_cnt),
        "missing_signal_cnt": int(missing_signal_cnt),
        "seq_out_of_mv_cnt": int(seq_out_of_mv_cnt),
        "moves_value_counts": {str(k): int(v) for k, v in sorted(mv_value_counter.items())},
        "signal_len_div_steps_percentiles": safe_percentiles(ratios_signal_steps),
        "seq_len_percentiles": safe_percentiles(seq_len_list),
        "max_base_in_mv_percentiles": safe_percentiles(max_base_in_mv_list),
        "base_span_len_samples_percentiles": safe_percentiles(span_len_list),
        "empty_or_invalid_base_span_count": int(empty_base_span_cnt),
        "notes": {
            "stride_check": "If signal_len/steps is near your STRIDE (e.g., ~5), stride is likely correct.",
            "offset_check": "If seq_out_of_mv_cnt is high, seq base indices may not align with moves base indices (offset).",
            "matching_rule": "Pattern supports only A/T/G/C/N, where N matches any A/T/G/C.",
            "orientation_rule": "For reverse-strand matches, output window signal and per-base spans are reversed to the same motif orientation as forward strand.",
        }
    }

    qc_path = os.path.join(args.out_dir, "qc_summary.json")
    with open(qc_path, "w", encoding="utf-8") as fw:
        json.dump(qc, fw, ensure_ascii=False, indent=2)

    print(f"[DONE] scanned lines: {scanned_lines}")
    print(f"[DONE] written occurrences (up to {args.max_occurrences}): {written_occurrences}")
    print(f"[DONE] forward matches: {forward_match_cnt}")
    print(f"[DONE] reverse matches: {reverse_match_cnt}")
    print(f"[DONE] PNG dir: {args.out_dir} (batches: {batch_idx})")
    print(f"[DONE] result TXT: {args.out_txt}")
    print(f"[DONE] result JSONL: {args.out_jsonl}")
    print(f"[DONE] QC summary: {qc_path}")
    print("[NOTE] Output window signal and base_sample_spans_rel are already orientation-unified when --unify_motif_orientation is enabled.")


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Step1: Extract motif windows from nanopore signals using movetable, "
            "write window signal + per-base spans for later token slicing. "
            "Pattern supports only A/T/G/C/N. Reverse-strand matches can be "
            "re-oriented to the same motif direction as forward-strand matches."
        )
    )

    ap.add_argument("--jsonl_path", required=True, help="movetable jsonl, containing seq/moves/align_strand/id")
    ap.add_argument("--h5_path", required=True, help="signal.h5 with group /signals/<key>")

    ap.add_argument("--pattern", default="ATAACAGGT", help="motif sequence to search in seq; supports only A/T/G/C/N")
    ap.add_argument("--stride", type=int, default=5, help="raw samples per move step, often 5")
    ap.add_argument("--pad_samples", type=int, default=0, help="expand motif signal window on both sides")

    ap.add_argument("--max_occurrences", type=int, default=1000, help="maximum motif occurrences to write/plot")
    ap.add_argument("--batch_size", type=int, default=100, help="number of subplots per PNG")

    ap.add_argument("--out_dir", required=True, help="directory to save PNG and qc_summary.json")
    ap.add_argument("--out_txt", required=True, help="TSV output path")
    ap.add_argument("--out_jsonl", required=True, help="JSONL output path")

    ap.add_argument(
        "--normalize_mode",
        default="none",
        choices=["apple", "stone", "none", "lemon", "mongo", "mongo_q30"],
        help="signal normalization mode"
    )
    ap.add_argument("--write_window_signal", action="store_true", help="write orientation-unified window signal into output JSONL")
    ap.add_argument("--write_moves", action="store_true", help="write full moves into output JSONL")
    ap.add_argument("--write_seq", action="store_true", help="write full normalized seq into output JSONL")
    ap.add_argument("--write_debug_meta", action="store_true", help="write extra debug metadata")

    ap.add_argument(
        "--unify_motif_orientation",
        action="store_true",
        help="for reverse-strand matches, reverse window signal and per-base spans so all outputs are in the same motif orientation"
    )

    ap.add_argument("--no_plot", action="store_true", help="disable PNG plotting")
    ap.add_argument("--downsample_plot", type=int, default=1, help="downsample factor for plotting only")
    ap.add_argument("--linewidth", type=float, default=0.8, help="signal line width")
    ap.add_argument("--fig_width_inch", type=float, default=18, help="figure width")
    ap.add_argument("--row_height_inch", type=float, default=1.0, help="height per subplot row")
    ap.add_argument("--dpi", type=int, default=120, help="PNG dpi")

    ap.add_argument("--share_ylim", action="store_true", help="share y-limits across plots in a batch")
    ap.add_argument("--y_lim_mode", default="manual", choices=["auto", "batch", "manual"], help="y-axis mode")
    ap.add_argument("--y_lim", type=float, nargs=2, default=(-3, 3), help="manual y-limit: min max")

    ap.add_argument(
        "--mark_base_pos",
        type=int,
        nargs="*",
        default=[4],
        help="highlight multiple base positions, 0-based, e.g. --mark_base_pos 2 4 7"
    )

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
