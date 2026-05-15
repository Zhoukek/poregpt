#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def nanopore_normalize(signal: np.ndarray) -> np.ndarray:
    if signal.size == 0:
        return np.array([], dtype=np.float32)
    med = np.median(signal)
    mad = 1.4826 * np.median(np.abs(signal - med))
    mad = max(mad, 1.0)
    return ((signal - med) / mad).astype(np.float32)


def load_jsonl_record_by_read_id(jsonl_path, read_id):
    with open(jsonl_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("read_id") == read_id:
                return rec
    raise ValueError(f"read_id not found in jsonl: {read_id}")


def reconstruct_read_level_signal_from_chunks(length, chunk_starts, chunk_values, chunk_len):
    """
    Average overlapping chunk-level values back to read-level coordinates.
    """
    acc = np.zeros(length, dtype=np.float32)
    cnt = np.zeros(length, dtype=np.float32)

    for s, v in zip(chunk_starts, chunk_values):
        e = min(s + chunk_len, length)
        valid_len = e - s
        if valid_len <= 0:
            continue
        acc[s:e] += v[:valid_len]
        cnt[s:e] += 1.0

    out = np.zeros(length, dtype=np.float32)
    mask = cnt > 0
    out[mask] = acc[mask] / cnt[mask]
    return out



def parse_mod_base_indices(mod_bases_str):
    if mod_bases_str is None or mod_bases_str.strip() == "":
        return []
    return [int(x) for x in mod_bases_str.split(",") if x.strip() != ""]


def base_indices_to_sample_spans(base_sample_spans_rel, mod_base_indices):
    """
    Convert modified base indices (relative to base_sample_spans_rel)
    into sample spans.
    """
    spans = []
    n = len(base_sample_spans_rel)
    for b in mod_base_indices:
        if 0 <= b < n:
            s, e = base_sample_spans_rel[b]
            spans.append((int(s), int(e)))
    return spans


def merge_spans(spans):
    if len(spans) == 0:
        return []
    spans = sorted(spans, key=lambda x: x[0])
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def get_states_overlapping_modified_regions(chunk_starts, chunk_states, chunk_len, mod_spans):
    overlap_states = []
    for ms, me in mod_spans:
        for s, st in zip(chunk_starts, chunk_states):
            ce = s + chunk_len
            if not (ce <= ms or s >= me):
                overlap_states.append(int(st))
    return sorted(set(overlap_states))


def save_selected_state_chunk_examples_for_one_read(
    chunk_residuals,
    chunk_states,
    chunk_starts,
    selected_states,
    out_dir,
    read_id,
    n_examples=6,
    seed=42
):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    for st in selected_states:
        idx = np.where(chunk_states == st)[0]
        if len(idx) == 0:
            continue

        chosen = idx if len(idx) <= n_examples else rng.choice(idx, size=n_examples, replace=False)

        plt.figure(figsize=(8, 4))
        for j in chosen:
            start = int(chunk_starts[j])
            plt.plot(chunk_residuals[j], alpha=0.8, linewidth=1, label=f"start={start}")

        plt.xlabel("position in chunk")
        plt.ylabel("residual")
        plt.title(f"{read_id} | state {st} | n_show={len(chosen)}")
        if len(chosen) <= 10:
            plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{read_id}.state_{st}.chunk_examples.png"), dpi=200)
        plt.close()


def save_state_summary_for_mod_regions(chunk_starts, state_ids, chunk_len, mod_spans, out_txt):
    """
    Summarize which chunk states overlap modified spans.
    """
    lines = []
    lines.append("[Modified region state summary]")

    all_overlap_states = []

    for i, (ms, me) in enumerate(mod_spans):
        overlap = []
        for s, st in zip(chunk_starts, state_ids):
            ce = s + chunk_len
            if not (ce <= ms or s >= me):
                overlap.append(int(st))
                all_overlap_states.append(int(st))

        if len(overlap) == 0:
            lines.append(f"region_{i}\t{ms}-{me}\tno_overlapping_chunks")
        else:
            uniq, cnt = np.unique(overlap, return_counts=True)
            summary = ",".join([f"{u}:{c}" for u, c in zip(uniq.tolist(), cnt.tolist())])
            lines.append(f"region_{i}\t{ms}-{me}\t{summary}")

    lines.append("")
    lines.append("[Overall states overlapping modified regions]")
    if len(all_overlap_states) > 0:
        uniq, cnt = np.unique(all_overlap_states, return_counts=True)
        summary = ",".join([f"{u}:{c}" for u, c in zip(uniq.tolist(), cnt.tolist())])
        lines.append(summary)
    else:
        lines.append("none")

    with open(out_txt, "w") as f:
        for line in lines:
            f.write(line + "\n")


def plot_read_visualization(
    read_id,
    signal_norm,
    residual_read,
    chunk_starts,
    state_ids,
    chunk_len,
    mod_spans,
    out_png,
):
    length = len(signal_norm)
    x = np.arange(length)

    uniq_states = sorted(np.unique(state_ids).tolist()) if len(state_ids) > 0 else []
    cmap = plt.get_cmap("tab20")
    state_to_color = {s: cmap(i % 20) for i, s in enumerate(uniq_states)}

    fig, axes = plt.subplots(
        3, 1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.5, 2.5, 1.2]}
    )

    # 1) signal
    axes[0].plot(x, signal_norm, linewidth=1)
    axes[0].set_title(f"{read_id} | normalized signal")
    axes[0].set_ylabel("signal")
    for s, e in mod_spans:
        axes[0].axvspan(s, e, color="red", alpha=0.18)

    # 2) residual
    axes[1].plot(x, residual_read, linewidth=1)
    axes[1].set_title("read-level residual (averaged from chunk residuals)")
    axes[1].set_ylabel("residual")
    for s, e in mod_spans:
        axes[1].axvspan(s, e, color="red", alpha=0.18)

    # 3) state track
    axes[2].set_title("chunk states")
    axes[2].set_ylabel("state")
    axes[2].set_xlabel("sample index")

    for s, st in zip(chunk_starts, state_ids):
        rect = Rectangle(
            (s, 0), chunk_len, 1,
            facecolor=state_to_color.get(int(st), "gray"),
            edgecolor=None,
            alpha=0.85
        )
        axes[2].add_patch(rect)

    for s, e in mod_spans:
        axes[2].axvspan(s, e, color="red", alpha=0.18)

    axes[2].set_xlim(0, length)
    axes[2].set_ylim(0, 1)

    if len(uniq_states) > 0:
        handles = []
        labels = []
        max_show = min(len(uniq_states), 12)
        for st in uniq_states[:max_show]:
            handles.append(Rectangle((0, 0), 1, 1, facecolor=state_to_color[st]))
            labels.append(f"state {st}")
        axes[2].legend(handles, labels, ncol=min(6, max_show), loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Visualize one synthetic read: signal, residual, modified spans, and states"
    )
    parser.add_argument("--jsonl", required=True, help="Original synthetic jsonl")
    parser.add_argument("--chunk_meta", required=True, help="chunk_meta.npz from build_chunks_from_jsonl.py")
    parser.add_argument("--residual", required=True, help="synthetic_model1_residual.npy")
    parser.add_argument("--state_id", required=True, help="synthetic_model2_state_id.npy")
    parser.add_argument("--read_id", required=True, help="Target read_id to visualize")
    parser.add_argument(
        "--mod_bases",
        default="",
        help="Comma-separated modified base indices relative to base_sample_spans_rel, e.g. 10,11,12,35"
    )
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--n_examples", type=int, default=6, help="Examples to draw per selected state")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # -----------------------------
    # Load original read record
    # -----------------------------
    rec = load_jsonl_record_by_read_id(args.jsonl, args.read_id)
    signal = np.asarray(rec["signal"], dtype=np.float32)
    signal_norm = nanopore_normalize(signal)
    base_sample_spans_rel = rec.get("base_sample_spans_rel", [])

    # -----------------------------
    # Modified base indices -> sample spans
    # -----------------------------
    mod_base_indices = parse_mod_base_indices(args.mod_bases)
    mod_spans = base_indices_to_sample_spans(base_sample_spans_rel, mod_base_indices)
    mod_spans = merge_spans(mod_spans)

    # -----------------------------
    # Load chunk meta / residual / states
    # -----------------------------
    meta = np.load(args.chunk_meta, allow_pickle=True)
    read_ids = meta["read_ids"]
    chunk_starts = meta["chunk_starts"]
    chunk_len = int(meta["chunk_len"][0])

    residual = np.load(args.residual).astype(np.float32)
    state_id = np.load(args.state_id).astype(np.int32)

    if not (len(read_ids) == len(chunk_starts) == len(residual) == len(state_id)):
        raise ValueError("chunk_meta / residual / state_id length mismatch")

    mask = (read_ids == args.read_id)
    if mask.sum() == 0:
        raise ValueError(f"No chunks found for read_id in chunk_meta: {args.read_id}")

    read_chunk_starts = chunk_starts[mask].astype(np.int64)
    read_chunk_residual = residual[mask]
    read_chunk_states = state_id[mask]

    # sort by start
    order = np.argsort(read_chunk_starts)
    read_chunk_starts = read_chunk_starts[order]
    read_chunk_residual = read_chunk_residual[order]
    read_chunk_states = read_chunk_states[order]

    # -----------------------------
    # Reconstruct read-level residual
    # -----------------------------
    residual_read = reconstruct_read_level_signal_from_chunks(
        length=len(signal_norm),
        chunk_starts=read_chunk_starts,
        chunk_values=read_chunk_residual,
        chunk_len=chunk_len
    )

    # -----------------------------
    # Main visualization
    # -----------------------------
    out_png = os.path.join(args.out_dir, f"{args.read_id}.signal_residual_state.png")
    plot_read_visualization(
        read_id=args.read_id,
        signal_norm=signal_norm,
        residual_read=residual_read,
        chunk_starts=read_chunk_starts,
        state_ids=read_chunk_states,
        chunk_len=chunk_len,
        mod_spans=mod_spans,
        out_png=out_png,
    )

    # -----------------------------
    # State summary in modified regions
    # -----------------------------
    out_txt = os.path.join(args.out_dir, f"{args.read_id}.modified_region_states.txt")
    save_state_summary_for_mod_regions(
        chunk_starts=read_chunk_starts,
        state_ids=read_chunk_states,
        chunk_len=chunk_len,
        mod_spans=mod_spans,
        out_txt=out_txt
    )

    # -----------------------------
    # Per-state chunk examples
    # -----------------------------
    selected_states = get_states_overlapping_modified_regions(
        chunk_starts=read_chunk_starts,
        chunk_states=read_chunk_states,
        chunk_len=chunk_len,
        mod_spans=mod_spans
    )

    save_selected_state_chunk_examples_for_one_read(
        chunk_residuals=read_chunk_residual,
        chunk_states=read_chunk_states,
        chunk_starts=read_chunk_starts,
        selected_states=selected_states,
        out_dir=os.path.join(args.out_dir, "per_state_chunk_examples"),
        read_id=args.read_id,
        n_examples=args.n_examples,
        seed=42
    )

    # -----------------------------
    # Small summary json
    # -----------------------------
    summary = {
        "read_id": args.read_id,
        "signal_length": int(len(signal_norm)),
        "num_chunks_for_read": int(len(read_chunk_starts)),
        "chunk_len": int(chunk_len),
        "mod_base_indices": mod_base_indices,
        "mod_spans": mod_spans,
        "selected_states": selected_states,
        "output_png": out_png,
        "output_txt": out_txt,
    }
    with open(os.path.join(args.out_dir, f"{args.read_id}.summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("[DONE]")
    print("saved:")
    print(" ", out_png)
    print(" ", out_txt)
    print(" ", os.path.join(args.out_dir, "per_state_chunk_examples"))


if __name__ == "__main__":
    main()
