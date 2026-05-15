#!/usr/bin/env python3
import argparse
import json
import csv
import os
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np


VALID = {"A", "T", "G", "C", "N"}


def normalize_seq(seq: str) -> str:
    seq = seq.upper().replace("U", "T")
    return "".join([c for c in seq if c in VALID])


def needleman_wunsch(ref, query, match_score=2, mismatch_score=-1, gap_score=-2):
    """
    Global alignment.
    Returns aligned_ref, aligned_query
    """
    n = len(ref)
    m = len(query)

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    tb = [[None] * (m + 1) for _ in range(n + 1)]  # D/U/L

    for i in range(1, n + 1):
        dp[i][0] = i * gap_score
        tb[i][0] = "U"
    for j in range(1, m + 1):
        dp[0][j] = j * gap_score
        tb[0][j] = "L"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = dp[i - 1][j - 1] + (match_score if ref[i - 1] == query[j - 1] else mismatch_score)
            up = dp[i - 1][j] + gap_score
            left = dp[i][j - 1] + gap_score

            best = diag
            direction = "D"
            if up > best:
                best = up
                direction = "U"
            if left > best:
                best = left
                direction = "L"

            dp[i][j] = best
            tb[i][j] = direction

    i, j = n, m
    aln_ref = []
    aln_query = []

    while i > 0 or j > 0:
        direction = tb[i][j]
        if direction == "D":
            aln_ref.append(ref[i - 1])
            aln_query.append(query[j - 1])
            i -= 1
            j -= 1
        elif direction == "U":
            aln_ref.append(ref[i - 1])
            aln_query.append("-")
            i -= 1
        elif direction == "L":
            aln_ref.append("-")
            aln_query.append(query[j - 1])
            j -= 1
        else:
            break

    return "".join(reversed(aln_ref)), "".join(reversed(aln_query))


def parse_alignment(aln_ref, aln_query):
    """
    Returns:
      summary dict
      per-position event arrays indexed by reference 1..len(ref)
    """
    matches = 0
    substitutions = 0
    insertions = 0
    deletions = 0

    ref_pos = 0
    events = []

    per_pos_sub = []
    per_pos_del = []
    per_pos_match = []

    i = 0
    L = len(aln_ref)
    while i < L:
        r = aln_ref[i]
        q = aln_query[i]

        if r != "-":
            ref_pos += 1

        if r != "-" and q != "-" and r == q:
            matches += 1
            per_pos_match.append((ref_pos, r, q))
            i += 1
            continue

        if r != "-" and q != "-" and r != q:
            substitutions += 1
            per_pos_sub.append((ref_pos, r, q))
            events.append({
                "type": "substitution",
                "ref_pos": ref_pos,
                "ref_base": r,
                "read_base": q
            })
            i += 1
            continue

        if r != "-" and q == "-":
            start = ref_pos
            deleted = [r]
            deletions += 1
            per_pos_del.append((ref_pos, r))
            i += 1
            while i < L and aln_ref[i] != "-" and aln_query[i] == "-":
                ref_pos += 1
                deleted.append(aln_ref[i])
                deletions += 1
                per_pos_del.append((ref_pos, aln_ref[i]))
                i += 1
            events.append({
                "type": "deletion",
                "ref_start": start,
                "ref_end": start + len(deleted) - 1,
                "ref_seq": "".join(deleted),
                "length": len(deleted)
            })
            continue

        if r == "-" and q != "-":
            ins_seq = [q]
            insertions += 1
            anchor = ref_pos
            i += 1
            while i < L and aln_ref[i] == "-" and aln_query[i] != "-":
                ins_seq.append(aln_query[i])
                insertions += 1
                i += 1
            events.append({
                "type": "insertion",
                "ref_anchor": anchor,
                "read_seq": "".join(ins_seq),
                "length": len(ins_seq)
            })
            continue

        i += 1

    return {
        "matches": matches,
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions,
        "alignment_length": len(aln_ref),
        "events": events,
        "per_pos_sub": per_pos_sub,
        "per_pos_del": per_pos_del,
        "per_pos_match": per_pos_match,
    }


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_hist(data, xlabel, title, out_png, bins=30):
    plt.figure(figsize=(8, 5), dpi=150)
    plt.hist(data, bins=bins)
    plt.xlabel(xlabel)
    plt.ylabel("Read count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def save_bar(x, y, xlabel, ylabel, title, out_png):
    plt.figure(figsize=(12, 4), dpi=150)
    plt.bar(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def save_heatmap(matrix, xlabels, ylabels, title, out_png):
    plt.figure(figsize=(6, 5), dpi=150)
    plt.imshow(matrix, aspect="auto")
    plt.xticks(range(len(xlabels)), xlabels)
    plt.yticks(range(len(ylabels)), ylabels)
    plt.xlabel("Read base")
    plt.ylabel("Reference base")
    plt.title(title)
    plt.colorbar(label="Count")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Compare result.jsonl middle_seq to a reference and generate plots.")
    ap.add_argument("--input_jsonl", required=True, help="result.jsonl from extraction step")
    ap.add_argument("--reference_seq", required=True, help="expected reference sequence, e.g. 137 bp")
    ap.add_argument("--out_dir", required=True, help="output directory")
    ap.add_argument("--min_len", type=int, default=120, help="minimum middle_seq length to keep")
    ap.add_argument("--max_len", type=int, default=150, help="maximum middle_seq length to keep")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    ref = normalize_seq(args.reference_seq)
    if len(ref) == 0:
        raise ValueError("reference_seq is empty after normalization")

    ref_len = len(ref)

    read_lengths = []
    identities_by_ref = []
    identities_by_alignment = []
    sub_counts = []
    ins_counts = []
    del_counts = []
    total_error_counts = []

    # per-position counts on reference
    pos_coverage = np.zeros(ref_len, dtype=np.int64)
    pos_sub = np.zeros(ref_len, dtype=np.int64)
    pos_del = np.zeros(ref_len, dtype=np.int64)
    pos_match = np.zeros(ref_len, dtype=np.int64)

    # substitutions matrix
    bases = ["A", "C", "G", "T", "N"]
    base_to_idx = {b: i for i, b in enumerate(bases)}
    sub_matrix = np.zeros((len(bases), len(bases)), dtype=np.int64)

    # insertion anchored after reference positions 0..ref_len
    insertion_anchor_counts = np.zeros(ref_len + 1, dtype=np.int64)

    per_read_tsv = os.path.join(args.out_dir, "per_read_summary.tsv")
    detailed_jsonl = os.path.join(args.out_dir, "per_read_events.jsonl")
    summary_txt = os.path.join(args.out_dir, "overview_summary.txt")

    total_reads = 0
    kept_reads = 0
    exact_match_reads = 0

    with open(args.input_jsonl, "r", encoding="utf-8") as fr, \
         open(per_read_tsv, "w", encoding="utf-8", newline="") as fw_tsv, \
         open(detailed_jsonl, "w", encoding="utf-8") as fw_jsonl:

        writer = csv.writer(fw_tsv, delimiter="\t")
        writer.writerow([
            "read_id",
            "ref_len",
            "read_len",
            "matches",
            "substitutions",
            "insertions",
            "deletions",
            "alignment_length",
            "identity_by_alignment",
            "identity_by_ref",
            "n_events",
            "middle_seq",
        ])

        for line in fr:
            total_reads += 1
            rec = json.loads(line)
            read_id = rec.get("read_id", "")
            middle_seq = normalize_seq(rec.get("middle_seq", ""))

            if len(middle_seq) == 0:
                continue
            if len(middle_seq) < args.min_len or len(middle_seq) > args.max_len:
                continue

            kept_reads += 1
            read_lengths.append(len(middle_seq))

            aln_ref, aln_read = needleman_wunsch(ref, middle_seq)
            parsed = parse_alignment(aln_ref, aln_read)

            identity_by_alignment = parsed["matches"] / parsed["alignment_length"] if parsed["alignment_length"] else 0.0
            identity_by_ref = parsed["matches"] / ref_len if ref_len else 0.0

            identities_by_alignment.append(identity_by_alignment)
            identities_by_ref.append(identity_by_ref)
            sub_counts.append(parsed["substitutions"])
            ins_counts.append(parsed["insertions"])
            del_counts.append(parsed["deletions"])
            total_error_counts.append(parsed["substitutions"] + parsed["insertions"] + parsed["deletions"])

            if parsed["substitutions"] == 0 and parsed["insertions"] == 0 and parsed["deletions"] == 0 and len(middle_seq) == ref_len:
                exact_match_reads += 1

            # coverage / per-position stats
            for pos, r, q in parsed["per_pos_match"]:
                idx = pos - 1
                pos_coverage[idx] += 1
                pos_match[idx] += 1

            for pos, r, q in parsed["per_pos_sub"]:
                idx = pos - 1
                pos_coverage[idx] += 1
                pos_sub[idx] += 1
                if r in base_to_idx and q in base_to_idx:
                    sub_matrix[base_to_idx[r], base_to_idx[q]] += 1

            for pos, r in parsed["per_pos_del"]:
                idx = pos - 1
                pos_coverage[idx] += 1
                pos_del[idx] += 1

            for ev in parsed["events"]:
                if ev["type"] == "insertion":
                    anchor = ev["ref_anchor"]
                    if 0 <= anchor <= ref_len:
                        insertion_anchor_counts[anchor] += ev["length"]

            writer.writerow([
                read_id,
                ref_len,
                len(middle_seq),
                parsed["matches"],
                parsed["substitutions"],
                parsed["insertions"],
                parsed["deletions"],
                parsed["alignment_length"],
                f"{identity_by_alignment:.6f}",
                f"{identity_by_ref:.6f}",
                len(parsed["events"]),
                middle_seq,
            ])

            out_rec = {
                "read_id": read_id,
                "middle_seq": middle_seq,
                "aligned_ref": aln_ref,
                "aligned_read": aln_read,
                "matches": parsed["matches"],
                "substitutions": parsed["substitutions"],
                "insertions": parsed["insertions"],
                "deletions": parsed["deletions"],
                "alignment_length": parsed["alignment_length"],
                "identity_by_alignment": identity_by_alignment,
                "identity_by_ref": identity_by_ref,
                "events": parsed["events"],
            }
            fw_jsonl.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    # avoid division by zero
    pos_sub_rate = np.divide(pos_sub, pos_coverage, out=np.zeros_like(pos_sub, dtype=float), where=pos_coverage > 0)
    pos_del_rate = np.divide(pos_del, pos_coverage, out=np.zeros_like(pos_del, dtype=float), where=pos_coverage > 0)
    pos_total_error_rate = np.divide(pos_sub + pos_del, pos_coverage, out=np.zeros_like(pos_sub, dtype=float), where=pos_coverage > 0)

    # plots
    if len(read_lengths) > 0:
        save_hist(
            read_lengths,
            xlabel="Middle sequence length (bp)",
            title="Read length distribution",
            out_png=os.path.join(args.out_dir, "read_length_hist.png"),
            bins=min(50, max(10, len(set(read_lengths))))
        )

        save_hist(
            identities_by_ref,
            xlabel="Identity by reference",
            title="Identity distribution",
            out_png=os.path.join(args.out_dir, "identity_hist.png"),
            bins=30
        )

        plt.figure(figsize=(9, 5), dpi=150)
        plt.hist(sub_counts, bins=30, alpha=0.7, label="Substitutions")
        plt.hist(ins_counts, bins=30, alpha=0.7, label="Insertions")
        plt.hist(del_counts, bins=30, alpha=0.7, label="Deletions")
        plt.xlabel("Count per read")
        plt.ylabel("Read count")
        plt.title("Per-read error counts")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "error_count_hist.png"))
        plt.close()

        positions = np.arange(1, ref_len + 1)

        save_bar(
            positions,
            pos_sub_rate,
            xlabel="Reference position",
            ylabel="Substitution rate",
            title="Per-position substitution rate",
            out_png=os.path.join(args.out_dir, "per_position_sub_rate.png")
        )

        save_bar(
            positions,
            pos_del_rate,
            xlabel="Reference position",
            ylabel="Deletion rate",
            title="Per-position deletion rate",
            out_png=os.path.join(args.out_dir, "per_position_del_rate.png")
        )

        save_bar(
            positions,
            pos_total_error_rate,
            xlabel="Reference position",
            ylabel="Total error rate",
            title="Per-position total error rate",
            out_png=os.path.join(args.out_dir, "per_position_total_error_rate.png")
        )

        save_bar(
            np.arange(0, ref_len + 1),
            insertion_anchor_counts,
            xlabel="Reference anchor position",
            ylabel="Inserted bases count",
            title="Insertion counts anchored to reference positions",
            out_png=os.path.join(args.out_dir, "insertion_anchor_counts.png")
        )

        save_heatmap(
            sub_matrix,
            xlabels=bases,
            ylabels=bases,
            title="Substitution heatmap",
            out_png=os.path.join(args.out_dir, "substitution_heatmap.png")
        )

    with open(summary_txt, "w", encoding="utf-8") as fw:
        fw.write(f"reference_length\t{ref_len}\n")
        fw.write(f"total_reads_in_jsonl\t{total_reads}\n")
        fw.write(f"kept_reads_after_length_filter\t{kept_reads}\n")
        fw.write(f"exact_match_reads\t{exact_match_reads}\n")
        fw.write(f"exact_match_fraction\t{(exact_match_reads / kept_reads) if kept_reads else 0.0:.6f}\n")
        fw.write(f"mean_read_length\t{np.mean(read_lengths) if read_lengths else 0.0:.6f}\n")
        fw.write(f"mean_identity_by_ref\t{np.mean(identities_by_ref) if identities_by_ref else 0.0:.6f}\n")
        fw.write(f"mean_substitutions_per_read\t{np.mean(sub_counts) if sub_counts else 0.0:.6f}\n")
        fw.write(f"mean_insertions_per_read\t{np.mean(ins_counts) if ins_counts else 0.0:.6f}\n")
        fw.write(f"mean_deletions_per_read\t{np.mean(del_counts) if del_counts else 0.0:.6f}\n")
        fw.write(f"mean_total_errors_per_read\t{np.mean(total_error_counts) if total_error_counts else 0.0:.6f}\n")

        if len(pos_total_error_rate) > 0:
            top_idx = np.argsort(pos_total_error_rate)[::-1][:10]
            fw.write("top10_error_positions\t" + ",".join([f"{i+1}:{pos_total_error_rate[i]:.6f}" for i in top_idx]) + "\n")

    print(f"[DONE] reference_len={ref_len}")
    print(f"[DONE] total_reads={total_reads}")
    print(f"[DONE] kept_reads={kept_reads}")
    print(f"[DONE] output_dir={args.out_dir}")
    print(f"[DONE] summary={summary_txt}")
    print(f"[DONE] per_read_summary={per_read_tsv}")
    print(f"[DONE] detailed_events={detailed_jsonl}")


if __name__ == "__main__":
    main()