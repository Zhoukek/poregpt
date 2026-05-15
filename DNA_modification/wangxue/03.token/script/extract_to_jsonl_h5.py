#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gzip
import json
import os
import re

import numpy as np
import pysam
import h5py
import pyccf5 as slow5


# ---------- ID 归一化：去掉 _<int>_<float> 后缀 ----------
def canonicalize_read_id(qname: str) -> str:
    parts = qname.split("_")
    if len(parts) >= 3 and re.fullmatch(r"\d+", parts[-2]) and re.fullmatch(r"\d+(\.\d+)?", parts[-1]):
        return "_".join(parts[:-2])
    return qname


# ---------- FASTQ 目录读取 ----------
def open_fastq(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r", encoding="utf-8", errors="ignore")


def collect_fastq_files(fastq_dir):
    files = []
    for root, _, fns in os.walk(fastq_dir):
        for fn in fns:
            if fn.endswith((".fastq", ".fq", ".fastq.gz", ".fq.gz")):
                files.append(os.path.join(root, fn))
    return sorted(files)


def load_fastq_seqs(fastq_dir, wanted_ids):
    """
    只提取 wanted_ids 的 seq，避免全量 FASTQ 占内存
    """
    seq_map = {}
    files = collect_fastq_files(fastq_dir)
    print(f"[FASTQ] {len(files)} files")

    for fpath in files:
        print(f"[FASTQ] reading: {fpath}")
        with open_fastq(fpath) as f:
            while True:
                h = f.readline()
                if not h:
                    break
                seq = f.readline().strip()
                f.readline()
                qual = f.readline()
                if not qual:
                    break

                if not h.startswith("@"):
                    continue

                qname = h[1:].split()[0]
                cid = canonicalize_read_id(qname)
                if cid in wanted_ids and cid not in seq_map:
                    seq_map[cid] = seq

    print(f"[FASTQ] extracted {len(seq_map)}/{len(wanted_ids)}")
    return seq_map


# ---------- CCF5 signal 标准化：按你给的公式 ----------
def normalize_signal(read) -> np.ndarray:
    if "lvdsmid" in read:
        signal = (read["signal"] - read["lvdsmid"]) * read["unit"]
    else:
        signal = read["K"] * read["scale"] * (read["signal"].astype(np.uint16) + read["offset"]) + read["B"]
    return np.asarray(signal, dtype=np.float32)


def collect_ccf5_files(ccf_dir):
    files = []
    for root, _, fns in os.walk(ccf_dir):
        for fn in fns:
            if fn.endswith(".ccf5"):
                files.append(os.path.join(root, fn))
    return sorted(files)


def write_signals_to_h5(ccf_dir, wanted_ids, h5_path, batchsize=512, threads=1, compression="lzf"):
    """
    将 wanted_ids 的 signal 写入 signals.h5 的 /signals/<read_id>
    """
    os.makedirs(os.path.dirname(h5_path) or ".", exist_ok=True)
    ccf_files = collect_ccf5_files(ccf_dir)
    print(f"[CCF5] {len(ccf_files)} files")

    with h5py.File(h5_path, "a") as h5:
        grp = h5.require_group("signals")

        saved = 0
        for cpath in ccf_files:
            print(f"[CCF5] reading: {cpath}")
            s5 = slow5.Open(cpath, "r", DEBUG=0)
            all_ids = s5.get_read_ids()[0]

            # 只取本文件中需要且尚未写入的
            need = [rid for rid in all_ids if (rid in wanted_ids and rid not in grp)]
            if not need:
                try:
                    s5.close()
                except Exception:
                    pass
                continue

            reads = s5.get_read_list_multi(need, threads=threads, batchsize=batchsize, aux="all")
            for r in reads:
                rid = r["read_id"]
                if rid not in wanted_ids or rid in grp:
                    continue

                sig = normalize_signal(r)

                grp.create_dataset(
                    rid,
                    data=sig,
                    dtype=np.float32,
                    compression=compression,   # "lzf" 快；或 "gzip" 更省空间
                )
                saved += 1

            try:
                s5.close()
            except Exception:
                pass

            print(f"[CCF5] saved so far: {saved}/{len(wanted_ids)}")

    print(f"[CCF5] done. signals stored in: {h5_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bam", required=True)
    ap.add_argument("--fastq_dir", required=True)
    ap.add_argument("--ccf_dir", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--out_h5", required=True, help="signals.h5")
    ap.add_argument("--batchsize", type=int, default=512)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--compression", default="lzf", choices=["lzf", "gzip", None])
    args = ap.parse_args()

    # 1) 从 BAM 收集 wanted ids（只要 flag 0/16）
    wanted = set()
    bam = pysam.AlignmentFile(args.bam, "rb")
    for aln in bam.fetch(until_eof=True):
        if int(aln.flag) in (0, 16):
            wanted.add(canonicalize_read_id(aln.query_name))
    bam.close()
    print(f"[BAM] wanted reads (flag 0/16): {len(wanted)}")

    # 2) FASTQ 提 seq（只提 wanted）
    seq_map = load_fastq_seqs(args.fastq_dir, wanted)

    # 3) CCF5 写入 signals.h5
    comp = args.compression if args.compression != "None" else None
    write_signals_to_h5(
        args.ccf_dir,
        wanted,
        args.out_h5,
        batchsize=args.batchsize,
        threads=args.threads,
        compression=comp,
    )

    # 4) 再扫 BAM 输出 JSONL（signal 不再写 path，而是写 signal_h5 + signal_key）
    bam = pysam.AlignmentFile(args.bam, "rb")
    with open(args.out_jsonl, "w", encoding="utf-8") as out:
        for aln in bam.fetch(until_eof=True):
            if int(aln.flag) not in (0, 16):
                continue

            cid = canonicalize_read_id(aln.query_name)
            tags = dict(aln.tags)

            rec = {
                "id": cid,
                "align_strand": int(aln.flag),
                "moves": list(tags["MV"]) if "MV" in tags else None,
                "CS": tags.get("CS"),
                "seq": seq_map.get(cid),

                # ✅ 关键：统一指向一个 HDF5 + key
                "signal_h5": args.out_h5,
                "signal_key": cid,
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    bam.close()
    print(f"[DONE] jsonl: {args.out_jsonl}")
    print(f"[DONE] h5:    {args.out_h5}")


if __name__ == "__main__":
    main()