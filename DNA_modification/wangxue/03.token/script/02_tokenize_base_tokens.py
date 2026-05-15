#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
import numpy as np
from tqdm import tqdm
from poregpt.tokenizers import VQETokenizer


def extract_token_ids(token_list):
    """从 <|bwav:1234|> 提取 1234"""
    out = []
    for t in token_list:
        m = re.search(r"\d+", t)
        if m is None:
            raise ValueError(f"Bad token format: {t}")
        out.append(int(m.group()))
    return out


def map_base_to_tokens_stride(base_spans, token_ids, token_stride=5, ensure_non_empty=True):
    """
    用固定 stride 映射 raw sample span -> token span
    base_spans: list of [s,e] (左闭右开), s/e是相对window起点的raw sample index
    """
    token_len = len(token_ids)
    base_token_spans = []
    base_token_ids = []

    for s, e in base_spans:
        s = int(s); e = int(e)

        # floor / ceil
        t0 = s // token_stride
        t1 = (e + token_stride - 1) // token_stride  # ceil(e/stride)

        # clamp
        t0 = max(0, min(t0, token_len))
        t1 = max(0, min(t1, token_len))

        # ensure non-empty
        if ensure_non_empty and t0 == t1 and t0 < token_len:
            t1 = t0 + 1

        base_token_spans.append([t0, t1])
        base_token_ids.append(token_ids[t0:t1])

    return base_token_spans, base_token_ids


def map_base_to_tokens_ratio(base_spans, token_ids, signal_len, ensure_non_empty=True):
    """
    ratio 线性映射（保留用于对照）
    """
    token_len = len(token_ids)
    ratio = token_len / float(signal_len)

    base_token_spans = []
    base_token_ids = []

    for s, e in base_spans:
        s = float(s); e = float(e)

        t0 = int(np.floor(s * ratio))
        t1 = int(np.ceil(e * ratio))

        t0 = max(0, min(t0, token_len))
        t1 = max(0, min(t1, token_len))

        if ensure_non_empty and t0 == t1 and t0 < token_len:
            t1 = t0 + 1

        base_token_spans.append([t0, t1])
        base_token_ids.append(token_ids[t0:t1])

    return base_token_spans, base_token_ids


def main():
    ap = argparse.ArgumentParser(
        description="Tokenize window signal and slice token ids per base using base_sample_spans_rel."
    )
    ap.add_argument("--input_jsonl", required=True, help="Input jsonl with fields: signal, base_sample_spans_rel")
    ap.add_argument("--output_jsonl", required=True, help="Output jsonl (token_ids + base_token_ids)")
    ap.add_argument("--model_ckpt", required=True, help="Tokenizer checkpoint .pth")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device for tokenizer")
    ap.add_argument("--token_batch_size", type=int, default=100, help="Tokenizer internal batch size")

    ap.add_argument("--map_mode", default="stride", choices=["stride", "ratio"],
                    help="Mapping from raw sample span to token span")
    ap.add_argument("--token_stride", type=int, default=5,
                    help="Only used when map_mode=stride. Token stride in raw samples (default=5)")
    ap.add_argument("--ensure_non_empty", action="store_true",
                    help="Ensure each base gets at least one token if possible")

    ap.add_argument("--max_records", type=int, default=-1,
                    help="Process at most N records (for debugging). -1 = no limit")
    ap.add_argument("--write_token_ids", action="store_true",
                    help="Also write full token_ids (can be large). Default: write token_ids anyway for debug.")

    args = ap.parse_args()

    # init tokenizer
    tokenizer = VQETokenizer(
        model_ckpt=args.model_ckpt,
        device=args.device,
        token_batch_size=args.token_batch_size
    )

    total = 0
    empty_base_cnt = 0
    token_lens = []

    with open(args.input_jsonl, "r", encoding="utf-8") as fin, \
         open(args.output_jsonl, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Tokenize+Slice"):
            if not line.strip():
                continue

            rec = json.loads(line)

            signal = rec.get("signal")
            base_spans = rec.get("base_sample_spans_rel")

            if signal is None or base_spans is None:
                continue

            signal_np = np.asarray(signal, dtype=np.float32)

            # tokenize window
            token_list = tokenizer.tokenize_data(signal_np)
            token_ids = extract_token_ids(token_list)
            token_lens.append(len(token_ids))

            # map base -> token
            if args.map_mode == "stride":
                base_token_spans, base_token_ids = map_base_to_tokens_stride(
                    base_spans, token_ids,
                    token_stride=args.token_stride,
                    ensure_non_empty=args.ensure_non_empty
                )
            else:
                base_token_spans, base_token_ids = map_base_to_tokens_ratio(
                    base_spans, token_ids,
                    signal_len=len(signal_np),
                    ensure_non_empty=args.ensure_non_empty
                )

            empty_base_cnt += sum(1 for x in base_token_ids if len(x) == 0)

            out = {
                "read_id": rec.get("read_id"),
                "pattern": rec.get("pattern"),
                "normalize_mode": rec.get("normalize_mode"),

                "signal_len": int(len(signal_np)),
                "token_len": int(len(token_ids)),

                "base_token_spans": base_token_spans,
                "base_token_ids": base_token_ids,

                "map_mode": args.map_mode,
                "token_stride": args.token_stride if args.map_mode == "stride" else None,
                "ensure_non_empty": bool(args.ensure_non_empty),
            }

            # 是否写 full token_ids
            if args.write_token_ids or True:
                out["token_ids"] = token_ids

            fout.write(json.dumps(out) + "\n")

            total += 1
            if args.max_records > 0 and total >= args.max_records:
                break

    # QC summary
    token_lens = np.asarray(token_lens, dtype=np.int32) if len(token_lens) else None

    print("\n[DONE]")
    print("processed records:", total)
    print("empty base_token_ids count:", empty_base_cnt)
    if token_lens is not None and len(token_lens) > 0:
        print("token_len: mean=", float(token_lens.mean()),
              "min=", int(token_lens.min()),
              "max=", int(token_lens.max()))
    print("output:", args.output_jsonl)


if __name__ == "__main__":
    main()
