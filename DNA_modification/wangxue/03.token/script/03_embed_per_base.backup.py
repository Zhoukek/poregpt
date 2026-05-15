#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from typing import List, Dict, Any, Tuple

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModel


def load_records(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def pad_2d(seqs: List[List[int]], device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    seqs: list of token_id lists (variable length)
    returns:
      x: [B, Lmax] long
      mask: [B, Lmax] long (1 for valid)
    """
    lengths = [len(s) for s in seqs]
    max_len = max(lengths) if lengths else 0
    x = torch.zeros(len(seqs), max_len, dtype=torch.long)
    m = torch.zeros(len(seqs), max_len, dtype=torch.long)
    for i, s in enumerate(seqs):
        if len(s) == 0:
            continue
        x[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        m[i, :len(s)] = 1
    return x.to(device), m.to(device)


def pool_base_embeddings(
    hidden: torch.Tensor,              # [B, L, D]
    mask: torch.Tensor,                # [B, L]
    base_token_ids: List[List[List[int]]],
    pool: str = "mean"
) -> torch.Tensor:
    """
    hidden: last hidden states for padded token ids, aligned with input token sequences
    base_token_ids: batch sized list:
        base_token_ids[b][pos] = list of token ids for that base
        NOTE: these are the SAME ids as were fed into the model for this base sequence.
    But we need token *positions* in the padded sequence, not token *values*.
    Here we assume for each base we used exactly the token list we fed in as input,
    in the SAME order, so base spans correspond to contiguous slices in the per-base input.

    So we don't match by id; we use the fact that each base is embedded by feeding its own token list.
    That is: we will call this function on per-base padded sequences, not whole-window sequences.

    Therefore base_token_ids is only used to know which entries are empty and lengths.
    """
    B, L, D = hidden.shape
    out = torch.zeros(B, D, device=hidden.device, dtype=hidden.dtype)

    for b in range(B):
        valid_len = int(mask[b].sum().item())
        if valid_len <= 0:
            continue
        hb = hidden[b, :valid_len, :]  # [len, D]

        if pool == "mean":
            out[b] = hb.mean(dim=0)
        elif pool == "middle":
            mid = valid_len // 2
            out[b] = hb[mid]
        else:
            raise ValueError(f"Unknown pool: {pool}")

    return out  # [B, D]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--input_jsonl", required=True,
                    help="JSONL that contains base_token_ids (list[list[int]]) per record")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--label", required=True)

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=64)

    ap.add_argument("--pool", choices=["mean", "middle"], default="mean",
                    help="How to aggregate multiple tokens per base")

    # which base(s) to export: all bases or one index
    ap.add_argument("--base_index", type=int, default=-1,
                    help="If >=0, only export this base position embedding (0-based). If -1, export all bases.")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_emb = os.path.join(args.out_dir, f"{args.label}_base_emb_{args.pool}.npy")
    out_ids = os.path.join(args.out_dir, f"{args.label}_read_ids.jsonl")

    # load model
    backbone = AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
    backbone = backbone.to(args.device)
    backbone.eval()

    # We'll build a dataset of (read_id, base_pos, token_seq_for_base)
    # Then run in batches.
    samples = []
    for rec in tqdm(load_records(args.input_jsonl), desc="Loading"):
        rid = rec.get("read_id")
        pattern = rec.get("pattern")
        base_token_ids = rec.get("base_token_ids")  # list of list[int], length=pattern_len

        if rid is None or base_token_ids is None:
            continue

        if args.base_index >= 0:
            if args.base_index >= len(base_token_ids):
                continue
            tok = base_token_ids[args.base_index]
            samples.append((rid, pattern, args.base_index, tok))
        else:
            for pos, tok in enumerate(base_token_ids):
                samples.append((rid, pattern, pos, tok))

    if len(samples) == 0:
        raise RuntimeError("No valid samples loaded. Ensure input JSONL has base_token_ids.")

    # We want output shape:
    # - if base_index>=0: [N, D] where N = number of reads (records)
    # - else: [N_reads, L_pattern, D]
    #
    # But we expanded samples per base, so we will reconstruct.
    #
    # We'll also store read_id order.

    # First, determine pattern length per record if exporting all bases
    # We'll assume constant pattern length across records.
    # For safety, track max pos.
    max_pos = max(pos for _, _, pos, _ in samples)
    L_pattern = max_pos + 1

    # We'll collect per-(rid, pos) embeddings then stack.
    # Use dict of rid -> [L_pattern, D] (filled with nan for missing)
    rid_to_idx = {}
    rid_list = []
    emb_chunks = {}  # rid -> list of embeddings per pos

    # batching over samples
    bs = args.batch_size
    i = 0

    pbar = tqdm(total=len(samples), desc="Embedding")
    while i < len(samples):
        batch_items = samples[i:i+bs]
        i += bs

        token_seqs = [tok if tok is not None else [] for (_, _, _, tok) in batch_items]
        # Ensure non-empty: if a base has no tokens, we cannot embed; keep placeholder
        # We'll still pass an empty to pad_2d which makes length 0 -> would break max_len,
        # so handle by replacing empty with [0] and then ignore via mask.
        fixed_token_seqs = []
        for tok in token_seqs:
            if tok is None or len(tok) == 0:
                fixed_token_seqs.append([0])  # dummy token id
            else:
                fixed_token_seqs.append(tok)

        x, mask = pad_2d(fixed_token_seqs, args.device)

        with torch.no_grad():
            out = backbone(x, attention_mask=mask, output_hidden_states=True)
            h = out.hidden_states[-1]  # [B, L, D]

        # pool per base sequence
        pooled = pool_base_embeddings(h, mask, base_token_ids=[[]]*len(batch_items), pool=args.pool)  # [B, D]
        pooled_np = pooled.detach().cpu().numpy()

        # assign back
        for j, (rid, pattern, pos, tok) in enumerate(batch_items):
            if rid not in rid_to_idx:
                rid_to_idx[rid] = len(rid_list)
                rid_list.append({"read_id": rid, "pattern": pattern})

            if rid not in emb_chunks:
                emb_chunks[rid] = [None] * L_pattern

            # if tok empty, set to nan
            if tok is None or len(tok) == 0:
                emb_chunks[rid][pos] = None
            else:
                emb_chunks[rid][pos] = pooled_np[j]

        pbar.update(len(batch_items))
    pbar.close()

    # build final array
    # if base_index>=0: [N_reads, D]
    # else: [N_reads, L_pattern, D]
    # Determine D from first non-null
    D = None
    for rid in emb_chunks:
        for v in emb_chunks[rid]:
            if v is not None:
                D = v.shape[0]
                break
        if D is not None:
            break
    if D is None:
        raise RuntimeError("All base embeddings are empty. Check tokenization output.")

    if args.base_index >= 0:
        out_arr = np.full((len(rid_list), D), np.nan, dtype=np.float32)
        for rid, idx in rid_to_idx.items():
            v = emb_chunks[rid][args.base_index]
            if v is not None:
                out_arr[idx] = v.astype(np.float32)
    else:
        out_arr = np.full((len(rid_list), L_pattern, D), np.nan, dtype=np.float32)
        for rid, idx in rid_to_idx.items():
            for pos in range(L_pattern):
                v = emb_chunks[rid][pos]
                if v is not None:
                    out_arr[idx, pos, :] = v.astype(np.float32)

    np.save(out_emb, out_arr)

    # save read id order
    with open(out_ids, "w", encoding="utf-8") as f:
        for item in rid_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("[DONE]")
    print("emb shape:", out_arr.shape)
    print("saved:", out_emb)
    print("read_id list:", out_ids)


if __name__ == "__main__":
    main()