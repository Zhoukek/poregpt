import os
import json
import argparse
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel


def load_records(path: str):
    """
    逐行读取 JSONL
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def pad_2d(
    seqs: List[List[int]],
    device: str,
    pad_id: int = 0,
    pad_id_model: int = 1,
    offset: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将变长 token 序列 pad 成等长矩阵。

    实现要点：只对真实 token 应用 `offset`，保证最终 padding 的索引等于模型期望的 pad id。

    Args:
        seqs: list of token-id list (输入空间的 token id，padding 使用 `pad_id` 表示，例如 0)
        device: cuda / cpu
        pad_id: 输入中用于表示 padding 的 id（默认 0）
        pad_id_model: 模型 embedding 中实际的 padding id（默认 1）
        offset: 对非 padding token 应用的偏移量（默认 128）

    Returns:
        x:    [B, Lmax] long tensor（已转换到模型索引空间并移动到 device）
        mask: [B, Lmax] long tensor, 有效 token 为 1，padding 为 0
    """
    batch_size = len(seqs)
    max_len = max((len(s) for s in seqs), default=0)

    # 先用输入空间的 pad_id 填充
    x = torch.full((batch_size, max_len), fill_value=pad_id, dtype=torch.long)
    mask = torch.zeros((batch_size, max_len), dtype=torch.long)

    for i, s in enumerate(seqs):
        if len(s) == 0:
            continue
        length = len(s)
        x[i, :length] = torch.tensor(s, dtype=torch.long)
        mask[i, :length] = 1

    # 只对非 padding 的位置应用偏移，保证 padding 最终是 pad_id_model
    nonpad_mask = mask.bool()
    x_offset = x.clone()
    if offset != 0:
        x_offset[nonpad_mask] = x_offset[nonpad_mask] + offset
    # 把原始 padding 位置置为模型所需的 pad id
    x_offset[~nonpad_mask] = pad_id_model

    return x_offset.to(device), mask.to(device)


def flatten_base_token_ids(
    base_token_ids: List[List[int]]
) -> Tuple[List[int], List[Optional[Tuple[int, int]]]]:
    """
    将每个 base 对应的 token 列表展开成一个完整 token 序列，
    同时记录每个 base 在这个展开序列中的 span。

    例如:
        base_token_ids = [[11, 12], [13], [], [21, 22, 23]]

    返回:
        flat_tokens = [11, 12, 13, 21, 22, 23]
        spans      = [(0,2), (2,3), None, (3,6)]

    说明:
        span 使用 [start, end) 左闭右开
        如果这个 base 没有 token，则记为 None
    """
    flat_tokens: List[int] = []
    spans: List[Optional[Tuple[int, int]]] = []

    cursor = 0
    for tok_list in base_token_ids:
        if tok_list is None or len(tok_list) == 0:
            spans.append(None)
            continue

        start = cursor
        flat_tokens.extend(tok_list)
        cursor += len(tok_list)
        end = cursor

        spans.append((start, end))

    return flat_tokens, spans



def get_hidden_size(backbone, device: str) -> int:
    """
    优先从 config 中取 hidden_size，取不到时做一次 dummy forward 推断
    """
    hidden_size = getattr(backbone.config, "hidden_size", None)
    if hidden_size is not None:
        return int(hidden_size)

    with torch.no_grad():
        x = torch.zeros((1, 1), dtype=torch.long, device=device)
        mask = torch.ones((1, 1), dtype=torch.long, device=device)
        out = backbone(x, attention_mask=mask, output_hidden_states=True)
        hidden = out.hidden_states[-1]
        return int(hidden.shape[-1])


def pool_one_span(hidden_span: torch.Tensor, pool: str) -> torch.Tensor:
    """
    对某一个 span 的 hidden states 做 pooling

    Args:
        hidden_span: [T, D]
        pool: mean / middle / first / last

    Returns:
        [D]
    """
    if hidden_span.size(0) == 0:
        raise ValueError("Cannot pool empty span.")

    if pool == "mean":
        return hidden_span.mean(dim=0)
    elif pool == "middle":
        return hidden_span[hidden_span.size(0) // 2]
    elif pool == "first":
        return hidden_span[0]
    elif pool == "last":
        return hidden_span[-1]
    else:
        raise ValueError(f"Unsupported pool mode: {pool}")


def pool_hidden_by_spans(
    hidden: torch.Tensor,
    spans_batch: List[List[Optional[Tuple[int, int]]]],
    pool: str,
    base_index: int,
) -> np.ndarray:
    """
    将模型输出 hidden states 按 base span 聚合为 base-level embedding

    Args:
        hidden: [B, L, D]
        spans_batch: 长度为 B，每个元素是当前样本每个 base 的 span 列表
        pool: pooling 方式
        base_index:
            -1: 输出全部 base -> [B, L_pattern, D]
            >=0: 只输出某一个 base -> [B, D]

    Returns:
        numpy array
    """
    B, _, D = hidden.shape

    if base_index >= 0:
        out = np.full((B, D), np.nan, dtype=np.float32)

        for b in range(B):
            spans = spans_batch[b]
            if base_index >= len(spans):
                continue

            span = spans[base_index]
            if span is None:
                continue

            start, end = span
            pooled = pool_one_span(hidden[b, start:end, :], pool=pool)
            out[b] = pooled.detach().cpu().numpy().astype(np.float32)

        return out

    max_bases = max(len(spans) for spans in spans_batch) if spans_batch else 0
    out = np.full((B, max_bases, D), np.nan, dtype=np.float32)

    for b in range(B):
        spans = spans_batch[b]
        for pos, span in enumerate(spans):
            if span is None:
                continue

            start, end = span
            pooled = pool_one_span(hidden[b, start:end, :], pool=pool)
            out[b, pos, :] = pooled.detach().cpu().numpy().astype(np.float32)

    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_jsonl", type=str, required=True,
                        help="每条记录至少包含 read_id / pattern / base_token_ids")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--label", type=str, required=True)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--pad_id", type=int, default=1)

    parser.add_argument("--pool", type=str, default="mean",
                        choices=["mean", "middle", "first", "last"])

    parser.add_argument("--base_index", type=int, default=-1,
                        help="-1 表示导出全部 base；>=0 表示只导出某一个 base")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    out_emb = os.path.join(args.out_dir, f"base_emb_{args.pool}.npy")
    out_ids = os.path.join(args.out_dir, f"read_ids.jsonl")


    backbone = AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
    backbone = backbone.to(args.device)
    backbone.eval()

    hidden_size = get_hidden_size(backbone, args.device)

    records: List[Dict[str, Any]] = []
    global_max_bases = 0

    for rec in tqdm(load_records(args.input_jsonl), desc="Loading records"):
        read_id = rec.get("read_id")
        pattern = rec.get("pattern")
        base_token_ids = rec.get("base_token_ids")

        if read_id is None or base_token_ids is None:
            continue
        if not isinstance(base_token_ids, list):
            continue

        flat_tokens, spans = flatten_base_token_ids(base_token_ids)

        records.append({
            "read_id": read_id,
            "pattern": pattern,
            "base_token_ids": base_token_ids,
            "flat_tokens": flat_tokens,
            "spans": spans,
        })

        global_max_bases = max(global_max_bases, len(base_token_ids))

    if len(records) == 0:
        raise RuntimeError("No valid records found in input_jsonl.")

    num_records = len(records)


    if args.base_index >= 0:
        final_arr = np.full((num_records, hidden_size), np.nan, dtype=np.float32)
    else:
        final_arr = np.full((num_records, global_max_bases, hidden_size), np.nan, dtype=np.float32)

    read_meta = [
        {
            "read_id": r["read_id"],
            "pattern": r["pattern"]
        }
        for r in records
    ]

    pbar = tqdm(total=num_records, desc="Embedding")

    for start_idx in range(0, num_records, args.batch_size):
        batch = records[start_idx:start_idx + args.batch_size]

        token_seqs = [r["flat_tokens"] for r in batch]
        spans_batch = [r["spans"] for r in batch]

        non_empty_indices = [i for i, seq in enumerate(token_seqs) if len(seq) > 0]

        if len(non_empty_indices) == 0:
            pbar.update(len(batch))
            continue

        token_seqs_non_empty = [token_seqs[i] for i in non_empty_indices]
        spans_non_empty = [spans_batch[i] for i in non_empty_indices]

        x, mask = pad_2d(token_seqs_non_empty, device=args.device, pad_id=args.pad_id)

        with torch.no_grad():
            outputs = backbone(x, attention_mask=mask, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]   # [B_nonempty, L, D]

        pooled_np = pool_hidden_by_spans(
            hidden=hidden,
            spans_batch=spans_non_empty,
            pool=args.pool,
            base_index=args.base_index,
        )

        for local_i, original_i in enumerate(non_empty_indices):
            global_i = start_idx + original_i

            if args.base_index >= 0:
                final_arr[global_i] = pooled_np[local_i]
            else:
                num_bases = len(spans_batch[original_i])
                final_arr[global_i, :num_bases, :] = pooled_np[local_i, :num_bases, :]

        pbar.update(len(batch))

    pbar.close()


    np.save(out_emb, final_arr)

    with open(out_ids, "w", encoding="utf-8") as f:
        for item in read_meta:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("[DONE]")
    print("emb shape:", final_arr.shape)
    print("saved emb:", out_emb)
    print("saved ids:", out_ids)


if __name__ == "__main__":
    main()


