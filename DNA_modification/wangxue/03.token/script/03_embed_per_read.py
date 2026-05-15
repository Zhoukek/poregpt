import os
import json
import argparse
from typing import List, Tuple, Dict, Any

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


def flatten_base_token_ids(base_token_ids: List[List[int]]) -> List[int]:
    """
    将一条 read 的 base_token_ids 展平成一条完整 token 序列

    Example:
        [[11,12], [13], [], [21,22]]
        -> [11,12,13,21,22]
    """
    flat_tokens = []
    for tok_list in base_token_ids:
    
        if tok_list is None or len(tok_list) == 0:
            continue
        flat_tokens.extend(tok_list)
    return flat_tokens


def pad_2d(
    seqs: List[List[int]],
    device: str,
    pad_id: int = 0,
    pad_id_model: int = 1,
    offset: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将变长 token 序列 pad 成等长矩阵。

    只对真实 token 应用 `offset`，保证最终 padding 的索引等于模型期望的 pad id。

    Args:
        seqs: list of token-id list (输入空间的 token id，padding 使用 `pad_id` 表示，例如 0)
        device: cuda / cpu
        pad_id: 输入中用于表示 padding 的 id（默认 0）
        pad_id_model: 模型 embedding 中实际的 padding id（默认 1）
        offset: 对非 padding token 应用的偏移量（默认 128）

    Returns:
        x:    [B, Lmax] long tensor（已转换为模型索引并移动到 device）
        mask: [B, Lmax] long tensor, 有效 token 为 1，padding 为 0
    """
    batch_size = len(seqs)
    max_len = max((len(s) for s in seqs), default=0)

    x = torch.full((batch_size, max_len), fill_value=pad_id, dtype=torch.long)
    mask = torch.zeros((batch_size, max_len), dtype=torch.long)

    for i, s in enumerate(seqs):
        if len(s) == 0:
            continue
        length = len(s)
        x[i, :length] = torch.tensor(s, dtype=torch.long)
        mask[i, :length] = 1

    nonpad_mask = mask.bool()
    x_offset = x.clone()
    if offset != 0:
        x_offset[nonpad_mask] = x_offset[nonpad_mask] + offset
    x_offset[~nonpad_mask] = pad_id_model

    return x_offset.to(device), mask.to(device)


def get_hidden_size(backbone, device: str) -> int:
    """
    优先从 config 中获取 hidden_size，失败则 dummy forward 推断
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


def pool_read_embeddings(
    hidden: torch.Tensor,   # [B, L, D]
    mask: torch.Tensor,     # [B, L]
    pool: str = "mean"
) -> torch.Tensor:
    """
    对整条 read 的 hidden states 做 pooling，得到 per-read embedding

    Args:
        hidden: [B, L, D]
        mask:   [B, L]
        pool:   mean / first / last / middle

    Returns:
        pooled: [B, D]
    """
    B, L, D = hidden.shape
    out = torch.zeros((B, D), device=hidden.device, dtype=hidden.dtype)

    for b in range(B):
        valid_len = int(mask[b].sum().item())
        if valid_len == 0:
            continue

        hb = hidden[b, :valid_len, :]   # [valid_len, D]

        if pool == "mean":
            out[b] = hb.mean(dim=0)
        elif pool == "first":
            out[b] = hb[0]
        elif pool == "last":
            out[b] = hb[-1]
        elif pool == "middle":
            out[b] = hb[valid_len // 2]
        else:
            raise ValueError(f"Unsupported pool mode: {pool}")

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
                        choices=["mean", "first", "last", "middle"])

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    out_emb = os.path.join(args.out_dir, f"read_emb_{args.pool}.npy")
    out_ids = os.path.join(args.out_dir, f"read_ids.jsonl")

    # --------------------------------------------------
    # 1. 加载模型
    # --------------------------------------------------
    backbone = AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
    backbone = backbone.to(args.device)
    backbone.eval()

    hidden_size = get_hidden_size(backbone, args.device)

    # --------------------------------------------------
    # 2. 预处理 records
    # --------------------------------------------------
    records: List[Dict[str, Any]] = []

    for rec in tqdm(load_records(args.input_jsonl), desc="Loading records"):
        read_id = rec.get("read_id")
        pattern = rec.get("pattern")
        base_token_ids = rec.get("base_token_ids")

        if read_id is None or base_token_ids is None:
            continue
        if not isinstance(base_token_ids, list):
            continue

        flat_tokens = flatten_base_token_ids(base_token_ids)

        records.append({
            "read_id": read_id,
            "pattern": pattern,
            "flat_tokens": flat_tokens,
        })

    if len(records) == 0:
        raise RuntimeError("No valid records found in input_jsonl.")

    num_records = len(records)

    # --------------------------------------------------
    # 3. 预分配输出
    # --------------------------------------------------
    final_arr = np.full((num_records, hidden_size), np.nan, dtype=np.float32)

    read_meta = [
        {
            "read_id": r["read_id"],
            "pattern": r["pattern"]
        }
        for r in records
    ]

    # --------------------------------------------------
    # 4. 分 batch forward
    # --------------------------------------------------
    pbar = tqdm(total=num_records, desc="Embedding reads")

    for start_idx in range(0, num_records, args.batch_size):
        batch = records[start_idx:start_idx + args.batch_size]
        token_seqs = [r["flat_tokens"] for r in batch]

        # 只处理非空 read
        non_empty_indices = [i for i, seq in enumerate(token_seqs) if len(seq) > 0]

        if len(non_empty_indices) == 0:
            pbar.update(len(batch))
            continue

        token_seqs_non_empty = [token_seqs[i] for i in non_empty_indices]

        x, mask = pad_2d(token_seqs_non_empty, device=args.device, pad_id=args.pad_id)

        with torch.no_grad():
            outputs = backbone(x, attention_mask=mask, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]   # [B_nonempty, L, D]

        pooled = pool_read_embeddings(hidden, mask, pool=args.pool)  # [B_nonempty, D]
        pooled_np = pooled.detach().cpu().numpy().astype(np.float32)

        # 写回总数组
        for local_i, original_i in enumerate(non_empty_indices):
            global_i = start_idx + original_i
            final_arr[global_i] = pooled_np[local_i]

        pbar.update(len(batch))

    pbar.close()

    # --------------------------------------------------
    # 5. 保存结果
    # --------------------------------------------------
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


