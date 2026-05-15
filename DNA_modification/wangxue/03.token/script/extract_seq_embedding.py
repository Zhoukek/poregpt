#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModel


# =========================
# 参数
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Extract sequence embeddings from token_ids")

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


# =========================
# 数据读取
# =========================
def load_data(path):
    with open(path, "r") as f:
        for line in f:
            rec = json.loads(line)
            yield rec["token_ids"]


# =========================
# padding
# =========================
def collate_batch(batch, pad_token_id=0, device="cpu"):
    seqs = [torch.tensor(x, dtype=torch.long) for x in batch]

    lengths = [len(s) for s in seqs]
    max_len = max(lengths)

    padded = torch.full((len(seqs), max_len), pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.long)

    for i, s in enumerate(seqs):
        padded[i, :len(s)] = s
        mask[i, :len(s)] = 1

    return padded.to(device), mask.to(device)


# =========================
# 主流程
# =========================
def main():
    args = parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"[INFO] device = {device}")

    # 加载模型
    backbone = AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
    backbone = backbone.to(device)
    backbone.eval()

    all_embeddings = []
    batch = []

    for token_ids in tqdm(load_data(args.input_jsonl), desc="Embedding"):
        batch.append(token_ids)

        if len(batch) >= args.batch_size:
            x, mask = collate_batch(batch, device=device)

            with torch.no_grad():
                out = backbone(x, attention_mask=mask)
                h = out.last_hidden_state  # [B, L, D]

                mask_f = mask.unsqueeze(-1).float()
                seq_emb = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-8)

            all_embeddings.append(seq_emb.cpu().numpy())
            batch = []

    # 最后一批
    if batch:
        x, mask = collate_batch(batch, device=device)

        with torch.no_grad():
            out = backbone(x, attention_mask=mask)
            h = out.last_hidden_state

            mask_f = mask.unsqueeze(-1).float()
            seq_emb = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-8)

        all_embeddings.append(seq_emb.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)

    # 保存
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    np.save(args.out_path, embeddings)

    print("\n[DONE]")
    print("shape:", embeddings.shape)
    print("saved to:", args.out_path)


if __name__ == "__main__":
    main()