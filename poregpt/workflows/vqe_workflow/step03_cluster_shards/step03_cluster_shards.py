# -*- coding: utf-8 -*-
"""
对 cnn_eval.py 生成的 token embeddings（memmap 格式）进行 FAISS 聚类。
不再采样，直接使用指定的 shards_xxx.json 文件中的所有 shards。
【已优化：预分配数组避免 list append + concat】
"""

import os
import json
import numpy as np
import faiss
import time
import argparse
from tqdm import tqdm


def cluster_memmap_tokens_from_shards_json(
    feature_shards_dir: str,
    shards_json: str,
    output_prefix: str,
    max_sampled_tokens: int = 1000_000_000,
    num_clusters: int = 16384,
    niter: int = 20,
    nredo: int = 100,
    max_points_per_centroid: int = 65536,
    seed: int = 42,
):
    # === 打印参数 ===
    print("🔧 Running with arguments:")
    print(f"    feature_shards_dir       = {feature_shards_dir}")
    print(f"    shards_json              = {shards_json}")
    print(f"    output_prefix            = {output_prefix}")
    print(f"    max_sampled_tokens       = {max_sampled_tokens:,}")
    print(f"    num_clusters             = {num_clusters}")
    print(f"    niter                    = {niter}")
    print(f"    nredo                    = {nredo}")
    print(f"    max_points_per_centroid  = {max_points_per_centroid}")
    print(f"    seed                     = {seed}")
    print("-" * 50)

    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    shards_json_path = os.path.join(feature_shards_dir, shards_json)
    assert os.path.exists(shards_json_path), f"❌ Missing {shards_json_path}"

    with open(shards_json_path, 'r') as f:
        meta = json.load(f)

    total_tokens = meta["total_tokens"]
    feature_dim = meta["feature_dim"]
    shards_info = meta["shards"]

    print(f"📊 Total tokens in {shards_json}: {total_tokens:,}, dim: {feature_dim}")

    # === 计算实际要加载的 token 总数 ===
    actual_total = min(total_tokens, max_sampled_tokens)

    # 预分配大数组
    all_vectors = np.empty((actual_total, feature_dim), dtype=np.float32)
    all_ids = np.empty(actual_total, dtype=np.int64)

    offset = 0
    global_token_offset = 0
    pbar = tqdm(total=len(shards_info), desc="Loading shards")

    for shard in shards_info:
        if offset >= actual_total:
            break

        num_tokens_in_shard = shard["num_tokens"]
        shard_file = os.path.join(feature_shards_dir, shard["shard_file"])

        # 只加载需要的部分（如果最后一 shard 超出上限）
        tokens_needed = actual_total - offset
        tokens_to_load = min(num_tokens_in_shard, tokens_needed)

        # 直接从 memmap 读取子集（不转成完整 array）
        data = np.memmap(shard_file, dtype='float32', mode='r', shape=(num_tokens_in_shard, feature_dim))
        all_vectors[offset:offset + tokens_to_load] = data[:tokens_to_load]
        all_ids[offset:offset + tokens_to_load] = np.arange(
            global_token_offset,
            global_token_offset + tokens_to_load,
            dtype=np.int64
        )

        offset += tokens_to_load
        global_token_offset += num_tokens_in_shard
        pbar.update(1)

    pbar.close()

    # 裁剪（理论上不需要，但保险）
    if offset < actual_total:
        all_vectors = all_vectors[:offset]
        all_ids = all_ids[:offset]

    print(f"✅ Loaded {len(all_vectors):,} tokens directly into pre-allocated array.")

    # === L2 归一化：为 spherical K-Means 做准备 ===
    #norms = np.linalg.norm(all_vectors, axis=1, keepdims=True)


    # 防止除零：将极小范数设为1（实际中很少发生，但安全起见）
    #norms = np.where(norms == 0, 1.0, norms)
    # 打印归一化前后的范数统计
    #print(f"Before normalization: mean norm = {np.mean(np.linalg.norm(all_vectors, axis=1)):.6f}")
    #all_vectors = all_vectors / norms
    #print(f"📐 L2 normalized vectors (mean norm: {np.mean(np.linalg.norm(all_vectors, axis=1)):.6f})")

    # === FAISS KMeans 聚类 ===
    time1 = time.time()
    kmeans = faiss.Kmeans(
        d=feature_dim,
        k=num_clusters,
        niter=niter,
        nredo=nredo,
        verbose=True,
        gpu=True,
        spherical=False,
        max_points_per_centroid=max_points_per_centroid,
        min_points_per_centroid=1,
        seed=seed
    )

    print("🚀 Training K-Means...")
    kmeans.train(all_vectors)
    time2 = time.time()
    print(f"⏱️  Training time: {time2 - time1:.2f}s")

    print("🔍 Assigning clusters...")
    distances, assignments = kmeans.assign(all_vectors)
    time3 = time.time()
    print(f"⏱️  Assignment time: {time3 - time2:.2f}s")

    # === 保存结果 ===
    cluster_results = np.column_stack((all_ids, assignments, distances))
    output_file = f"{output_prefix}_clustered_k{num_clusters}_mppc{max_points_per_centroid}.npy"
    np.save(output_file, cluster_results)
    print(f"💾 Cluster results saved to: {output_file}")

    centroids_file_npy = f"{output_prefix}_centroids_k{num_clusters}_mppc{max_points_per_centroid}.npy"
    np.save(centroids_file_npy, kmeans.centroids)
    print(f"💾 Centroids saved to: {centroids_file_npy}")

    try:
        import h5py
        centroids_file_h5 = f"{output_prefix}_centroids_k{num_clusters}_mppc{max_points_per_centroid}.h5"
        with h5py.File(centroids_file_h5, 'w') as f:
            f.create_dataset("centroids", data=kmeans.centroids)
        print(f"💾 Centroids also saved to: {centroids_file_h5}")
    except ImportError:
        pass

    print("🎉 Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster tokens from a specified shards JSON file (no sampling).")
    parser.add_argument("--feature_shards_dir", type=str, required=True,
                        help="Directory containing shard_*.npy files")
    parser.add_argument("--shards_json", type=str, default="shards.json",
                        help="Name of the shards metadata file (e.g., shards_sampled_20p.json)")
    parser.add_argument("--output_prefix", type=str, default="cluster",
                        help="Prefix for output files")
    parser.add_argument("--max_sampled_tokens", type=int, default=10_000_000,
                        help="Maximum number of tokens to load into memory (default: 10M)")
    parser.add_argument("--num_clusters", type=int, default=16384,
                        help="Number of K-Means clusters")
    parser.add_argument("--niter", type=int, default=100)
    parser.add_argument("--nredo", type=int, default=10)
    parser.add_argument("--max_points_per_centroid", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    cluster_memmap_tokens_from_shards_json(**vars(args))
