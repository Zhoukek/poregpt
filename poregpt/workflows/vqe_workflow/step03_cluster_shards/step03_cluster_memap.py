# -*- coding: utf-8 -*-
"""
对 cnn_eval.py 生成的 token embeddings（单个 .npy memmap 文件）进行 FAISS 聚类。
- 输入：一个完整的 .npy 文件路径
- 若 max_sampled_tokens == -1：加载全部
- 否则：加载前 N 个 tokens
"""

import os
import numpy as np
import faiss
import time
import argparse


def cluster_memmap_tokens(
    memmap_npy_path: str,
    output_prefix: str,
    max_sampled_tokens: int = -1,
    num_clusters: int = 16384,
    niter: int = 20,
    nredo: int = 100,
    max_points_per_centroid: int = 65536,
    seed: int = 42,
):
    print("🔧 Running with arguments:")
    print(f"    memmap_npy_path          = {memmap_npy_path}")
    print(f"    output_prefix            = {output_prefix}")
    print(f"    max_sampled_tokens       = {max_sampled_tokens} (-1 means load all)")
    print(f"    num_clusters             = {num_clusters}")
    print(f"    niter                    = {niter}")
    print(f"    nredo                    = {nredo}")
    print(f"    max_points_per_centroid  = {max_points_per_centroid}")
    print(f"    seed                     = {seed}")
    print("-" * 50)

    np.random.seed(seed)

    if not os.path.exists(memmap_npy_path):
        raise FileNotFoundError(f"❌ File not found: {memmap_npy_path}")

    # 打开 memmap（只读，不加载）
    print(f"📥 Opening memmap file: {memmap_npy_path}")

    total_tokens = 1000000000      # ← 必须提供！
    feature_dim =64       # ← 必须提供！

    data = np.memmap(
        memmap_npy_path,
        dtype=np.float32,
        mode='r',
        shape=(total_tokens, feature_dim),
        order='C'
    )

    total_tokens, feature_dim = data.shape
    print(f"📊 File shape: ({total_tokens:,}, {feature_dim})")

    # 决定加载数量
    if max_sampled_tokens == -1:
        actual_total = total_tokens
        print("🔄 Loading ALL tokens (max_sampled_tokens = -1)")
    else:
        actual_total = min(total_tokens, max_sampled_tokens)
        print(f"🔄 Loading first {actual_total:,} tokens")

    # 加载到内存（转为普通 array）
    #all_vectors = np.array(data[:actual_total], dtype=np.float32)
    #all_ids = np.arange(actual_total, dtype=np.int64)

    #直接切片并确保是 float32（mmap 本身已是 float32）
    X = data[:actual_total]  # 这仍然是一个 memory-mapped array view
    print(f"✅ Loaded {len(X):,} tokens into memory.")

# 关键：FAISS 需要 C-contiguous array
# 如果 X 不是 contiguous，FAISS 会报错或静默出错
    if not X.flags.c_contiguous:
        print("⚠️  Data is not C-contiguous. Making a copy...")
        X = np.ascontiguousarray(X, dtype=np.float32)
    else:
    # 确保 dtype 是 float32（FAISS 要求）
       X = X.astype(np.float32, copy=False)

    print(f"✅ Using {len(X):,} tokens for training (shape={X.shape}, contiguous={X.flags.c_contiguous})")
    # === L2 归一化 ===
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    print(f"Before normalization: mean norm = {np.mean(norms):.6f}")
    #all_vectors = all_vectors / norms
    #print(f"📐 After normalization: mean norm = {np.mean(np.linalg.norm(all_vectors, axis=1)):.6f}")

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
    kmeans.train(X)
    time2 = time.time()
    print(f"⏱️  Training time: {time2 - time1:.2f}s")

    print("🔍 Assigning clusters...")
    distances, assignments = kmeans.assign(all_vectors)
    time3 = time.time()
    print(f"⏱️  Assignment time: {time3 - time2:.2f}s")

    # === 保存结果 ===
    cluster_results = np.column_stack((all_ids, assignments, distances))
    output_file = f"{output_prefix}_clustered_k{num_clusters}.npy"
    np.save(output_file, cluster_results)
    print(f"💾 Cluster results saved to: {output_file}")

    centroids_file_npy = f"{output_prefix}_centroids_k{num_clusters}.npy"
    np.save(centroids_file_npy, kmeans.centroids)
    print(f"💾 Centroids saved to: {centroids_file_npy}")

    try:
        import h5py
        centroids_file_h5 = f"{output_prefix}_centroids_k{num_clusters}.h5"
        with h5py.File(centroids_file_h5, 'w') as f:
            f.create_dataset("centroids", data=kmeans.centroids)
        print(f"💾 Centroids also saved to: {centroids_file_h5}")
    except ImportError:
        pass

    print("🎉 Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster tokens from a single .npy memmap file.")
    parser.add_argument("--memmap_npy_path", type=str, required=True,
                        help="Full path to the .npy memmap file (e.g., /data/merged_features.npy)")
    parser.add_argument("--output_prefix", type=str, default="cluster",
                        help="Prefix for output files")
    parser.add_argument("--max_sampled_tokens", type=int, default=-1,
                        help="Number of tokens to load; -1 means load all")
    parser.add_argument("--num_clusters", type=int, default=16384)
    parser.add_argument("--niter", type=int, default=100)
    parser.add_argument("--nredo", type=int, default=10)
    parser.add_argument("--max_points_per_centroid", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    cluster_memmap_tokens(**vars(args))
