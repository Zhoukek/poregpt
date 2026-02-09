
from poregpt.utils import nanopore_process_signal
from .process_data import sliding_window_chunks,process_read
import faiss
import gzip
import json
from tqdm import tqdm
from ont_fast5_api.fast5_interface import get_fast5_file
import numpy as np
from abc import ABC, abstractmethod
import os

# 基类：抽象类
class InterfaceTokenizer(ABC):
    @abstractmethod
    def tokenize_data(self, signal: np.ndarray) -> list:
        """将原始信号数据转换为 token 字符串"""
        pass

    @abstractmethod
    def tokenize_read(self, read, nanopore_signal_process_strategy="apple") -> list:
        """将测序读段（read）对象转换为 token 字符串"""
        pass

    @abstractmethod
    def tokenize_fast5(self, fast5_path: str, output_path:str, nanopore_signal_process_strategy="apple"):
        """从 FAST5 文件中读取信号并保存 token 到输出路径"""
        pass

# 全局 FAISS 索引（每个子进程初始化一次）
_GLOBAL_INDEX = None

def init_worker(centroids_path: str, use_gpu: bool = True, gpu_id: int = 0):
    global _GLOBAL_INDEX
    if _GLOBAL_INDEX is not None:
        return
    data = np.load(centroids_path)
    centroids = data['centroids'].astype(np.float32)
    d = centroids.shape[1]

    if use_gpu and hasattr(faiss, 'StandardGpuResources'):
        # === GPU 模式 ===
        print("🚀 Initializing FAISS GPU index...")
        res = faiss.StandardGpuResources()  # GPU 资源管理器
        cpu_index = faiss.IndexFlatL2(d)
        cpu_index.add(centroids) # type: ignore
        # 将 CPU 索引搬到 GPU（默认 device=0）
        _GLOBAL_INDEX = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)
        # print(f"✅ FAISS GPU index ready on device {gpu_id}, {centroids.shape[0]} centroids")
    else:
        # === CPU 回退模式 ===
        print("💻 Using FAISS CPU index...")
        cpu_index = faiss.IndexFlatL2(d)
        cpu_index.add(centroids) # type: ignore
        _GLOBAL_INDEX = cpu_index
        
def tokenize_signal_with_global_index(signal: np.ndarray) -> list: 
    _, I = _GLOBAL_INDEX.search(signal, 1) # type: ignore
    cluster_ids = I[:, 0].tolist()

    parts = []
    for token_id in cluster_ids:
        parts.append(f"<|bwav:{int(token_id) + 128}|>") # token_id 偏移 128，避免与特殊符号冲突
    return parts
class KMSTokenizer(InterfaceTokenizer):
    """
    Nanopore RVQ Tokenizer 封装类。

    功能：
        - 加载预训练 RVQ 模型
        - tokenize 单个 read / numpy 信号 / 整个 FAST5 目录
    """

    def __init__(
        self,
        centroids_path: str,
        gpu_id: int = 0,
    ):
        """
        初始化 tokenizer。
        """
        data = np.load(centroids_path)
        self.window_size = data["dim"]
        self.stride = data["stride"]
        init_worker(centroids_path, use_gpu=True, gpu_id=gpu_id)

    def tokenize_data(self, signal: np.ndarray) -> list:
        if signal.size == 0:
            return []
        vec_list = sliding_window_chunks(signal, self.window_size, self.stride)
        if not vec_list:
            return []
        try:
            X = np.stack(vec_list, axis=0).astype(np.float32)
        except Exception:
            return []
        return tokenize_signal_with_global_index(X)


    def tokenize_read(self, read, nanopore_signal_process_strategy="apple") -> list:
        signal_raw = process_read(read)
        if signal_raw is None:
            return []

        signal_processed = nanopore_process_signal(signal_raw,nanopore_signal_process_strategy)
        if signal_processed is None:
            return []
        return self.tokenize_data(signal_processed)


 
    def tokenize_fast5(self, fast5_path: str, output_path:str, nanopore_signal_process_strategy="apple"):
        print(f"✅ Processing {fast5_path} with strategy {nanopore_signal_process_strategy}")
        results = []
        with get_fast5_file(fast5_path, mode="r") as f5:
            for read in tqdm(f5.get_reads(), desc=os.path.basename(fast5_path)):
                try:
                    token_list = self.tokenize_read(read,nanopore_signal_process_strategy)
                    token_str = "".join(token_list)
                    results.append({"id": read.read_id, "text": token_str})
                except Exception as e:
                    print(f"❌ Failed on read {read.read_id}: {e}")
                    continue

        with gzip.open(output_path, 'wt', encoding='utf-8') as f:
            for item in results:
                f.write(json.dumps(item) + '\n')
        print(f"✅ Wrote {len(results)} reads to {output_path}")
