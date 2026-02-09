# nanopore_signal_tokenizer/vq_tokenizer.py
# Suppress known deprecation warnings

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*pkg_resources is deprecated.*",
    category=UserWarning,
    module="ont_fast5_api"
)

import os
import json
import gzip
import numpy as np
import torch
from math import ceil
from ont_fast5_api.fast5_interface import get_fast5_file
from ...utils.signal import nanopore_process_signal
from tqdm import tqdm
from scipy.signal import medfilt
import numpy as np
from typing import List

import numpy as np
from typing import List


# Import your model definition (must define NanoporeVQModel)
from .vq_model import NanoporeVQModel


class VQETokenizer:
    """
    Nanopore Single-Layer VQ Tokenizer.
    - Uses VectorQuantize (not RVQ)
    - No reconstruction loss needed
    - Designed for diversity + waveform backbone retention
    """

    def __init__(
        self,
        model_ckpt: str = "nanopore_vq_tokenizer.pth",
        device: str = "cuda",
        token_batch_size: int = 8000,
    ):
        # --- Device setup ---
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = device.strip()
            if device.startswith("cuda"):
                if not torch.cuda.is_available():
                    print("⚠️ CUDA not available, falling back to CPU.")
                    self.device = "cpu"
                else:
                    self.device = device
            elif device == "cpu":
                self.device = "cpu"
            else:
                raise ValueError(f"Unsupported device: {device}")
        
        print(f"✅ Using device: {self.device}")

        # --- Load checkpoint ---
        print(f"📂 Loading checkpoint: {model_ckpt}")
        ckpt_data = torch.load(model_ckpt, map_location="cpu",weights_only=False)

        # ✅ 1. 从 checkpoint 中读取 cnn_type（关键！）
        if 'cnn_type' not in ckpt_data:
            print("Checkpoint does not contain 'cnn_type'. forced to 0")
            cnn_type = 0
        else:
            cnn_type = ckpt_data['cnn_type']

        # ✅ 正确：从 model_state_dict 中找 codebook
        state_dict = ckpt_data['model_state_dict']
        embed_keys = [k for k in state_dict.keys() if "_codebook.embed" in k]
        if not embed_keys:
            raise RuntimeError("No codebook embedding found in checkpoint.")

        # Assume single quantizer: key like 'quantizer._codebook.embed'
        embed_key = embed_keys[0]
        embed_tensor = state_dict[embed_key]  # shape: [codebook_size, dim] or [1, codebook_size, dim]

        if len(embed_tensor.shape) == 3:
            codebook_size = int(embed_tensor.shape[1])
            dim = int(embed_tensor.shape[2])
        elif len(embed_tensor.shape) == 2:
            codebook_size = int(embed_tensor.shape[0])
            dim = int(embed_tensor.shape[1])
        else:
            raise RuntimeError(f"Unexpected codebook shape: {embed_tensor.shape}")

        self.codebook_size = codebook_size
        self.dim = dim
        print(f"🎯 Inferred: codebook_size={codebook_size}, dim={dim}, cnn_type={cnn_type}")

        # --- Instantiate model ---
        self.model = NanoporeVQModel(codebook_size=codebook_size,cnn_type=cnn_type)

        if not hasattr(self.model, 'cnn_stride'):
            raise AttributeError("Model must define 'cnn_stride' (total downsampling rate).")
        if not hasattr(self.model, 'margin_stride_count'):
            self.model.margin_stride_count = 2  # default fallback

        self.downsample_rate = self.model.cnn_stride
        self.margin_stride_count = self.model.margin_stride_count
        self.margin = self.margin_stride_count * self.downsample_rate
        self.model_RF = self.model.RF
        if token_batch_size < 1:
            token_batch_size = 1
        self.chunk_size = token_batch_size * self.downsample_rate

        # --- Load state dict ---
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.device)

        print("\n✅ VQTokenizer initialized:")
        print(f"   Checkpoint       : {os.path.abspath(model_ckpt)}")
        print(f"   Device           : {self.device}")
        print(f"   Codebook size    : {self.codebook_size}")
        print(f"   Latent dim       : {self.dim}")
        print(f"   Downsample rate  : {self.downsample_rate}")
        print(f"   Chunk size       : {self.chunk_size}")
        print(f"   Margin           : {self.margin} samples")
        print("-" * 60)
    

    def _tokenize_chunked_signal(self, signal: np.ndarray) -> np.ndarray:
        """Tokenize 1D signal using sliding window with margin."""
        if signal.ndim != 1:
            raise ValueError("Signal must be 1D.")
        L = len(signal)
        if L < self.model_RF:
            return np.array([], dtype=np.int64)

        if L == 0:
            return np.array([], dtype=np.int64)

        T_expected = (L + self.downsample_rate - 1) // self.downsample_rate

        if L <= self.chunk_size:
            padded = np.pad(signal, (0, self.chunk_size - L), mode='constant')
            x = torch.from_numpy(padded).float().unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                recon,tokens,loss,loss_breakdown = self.model(x)  # returns [B, T] or [B, T, 1] → squeeze to [T]
            tokens = tokens.squeeze(0).cpu().numpy()
            if tokens.ndim == 2:
                tokens = tokens[:, 0]  # take first (and only) layer
            return tokens[:T_expected].astype(np.int64)

        # Long signal: sliding window
        margin_samples = self.margin
        step_samples = self.chunk_size - 2 * margin_samples
        if step_samples <= 0:
            raise ValueError("chunk_size too small for margin.")

        all_tokens = []
        start = 0
        chunk_index = 0

        while start < L:
            real_len = min(self.chunk_size, L - start)
            chunk = signal[start:start + real_len]
            if len(chunk) < self.chunk_size:
                chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)), mode='constant')
            x = torch.from_numpy(chunk).float().unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                recon,tokens,loss,loss_breakdown = self.model(x)
            tokens = tokens.squeeze(0).cpu().numpy()
            if tokens.ndim == 2:
                tokens = tokens[:, 0]
            T_valid = (real_len + self.downsample_rate - 1) // self.downsample_rate
            kept_tokens = np.array([], dtype=np.int64)
            if chunk_index == 0:
                end_idx = T_valid - self.margin_stride_count if self.margin_stride_count > 0 else T_valid
                kept_tokens = tokens[:max(0, end_idx)]
            elif start + step_samples >= L:
                start_idx = self.margin_stride_count
                max_len = T_valid - self.margin_stride_count
                if max_len > 0:
                    kept_tokens = tokens[start_idx : start_idx + max_len]
            else:
                if self.margin_stride_count > 0 and len(tokens) > 2 * self.margin_stride_count:
                    max_len = T_valid - 2 * self.margin_stride_count
                    if max_len > 0:
                        kept_tokens = tokens[
                            self.margin_stride_count : self.margin_stride_count + max_len
                        ]
                else:
                    kept_tokens = tokens[:T_valid]

            if kept_tokens.size > 0:
                all_tokens.append(kept_tokens)

            start += step_samples
            chunk_index += 1

        if not all_tokens:
            return np.zeros(T_expected, dtype=np.int64)

        final_tokens = np.concatenate(all_tokens, axis=0)
        if len(final_tokens) > T_expected:
            final_tokens = final_tokens[:T_expected]
        elif len(final_tokens) < T_expected:
            final_tokens = np.pad(final_tokens, (0, T_expected - len(final_tokens)), constant_values=0)

        return final_tokens.astype(np.int64)


    def tokenize_signal_batched(self, 
                           signal: np.ndarray, 
                           signal_chunk_size: int, 
                           signal_chunk_overlap_size: int, 
                           max_batch_size: int) -> List[np.ndarray]:
        """
        将信号严格切分为等长块（不足长度的直接丢弃），并按批次进行推理。
        使用 extend，返回的是一维列表（所有块的 tokens 连在一起）。
        
        Args:
            signal: 输入的一维信号数组。
            signal_chunk_size: 每个块的严格大小。
            signal_chunk_overlap_size: 块之间的重叠大小。
            max_batch_size: 每个推理批次的最大块数量。
            
        Returns:
            List[np.ndarray]: 一维列表。每个元素是单个块的推理结果 (tokens)。
                             (注意：不再是二维列表，所有批次的数据都被合并到了同一个列表中)
        """
    
        if signal.ndim != 1:
            raise ValueError("Signal must be 1D.")
        
        L = len(signal)
        batched_results = [] # 初始化结果列表
        
        # --- 情况 1: 信号总长度小于一个块的大小 ---
        if L < signal_chunk_size:
            return batched_results

        # --- 情况 2: 信号足够长，进行严格切分 ---
        step_size = signal_chunk_size - signal_chunk_overlap_size
        if step_size <= 0:
            raise ValueError("signal_chunk_size must be greater than signal_chunk_overlap_size.")
        
        # 1. 第一阶段：严格切分 (不填充)
        chunks = []
        start = 0    
        while start + signal_chunk_size <= L:
            chunk = signal[start : start + signal_chunk_size]
            chunks.append(chunk)
            start += step_size

        if not chunks:
            return batched_results

        # 2. 第二阶段：批量推理
        for i in range(0, len(chunks), max_batch_size):
            batch_chunks = chunks[i : i + max_batch_size]
            
            # --- 核心推理代码 ---
            batch_np = np.array(batch_chunks)
            x = torch.from_numpy(batch_np).float().unsqueeze(1).to(self.device)
            
            with torch.no_grad():
                recon, tokens, loss, loss_breakdown = self.model(x) 
            
            tokens_np = tokens.cpu().numpy()
            if tokens_np.ndim == 3:
                tokens_np = tokens_np.squeeze(-1) # [B, T, 1] -> [B, T]
            
            # --- 修改点：使用 extend ---
            # 将当前批次中每一个块的 tokens 结果直接添加到主列表中
            # 这样做会“展平”批次结构，最终得到一个包含所有块结果的一维列表
            batched_results.extend(tokens_np)
            # -------------------------
            
        return batched_results # 返回 List[np.ndarray] (一维)


    # tokenize_data不支持任何归一化, medf, lpf等操作
    def tokenize_data(self, signal: np.ndarray) -> list:
        flat_tokens = self._tokenize_chunked_signal(signal)
        if flat_tokens.size == 0:
            return []
        parts = []
        for token_id in flat_tokens:
            parts.append(f"<|bwav:{int(token_id)}|>")
        return parts

    # tokenize_data不支持任何归一化, medf, lpf等操作
    def tokenize_chunk(self, signal: np.ndarray) -> list:
        flat_tokens = self._tokenize_chunked_signal(signal)
        if flat_tokens.size == 0:
            return []
        parts = []
        for token_id in flat_tokens:
            parts.append(f"<|bwav:{int(token_id)}|>")
        return parts



    def tokenize_read(self, read, nanopore_signal_process_strategy="apple") -> list:
        try:
            channel_info = read.handle[read.global_key + 'channel_id'].attrs
            offset = int(channel_info['offset'])
            scaling = channel_info['range'] / channel_info['digitisation']
            raw = read.handle[read.raw_dataset_name][:]
            signal_raw = np.array(scaling * (raw + offset), dtype=np.float32)
            signal_processed = nanopore_process_signal(signal_raw,nanopore_signal_process_strategy)
            return self.tokenize_data(signal_processed)
        except Exception as e:
            fast5_path = getattr(read.handle, 'filename', 'unknown.fast5')
            print(f"❌ Error on read {read.read_id} in {fast5_path}: {e}")
            return []
    

    def tokenize_fast5(self, fast5_path: str, output_path:str, nanopore_signal_process_strategy="apple"):
        print(f"✅ Processing {fast5_path} with strategy{nanopore_signal_process_strategy}")
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

    def tokenize_data_batched(self, 
                 signal: np.ndarray, 
                 signal_chunk_size: int, 
                 signal_chunk_overlap_size: int, 
                 max_batch_size: int,
                 chunk_token_count: int) -> list:
        """
        使用批量推理函数获取 tokens，并进行严格校验与拼接。
        
        Args:
            signal: 输入信号
            signal_chunk_size: 信号块大小
            signal_chunk_overlap_size: 重叠大小
            max_batch_size: 最大批大小
            chunk_token_count: 期望每个 chunk 输出的 token 数量 (用于校验)
            
        Returns:
            list: 每个元素是一个字符串，由符合长度要求的 chunk tokens 拼接而成
        """
        
        # 调用批量处理函数
        chunks_tokens_list = self.tokenize_signal_batched(
            signal=signal,
            signal_chunk_size=signal_chunk_size,       
            signal_chunk_overlap_size=signal_chunk_overlap_size, 
            max_batch_size=max_batch_size
        )
        
        # 如果没有结果，返回空列表
        if not chunks_tokens_list:
            return []

        string_parts = []
        for chunk_tokens in chunks_tokens_list:
            # --- 新增校验逻辑 ---
            # 检查当前 chunk 的 token 数量是否符合预期
            if len(chunk_tokens) != chunk_token_count:
                # 如果长度不符，可以选择跳过(this)、填充或报错
                # 这里选择跳过，不加入结果列表
                continue 
            
            # 1. 将每个 token ID 转换为字符串格式
            token_strings = [f"<|bwav:{int(token_id)}|>" for token_id in chunk_tokens]
            # 2. 使用 "".join() 拼接
            joined_string = "".join(token_strings)
            string_parts.append(joined_string)
        return string_parts

    def tokenize_read_batched(self, read, 
        nanopore_signal_process_strategy:str="apple",
        # 新增参数：用于传递给 tokenize_data
        signal_chunk_size: int = 40000,
        signal_chunk_overlap_size: int = 10000,
        max_batch_size: int = 32,
        chunk_token_count: int = 8000 # 用于内部校验
    ) -> list:
        try:
            channel_info = read.handle[read.global_key + 'channel_id'].attrs
            offset = int(channel_info['offset'])
            scaling = channel_info['range'] / channel_info['digitisation']
            raw = read.handle[read.raw_dataset_name][:]
            signal_raw = np.array(scaling * (raw + offset), dtype=np.float32)
            signal_processed = nanopore_process_signal(signal_raw,nanopore_signal_process_strategy)
            return self.tokenize_data_batched(signal_processed,signal_chunk_size,signal_chunk_overlap_size,max_batch_size,chunk_token_count)
        except Exception as e:
            fast5_path = getattr(read.handle, 'filename', 'unknown.fast5')
            print(f"❌ Error on read {read.read_id} in {fast5_path}: {e}")
            return []



    def tokenize_fast5_batched(self,
        fast5_path: str,
        output_path: str,
        nanopore_signal_process_strategy="apple",
        # 新增参数：用于传递给 tokenize_data
        signal_chunk_size: int = 40000,
        signal_chunk_overlap_size: int = 10000,
        max_batch_size: int = 32,
        chunk_token_count: int = 8000 # 用于内部校验
    ):
        print(f"✅ Processing {fast5_path} with strategy {nanopore_signal_process_strategy}")
        results = []

        with get_fast5_file(fast5_path, mode="r") as f5:
            for read in tqdm(f5.get_reads(), desc=os.path.basename(fast5_path)):
                try:
                    # 调用 tokenize_data，传入所有必要的参数
                    # 返回的是 List[str]，例如 ["<|bwav:1|><|bwav:2|>", "<|bwav:3|><|bwav:4|>"]
                    chunked_token_strings = self.tokenize_read_batched(
                        read=read,
                        signal_chunk_size=signal_chunk_size,
                        signal_chunk_overlap_size=signal_chunk_overlap_size,
                        max_batch_size=max_batch_size,
                        chunk_token_count=chunk_token_count
                    )
                    # --- 修改开始 ---
                    # 循环每个分块字符串，作为独立行追加
                    for chunk_token_str in chunked_token_strings:
                        results.append({
                            "id": read.read_id, 
                            "text": chunk_token_str
                        })
                    # --- 修改结束 ---
                    # 将多个 chunk 的字符串用空格（或其他分隔符）连接成一个完整的字符串
                    # 如果不需要分隔符，使用 ""；如果需要区分 chunk 边界，建议使用 " " 或 "<|chunk_end|>"
                except Exception as e:
                    print(f"❌ Failed on read {read.read_id}: {e}")
                    continue

        # 写入文件
        with gzip.open(output_path, 'wt', encoding='utf-8') as f:
            for item in results:
                f.write(json.dumps(item) + '\n')
        print(f"✅ Wrote {len(results)} reads to {output_path}")



