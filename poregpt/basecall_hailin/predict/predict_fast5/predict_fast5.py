#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename: basecall.py
@Description: description of this file
@Datatime: 2026/01/08 13:46:08
@Author: Hailin Pan
@Email: panhailin@genomics.cn, hailinpan1988@163.com
@Version: v1.0
'''

import re
import os
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
from ont_fast5_api.fast5_interface import get_fast5_file
import torch
from typing import Dict, Optional, Tuple, Union, Literal, List
from poregpt.tokenizers.kms_tokenizer import KMSTokenizer
import json
import gzip



def get_overlap_stat(
    chunk_size: int, 
    window_len: int, 
    stride: int, 
    suppose_overlap_len: int = 500
) -> Tuple[int, int, int]:
    """
    Calculate the overlap needed for chunking.
    给定chunk_size, window_len, stride, 计算实际的overlap长度，以及前后chunk交界处的window索引。
    这里实际的overlap长度在suppose_overlap_len附近波动，以保证前后两个chunk重叠的区域的window是对齐的。

    tail = (chunk-window) % stride  %：求余数
    overlap = tail + window + stride * (n_window - 1) 找到合适的n_window，使得overlap最接近suppose_ov=500

    n_window_hat = (suppose_ov - tail - window) // stride + 1    //:求商的整数部分
    overlap_hat = tail + window + stride * (n_window_hat - 1)
    前一个chunk截取后的结尾所在的位置pre_end = chunk - overlap_hat + window + stride * (n_window_hat//2 - 1)       1-based
    前一个chunk截取后的结尾所在的window编号pre_end_w_idx = (pre_end-window)//stride    0-based
    后一个chunk截取后的开头所在的window编号post_start_w_idx = n_window_hat//2   0-based
    后一个chunk的原始起始和前一个trunk的原始起始的window编号差diff_window_num =  (chunk - overlap_hat) // stride


    Args:
        chunk_size (int): Size of each chunk.
        window_len (int): Length of each window.
        stride (int): Stride between windows.
        suppose_overlap_len (int, optional): Expected overlap length. Defaults to 500.
    Returns:
        Tuple[int, int, int]: Actual overlap length, previous chunk end window index, next chunk start window index.
    """
    tail_len = (chunk_size - window_len) % stride
    n_window_hat = (suppose_overlap_len - tail_len - window_len) // stride + 1 # 两个相邻chunk重叠区域的window数量
    overlap_hat = tail_len + window_len + stride * (n_window_hat - 1)

    total_window_num_in_a_chunk = (chunk_size - window_len) // stride + 1
    assert n_window_hat < total_window_num_in_a_chunk, "Overlap is too large for the given chunk size."

    #前一个chunk截取后的的结尾所在的位置 1-based
    pre_end = chunk_size - overlap_hat + window_len + stride * (n_window_hat//2 - 1)
    #前一个chunk截取后的的结尾所在的window编号 0-based
    pre_end_w_idx = (pre_end - window_len) // stride
    #后一个chunk截取后的的开头所在的window编号  0-based
    post_start_w_idx = n_window_hat // 2
    #后一个chunk的原始起始和前一个trunk的原始起始的window编号差
    diff_window_num = (chunk_size - overlap_hat) // stride


    ov_stat_dict = {
        'total_window_num_in_a_chunk': total_window_num_in_a_chunk, 
        'overlap_hat': overlap_hat, 
        'pre_end_w_idx': pre_end_w_idx, 
        'post_start_w_idx': post_start_w_idx,
        'diff_window_num': diff_window_num
    }

    return ov_stat_dict



def fast5_to_word(
    model_path: str,
    fast5_files_dir: str,
    method: Literal['kmeans', 'vq'] = 'kmeans',
    output_dir: str = './',
) -> Dict:
    """
    Convert raw signals from fast5 files to word sequences using a trained tokenizer model.
    
    Args:
        model_path (str): Path to the trained tokenizer model.
        fast5_files_dir (str): Directory containing fast5 files.
        method (Literal['kmeans', 'vq'], optional): Tokenization method. Defaults to 'kmeans'.
        output_dir (str, optional): Directory to save the output JSONL.GZ files. Defaults to './'.
    Returns:
        Dict: A dictionary containing the sliding window parameters used for tokenization.
    """
    sliding_dict = {}
    if method == 'kmeans':
        tokenizer = KMSTokenizer(
            centroids_path=model_path
        )
        sliding_dict['wnidow_size'] = tokenizer.window_size
        sliding_dict['stride'] = tokenizer.stride
    elif method == 'vq':
        tokenizer = VQTokenizer(
            model_ckpt=model_path,  # 必填：预训练模型路径
            device="cuda",                                # 可选：设备，默认自动选 cuda/cpu
            token_batch_size=100                          # 可选：批处理 token 的内部 batch size, 也就是每次同时转换多少个token
        )
    else:
        raise ValueError("Method must be either 'kmeans' or 'vq'.")

    print(sliding_dict)

    # 如果output_dir不存在，则创建它
    os.makedirs(output_dir, exist_ok=True)

    fast5_files = glob.glob(fast5_files_dir + '/*.fast5')
    for fast5_file in tqdm(fast5_files, desc="Processing fast5 files"):
        filename = os.path.basename(fast5_file)
        if filename[-5:] != 'fast5':
            print(f"Skipping non-fast5 file: {filename}")
            continue
        filename_wo_ext = filename[:-5]  # 去掉.fast5后缀
        tokenizer.tokenize_fast5(
            fast5_path=fast5_file,
            output_path=f"{output_dir}/{filename_wo_ext}.jsonl.gz"
        )

    return sliding_dict


def parse_concatenated_json(text):
    """解析拼接的多个 JSON 对象"""
    decoder = json.JSONDecoder()
    pos = 0
    results = []
    while pos < len(text):
        if text[pos].strip() == '':
            pos += 1
            continue
        obj, pos = decoder.raw_decode(text, pos)
        results.append(obj)
    return results


def read_jsonl_gz(file_path: str) -> List[Dict]:
    """
    读取 JSONL.GZ 文件并返回包含所有 JSON 对象的列表。
    
    Args:
        file_path (str): JSONL.GZ 文件路径。
    
    Returns:
        List[Dict]: 包含所有 JSON 对象的列表。
    """
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        content = f.read()
    
    json_objects = parse_concatenated_json(content)
    return json_objects