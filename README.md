# 🧬PoreGPT:  A GPT-based LLM for Nanopore signal

PoreGPT 是首个直接在原始 Nanopore 电流信号（raw squiggle）上进行建模的大语言模型（LLM）框架。不同于传统方法依赖碱基识别（basecalling）后的 DNA 序列，PoreGPT 通过自研的 向量量化（VQ）信号 tokenizer，将连续、高维、含噪的电信号转化为离散 token 序列，从而在信号原生空间中学习序列-结构-修饰的深层关联。
借助 Transformer 架构的强大泛化能力，PoreGPT 能够：
直接从 raw signal 预测 DNA/RNA 序列（免 basecaller）
检测表观遗传修饰（如 5mC、6mA）
重建缺失信号片段
为下游任务（如 variant calling、RNA isoform 分析）提供统一预训练表示
PoreGPT 开启了“端到端信号智能”的新范式，让大模型真正“听见”DNA 的电流之声。
> 将 Nanopore 原始电流信号（5 kHz）转换为离散 token 序列，用于下游语言模型（如 GPT）建模 DNA/RNA 序列。

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🔍 简介
本工具提供两种 Nanopore 原始电流信号（单位：pA）的 token 化方案，可将连续电流信号转换为结构化离散符号序列。适配不同建模需求：适用场景
Nanopore 信号语言建模（Signal LM）；
无参考序列的 RNA/DNA 表征学习；
多模态生物信息学分析流程（pipeline）构建。

1. RVQ Tokenizer（残差矢量量化）
基于自监督残差矢量量化（Residual VQ）模型实现信号 token 化，输出格式示例：<|bwav:L1_5336|><|bwav:L2_7466|><|bwav:L3_6973|><|bwav:L4_6340|>

核心特性：
支持 L1~L4 多层级 token 输出，可灵活适配不同粒度的建模任务；
兼容 FAST5 格式文件与原始浮点信号数组两种输入形式；
内置信号归一化 + Butterworth 滤波流程，提升 token 化鲁棒性；
支持长信号分块处理（滑动窗口 + 重叠策略），适配长序列建模场景

2. KMeans Tokenizer（Kmeans聚类）
基于 faiss K-Means 聚类算法实现信号 token 化：先通过聚类生成指定数量的聚类中心，再将电流信号切片为固定维度向量，通过匹配最相似的聚类中心向量，以聚类编号替换原始信号完成 token 化。注意：faiss keans不支持int64个向量的聚类，所以目前只支持2B（20亿）个向量的聚类。

本工具转换的格式为`<|bwav:5018|><|bwav:8156|><|bwav:23|><|bwav:5725|><|bwav:418|>`, 与后面的vq_tokenizer是一样的格式。

3. VQ tokenizer 

仅仅一层的离散化，支持将一个原始电流信号，转换为 `<|bwav:5808|><|bwav:702|><|bwav:330|><|bwav:6238|><|bwav:2432|>` 格式，其中数字从0拍到8191（在8k codebook情况下）


## ⚙️ 安装

### 从源码安装（推荐）

```bash
git clone https://github.com/lovearthai/poregpt.git
cd nanopore_signal_tokenizer
pip install -e . 
#如果阿里源报错，切换清华源或官方源 
pip install -e . \
  --no-build-isolation \
  -i https://pypi.tuna.tsinghua.edu.cn/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  --trusted-host pypi.tuna.tsinghua.edu.cn \
  --trusted-host pypi.org \
  --trusted-host files.pythonhosted.org

pip install vector-quantize-pytorch -i https://pypi.org/simple/
conda install -c pytorch faiss-gpu  
```

##  VQE_tokenizer

### 检查检查点格式

一个合格的检查点格式如下
```
.
├── metadata.json
├── model.safetensors
├── optimizer.bin
├── random_states_0.pkl
└── scheduler.bin
```
其中metadata.json内容举例如下：
```
{"epoch": 0, "spoch": 0, "global_step": 10000, "cnn_type": 7, "model_type": 3}
```

### 初始化 Tokenizer

```python
import numpy as np
from poregpt.tokenizers.vqe_tokenizer import VQETokenizer


tokenizer = VQETokenizer(
    model_ckpt="models/example_vqe_tokenizer.pth",
    token_batch_size=100
)

```

"""
##  配置参数说明
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `model_ckpt` | 预训练模型路径目录 | 必填 |
| `device` | 推理设备 | `"cuda"`,如果不填 |
"""


## 加载VQE检查点会有如下输出
```bash
✅ Using device: cuda
📂 Loading checkpoint: models/example_vqe_tokenizer.pth
🎯 Inferred: codebook_size=65536, dim=512, cnn_type=7
codebook_dim:512
Intialized VectorQuantize with the following hyperparameters:
  dim: 512
  codebook_size: 65536
  kmeans_init: True
  kmeans_iters: 10
  decay: 0.99
  threshold_ema_dead_code: 2
  commitment_weight: 1.0
  codebook_diversity_loss_weight: 1.0
  orthogonal_reg_weight: 1.0
  orthogonal_reg_max_codes: 256
  orthogonal_reg_active_codes_only: True
  cnn_type: 7
------------------------------------------------------------

✅ VQTokenizer initialized:
   Checkpoint       : /mnt/nas_syy/default/poregpt/poregpt/poregpt/test/test_vqe_tokenizer/models/example_vqe_tokenizer.pth
   Device           : cuda
   Model type       : 3
   Codebook size    : 65536
   Latent dim       : 512
   Downsample rate  : 5
   Chunk size       : 500
   Margin           : 10 samples
------------------------------------------------------------
```
注意：downsample_rate、chunk_size、margin 等参数由模型 checkpoint 自动推断，不可手动覆盖

### 对一段信号进行token化-默认开启huada归一化

```python

import numpy as np

# 模拟一段信号（单位：pA，长度任意 ≥ 25）
signal = np.random.randn(1000).astype(np.float32) * 5 + 100

# 默认返回 token 列表（每个元素是字符串，如 "<|bwav:1234|>"）
tokens_list = tokenizer.tokenize_data(signal)
print(len(tokens_list))        # 输出 token 数量
print("".join(tokens_list[:10]))  # 拼接前10个 token
```

### 对一段信号进行token化-确保在调用tokenize_data函数之前，你已经对signal作了标准化
```
# 模拟一段 1200 点的信号（~240ms @ 5kHz）
signal = np.random.randn(1000).astype(np.float32) * 5 + 100
# 获取全部层级 token (L1–L4)
tokens_list = tokenizer.tokenize_data(signal)
print(len(tokens_list))
tokens_str = "".join(tokens_list[:20])
print(tokens_str)
# <|bwav:L1_1309|><|bwav:L1_4762|><|bwav:L1_4762|><|bwav:L1_4762|><|bwav:L1_4762|><|bwav:L1_4762|><|bwav:L1_4762|><|bwav:L1_4762|><|bwav:L1_4762|><|bwav:L1_4762|><|bwav:L1_4762|><|bwav:L1_4762|><|bwav:L1_4762|><|bwav:L1_4762|><|bwav:L1_4762|><|bwav:L1_4762|><|bwav:L1_4762|><|bwav:L1_4762|><|bwav:L1_4762|><|bwav:L1_4762|>

```


### 对一个fast5文件进行token化
文件自动 gzip 压缩（.gz 后缀）
自动跳过无效或过短的 reads
支持 single-read 和 multi-read FAST5

```
#输出为 gzip 压缩的 JSONL 格式：
tokenizer.tokenize_fast5(
    fast5_path="fast5/demo.fast5",
    output_path="fast5/demo.jsonl.gz",
    nanopore_signal_process_strategy="stone"
)

```

输出的jsonl.gz内容格式

```
{"id": "read_12345", "text": "<|bwav:5808|><|bwav:702|><|bwav:330|>..."}
{"id": "read_67890", "text": "<|bwav:2432|><|bwav:812|>..."}
```

## Kmeans Tokenizer



## RVQ Tokenizer


1. 预训练模型准备
RVQ Tokenizer：将预训练 checkpoint 文件（如 nanopore_rvq_tokenizer_chunk12k.pth）放入项目 models/ 目录；
KMeans Tokenizer：直接使用 models/ 目录下的 centroids.npy 聚类中心文件。
2. 信号 Token 化示例
RVQ Tokenizer 使用示例：参考 test/ 目录下 example_rvq_tokenizer_data.py；
KMeans Tokenizer 使用示例：参考 test/ 目录下 example_kmeans_tokenize_data.py。



# basecall语料准备

## 对一批fast5文件计算对应的out_summary.processed.tsv文件和对应的references.npy文件

### out_summary.processed.tsv 这个文件里的格式如下
filename        read_id chunk_start     chunk_size      alignment_identity
validation_00003.fast5  250F701901011_23_2561_1360_397804127_558818     3000    6000    1.0
validation_00005.fast5  250F701901011_23_2561_1360_397804127_558818     6000    6000    1.0
validation_00004.fast5  250F701901011_23_2561_1360_397804127_558818     9000    6000    1.0
validation_00006.fast5  250F701901011_23_2561_1360_397804127_558818     12000   6000    1.0

### references.npy里包含相同行数的np.array. 格式如下File: references.npy
Data Type: uint8
Shape: (537111, 762)
Number of dimensions: 2
Total number of elements: 409278582
Size in bytes: 409278582

### 合并tsv和npy文件 成 .bc.csv文件

同时打开这两个文件，将npy里对应的数组读出来，把里面的0去掉。保存成12345这样的字符串， 然后增加一项 bases 用来表达这些数字字符串。 最后保存成csv: 表头是fast5, read_id, chunk_start, chunk_size, alignment_identity, bases

脚本: merge_tsv_and_npy_to_bccsv.sh 就是来执行这个合并操作， 它的功能是递归遍历一个目录下的所有子目录，如果这个子目录中有out_summary.processed.tsv和references.npy，合并成一个以子目录命名的.bc.csv文件。并且用multiprocessing_pool来并行处理。


