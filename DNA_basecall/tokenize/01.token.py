
from ont_fast5_api.fast5_interface import get_fast5_file
import numpy as np
import json
import gzip
import json
import numpy as np
import re
from poregpt.tokenizers import VQETokenizer
from poregpt.utils import nanopore_process_signal
from tqdm import tqdm
import bonito
import matplotlib.pyplot as plt


# #### 检查chunk数据集


chunk_npy="/mnt/zzbnew/rnamodel/shenhaojie/poregpt/DNA_basecall/tokenize/data/test-zhoukexuan-model-type25-cnn_type2-teacher_model_distill_1/shuffle_npy/chunks.npy"
ref_npy="/mnt/zzbnew/rnamodel/shenhaojie/data/test-zhoukexuan-model-type25-cnn_type2-teacher_model_distill_1/references_all.npy"

chunk_npy_data=np.load(chunk_npy)
ref_npy_data=np.load(ref_npy)

print(chunk_npy_data.shape)
print(ref_npy_data.shape)

# 画出第一个 chunk
plt.figure(figsize=(12, 4))
plt.plot(chunk_npy_data[0])
plt.title(f'Chunk 0 - Shape: {chunk_npy_data[0].shape}')
plt.xlabel('Position')
plt.ylabel('Signal Value')
plt.grid(True, alpha=0.3)
plt.savefig("/mnt/zzbnew/rnamodel/zhoukexuan/poregpt/DNA_basecall/tokenize/chunk_new.png")
# plt.show()


"""
tokenizer训练完后，用于检查是否重建成功
"""



jsonl_filepath="/mnt/zzbnew/rnamodel/zhoukexuan/signalDNAmodel/test_zhoukexuan-model-typr25-cnn_type2-teacher_model/basecall/validation_00006.jsonl.gz"


with gzip.open(jsonl_filepath, 'rt', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        print(data['fast5'])
        print(data['read_id'])
        print(data['chunk_start'])
        text = data["text"]

        # 提取数字
        print(data)
        raw_tokens = list(map(int, re.findall(r'\d+', text)))
        break

fast5_filepath="/mnt/si003067jezr/default/poregpt/dataset/human_dna_004g/fast5/validation/validation_00015.fast5"

with get_fast5_file(fast5_filepath, mode="r") as f5:
    for read in f5.get_reads():
        if read.read_id=="250F701901011_55_3218_3168_976442039_234026":
            print(read.read_id)
            channel_info = read.handle[read.global_key + 'channel_id'].attrs
            offset = int(channel_info['offset'])
            scaling = channel_info['range'] / channel_info['digitisation']
            print(channel_info)
            raw = read.handle[read.raw_dataset_name][:]
            print(raw)
            signal_raw = np.array(scaling * (raw + offset), dtype=np.float32)
            print(signal_raw)


signal=signal_raw[3000:9000]

signal_lemon=nanopore_process_signal(signal_raw,strategy='lemon')

MODEL_CKPT = "/mnt/zzbnew/rnamodel/zhoukexuan/poregpt/signalDNAmodel/test-zhoukexuan-model-type25-cnn_type2-teacher_model_distill_1/encoder"

DEVICE = "cuda"
TOKEN_BATCH_SIZE = 100

tokenizer = VQETokenizer(
    model_ckpt=MODEL_CKPT,
    device=DEVICE,
    token_batch_size=TOKEN_BATCH_SIZE
)

tokens_list = tokenizer.tokenize_data(signal_lemon[3000:9000])
handle_tokens = [int(re.search(r'\d+', x).group()) for x in tokens_list]


print(len(set(handle_tokens)),len(set(raw_tokens)))

matches = [h == r for h, r in zip(handle_tokens, raw_tokens)]


raw_signal=signal_lemon[3000:9000]
recon_signal=tokenizer.decode_token_ids(raw_tokens)


def analyze_raw_recon(
    raw_signal,
    recon_signal,
    window=200,
    topk=5,
    show_global=True,
    show_top_error=True,
    show_top_smoothing=True,
):
    """
    比较 raw signal 和 recon signal。

    参数
    ----
    raw_signal : array-like
        原始信号，1D
    recon_signal : array-like
        重建信号，1D
    window : int
        滑窗大小
    topk : int
        显示 top-k 个局部片段
    show_global : bool
        是否画全局图
    show_top_error : bool
        是否画误差最大的 top-k 区域
    show_top_smoothing : bool
        是否画最可能被平滑的 top-k 区域

    返回
    ----
    result : dict
        包含全局指标、top error窗口、top smoothing窗口等
    """
    # 兼容 torch tensor
    try:
        import torch
        if isinstance(raw_signal, torch.Tensor):
            raw_signal = raw_signal.detach().cpu().numpy()
        if isinstance(recon_signal, torch.Tensor):
            recon_signal = recon_signal.detach().cpu().numpy()
    except Exception:
        pass

    raw = np.asarray(raw_signal).squeeze().astype(float)
    recon = np.asarray(recon_signal).squeeze().astype(float)

    if raw.ndim != 1 or recon.ndim != 1:
        raise ValueError(f"raw/recon 必须是一维，当前 raw={raw.shape}, recon={recon.shape}")

    n = min(len(raw), len(recon))
    if n < window:
        raise ValueError(f"信号长度 {n} 小于 window={window}")

    raw = raw[:n]
    recon = recon[:n]

    # 全局指标
    diff = recon - raw
    abs_diff = np.abs(diff)

    d_raw = np.diff(raw)
    d_recon = np.diff(recon)

    mse = np.mean(diff ** 2)
    mae = np.mean(abs_diff)
    corr = np.corrcoef(raw, recon)[0, 1] if n > 1 else np.nan
    max_abs_error = np.max(abs_diff)

    diff_mae = np.mean(np.abs(d_raw - d_recon)) if len(d_raw) > 0 else np.nan
    diff_corr = np.corrcoef(d_raw, d_recon)[0, 1] if len(d_raw) > 1 else np.nan

    raw_slope_mean = np.mean(np.abs(d_raw)) if len(d_raw) > 0 else np.nan
    recon_slope_mean = np.mean(np.abs(d_recon)) if len(d_recon) > 0 else np.nan
    slope_shrink_ratio = recon_slope_mean / (raw_slope_mean + 1e-12)

    metrics = {
        "length": n,
        "MSE": mse,
        "MAE": mae,
        "Pearson corr": corr,
        "Max abs error": max_abs_error,
        "Diff MAE": diff_mae,
        "Diff corr": diff_corr,
        "Raw slope mean": raw_slope_mean,
        "Recon slope mean": recon_slope_mean,
        "Slope shrink ratio": slope_shrink_ratio,
    }

    print("Global metrics")
    for k, v in metrics.items():
        if isinstance(v, (float, np.floating)):
            print(f"{k:18s}: {v:.6f}")
        else:
            print(f"{k:18s}: {v}")

    # 全局图
    if show_global:
        plt.figure(figsize=(16, 10))

        plt.subplot(4, 1, 1)
        plt.plot(raw, label="raw", linewidth=1)
        plt.plot(recon, label="recon", linewidth=1, alpha=0.8)
        plt.legend()
        plt.title("Raw vs Recon")

        plt.subplot(4, 1, 2)
        plt.plot(diff, linewidth=1)
        plt.axhline(0, linestyle="--", linewidth=1)
        plt.title("Difference: recon - raw")

        plt.subplot(4, 1, 3)
        plt.plot(abs_diff, linewidth=1)
        plt.title("Absolute error")

        plt.subplot(4, 1, 4)
        plt.plot(d_raw, label="diff(raw)", linewidth=1)
        plt.plot(d_recon, label="diff(recon)", linewidth=1, alpha=0.8)
        plt.legend()
        plt.title("First derivative")

        plt.tight_layout()
        plt.show()

    # 滑窗打分
    error_scores = []
    smoothing_scores = []
    starts = []

    for s in range(0, n - window + 1):
        e = s + window

        raw_seg = raw[s:e]
        recon_seg = recon[s:e]

        # 点误差
        point_score = np.mean(np.abs(raw_seg - recon_seg))

        # 平滑嫌疑：原始导数绝对值均值 - 重建导数绝对值均值
        d_raw_seg = np.diff(raw_seg)
        d_recon_seg = np.diff(recon_seg)
        smooth_score = np.mean(np.abs(d_raw_seg)) - np.mean(np.abs(d_recon_seg))

        error_scores.append(point_score)
        smoothing_scores.append(smooth_score)
        starts.append(s)

    error_scores = np.array(error_scores)
    smoothing_scores = np.array(smoothing_scores)
    starts = np.array(starts)

    # 去重选 top-k，避免大量重叠
    def pick_top_nonoverlap(scores, starts, window, topk):
        order = np.argsort(scores)[::-1]
        selected = []
        selected_ranges = []

        for idx in order:
            s = int(starts[idx])
            e = s + window

            overlap = False
            for s2, e2 in selected_ranges:
                if not (e <= s2 or e2 <= s):
                    overlap = True
                    break

            if not overlap:
                selected.append(idx)
                selected_ranges.append((s, e))
                if len(selected) >= topk:
                    break
        return selected

    top_error_idx = pick_top_nonoverlap(error_scores, starts, window, topk)
    top_smooth_idx = pick_top_nonoverlap(smoothing_scores, starts, window, topk)

    # 画 top error
    if show_top_error:
        print("\nTop error windows")
        for rank, idx in enumerate(top_error_idx, 1):
            s = int(starts[idx])
            e = s + window

            raw_seg = raw[s:e]
            recon_seg = recon[s:e]

            seg_corr = np.corrcoef(raw_seg, recon_seg)[0, 1] if len(raw_seg) > 1 else np.nan
            seg_mae = np.mean(np.abs(raw_seg - recon_seg))

            print(f"rank={rank}, start={s}, end={e}, score={error_scores[idx]:.6f}, seg_mae={seg_mae:.6f}, seg_corr={seg_corr:.6f}")

            plt.figure(figsize=(12, 4))
            plt.plot(raw_seg, label="raw", linewidth=1.2)
            plt.plot(recon_seg, label="recon", linewidth=1.2, alpha=0.8)
            plt.title(f"Top error #{rank}: {s}-{e}")
            plt.legend()
            plt.tight_layout()
            plt.show()

    # 画 top smoothing
    if show_top_smoothing:
        print("\nTop smoothing-suspect windows")
        for rank, idx in enumerate(top_smooth_idx, 1):
            s = int(starts[idx])
            e = s + window

            raw_seg = raw[s:e]
            recon_seg = recon[s:e]
            d_raw_seg = np.diff(raw_seg)
            d_recon_seg = np.diff(recon_seg)

            raw_local_slope = np.mean(np.abs(d_raw_seg))
            recon_local_slope = np.mean(np.abs(d_recon_seg))

            print(
                f"rank={rank}, start={s}, end={e}, "
                f"smooth_score={smoothing_scores[idx]:.6f}, "
                f"raw_slope={raw_local_slope:.6f}, recon_slope={recon_local_slope:.6f}"
            )

            plt.figure(figsize=(12, 6))

            plt.subplot(2, 1, 1)
            plt.plot(raw_seg, label="raw", linewidth=1.2)
            plt.plot(recon_seg, label="recon", linewidth=1.2, alpha=0.8)
            plt.title(f"Top smoothing suspect #{rank}: {s}-{e}")
            plt.legend()

            plt.subplot(2, 1, 2)
            plt.plot(d_raw_seg, label="diff(raw)", linewidth=1)
            plt.plot(d_recon_seg, label="diff(recon)", linewidth=1, alpha=0.8)
            plt.legend()
            plt.title("First derivative in local window")

            plt.tight_layout()
            plt.show()

    result = {
        "metrics": metrics,
        "raw": raw,
        "recon": recon,
        "error_scores": error_scores,
        "smoothing_scores": smoothing_scores,
        "starts": starts,
        "top_error_idx": top_error_idx,
        "top_smooth_idx": top_smooth_idx,
        "window": window,
    }

    return result


result = analyze_raw_recon(raw_signal,recon_signal, window=50, topk=5)