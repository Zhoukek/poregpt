# vqe_train_accelerate.py
# Nanopore Signal Tokenizer Training Script with VQ-VAE
# Industrial-grade training pipeline for nanopore raw signal tokenization using Vector Quantization.
# Supports distributed training (DDP), dynamic logging, checkpointing, and independent evaluation dataset.
# This version uses Hugging Face Accelerate for simplified multi-GPU/mixed precision training.
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import csv
import time
import json
from pprint import pformat
from scipy.stats import entropy
import argparse
from typing import Dict, List, Optional
import yaml
# Import Accelerate components
from accelerate import Accelerator, DistributedType
# Relative imports from the same package
# Note: The relative imports might need adjustment depending on your package structure.
# If this script is run directly, you might need to add the parent directory to sys.path.
from .dataset import NanoporeSignalDataset
from .vqe_model_v1 import NanoporeVQEModel_V1
from .vqe_model_v2 import NanoporeVQEModel_V2
from .vqe_model_v3 import NanoporeVQEModel_V3
from accelerate import InitProcessGroupKwargs
from datetime import timedelta


def write_runtime_json(runtime_dict, filename="runtime.json"):
    """
    将字典写入指定的 json 文件。
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            # indent=4 让生成的 JSON 文件易于阅读
            # ensure_ascii=False 确保中文字符能正常显示
            json.dump(runtime_dict, f, indent=4, ensure_ascii=False)
        print(f"成功更新 {filename}")
    except Exception as e:
        print(f"写入文件时出错: {e}")

def read_runtime_json(filename="runtime.json"):
    """
    从 json 文件读取内容并返回字典。
    如果文件不存在或损坏，返回一个空字典。
    """
    if not os.path.exists(filename):
        print(f"提示: {filename} 不存在，返回空配置。")
        return None

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"错误: {filename} 格式损坏。")
        return None
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

# =============================================================================
# Utility Functions
# =============================================================================
def log_and_save(
    epoch: int,
    global_step: int,
    total_epochs: int,
    total_global_steps: int,
    train_start_time: float,
    epoch_total_steps: int,
    avg_recon_loss: float,
    avg_total_loss: float,
    avg_comit_loss: float,
    avg_diver_loss: float,
    avg_ortho_loss: float,
    avg_rperc_ratio: float,
    codebook_usage: float,
    loss_csv_path: str,
    lr: float,
    accelerator: Accelerator
):
    """
    Log training metrics to console and append to CSV for offline analysis.
    Time estimation is based on current epoch progress.
    Only the main process writes to console and CSV.
    """
    import time

    current_time = time.time()
    elapsed_seconds = current_time - train_start_time
    steps_done = global_step % total_global_steps or 1
    avg_time_per_step = elapsed_seconds / steps_done
    remaining_seconds = avg_time_per_step * max(0, total_global_steps - steps_done)

    def format_hms(seconds: float) -> str:
        seconds = int(seconds)
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"

    elapsed_str = format_hms(elapsed_seconds)
    remaining_str = format_hms(remaining_seconds)

    epoch_width = len(str(total_epochs))
    step_width = len(str(total_global_steps))

    # Only main process prints and writes to CSV
    if accelerator.is_main_process:
        print(
            f"[Epoch {epoch+1:>{epoch_width}}/{total_epochs} | "
            f"Step {global_step:>{step_width}}/{total_global_steps} | "
            f"{elapsed_str}<{remaining_str}] "
            f"Total: {avg_total_loss:>8.6f} | "
            f"Recon: {avg_recon_loss:>8.6f} | "
            f"Comit: {avg_comit_loss:>8.6f} | "
            f"Rperc: {avg_rperc_ratio:>8.6f} | "
            #f"Diver: {avg_diver_loss:>3.2f} | "
            f"Usage: {codebook_usage*100:>3.1f}% | "
            f"LR: {lr:>7.2e} |"
        )


def print_training_args(**kwargs):
    """
    Pretty-print all training hyperparameters at startup for reproducibility and debugging.
    """
    print("\n" + "="*60)
    print(" 🚀 Starting VQE Training with Accelerate. Configuration:")
    print("="*60)
    print(pformat(kwargs, width=100, sort_dicts=False))
    print("="*60 + "\n")


def save_full_checkpoint(
    path: str,
    model,
    optimizer,
    scheduler,
    epoch: int,
    spoch: int,
    global_step: int,
    cnn_type: int,
    model_type: int,
    dynamic_commitment_weight: float,
    accelerator: Accelerator
):
    """
    Save a full training checkpoint (model, optimizer, RNG states) for resuming.
    Uses Accelerate's save_state method which handles DDP/DDP sharding automatically.
    Only the main process saves the checkpoint.
    """
    # Accelerate's save_state handles distributed saving internally
    accelerator.save_state(path)
    if accelerator.is_main_process:
        print(f"✅ Full checkpoint saved to {path}")
        # Optionally save additional metadata separately
        metadata = {
            'epoch': epoch,
            'spoch': spoch,
            'global_step': global_step,
            'cnn_type': cnn_type,
            "model_type": model_type,
            'dynamic_commitment_weight': dynamic_commitment_weight,
        }
        meta_path = os.path.join(path, "metadata.json")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f)

def load_checkpoint_metadata(path: str):
    """
    Load metadata associated with a checkpoint.
    """
    # 使用 os.path.join 来安全地拼接路径
    meta_path = os.path.join(path, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            return json.load(f)
    return None



# =============================================================================
# Main Training Function
# =============================================================================

def vqe_train(
    train_npy_dir: str,
    model_type: int = 1,
    evaluation_npy_dir: Optional[str] = None,
    output_model_path: str = "nanopore_vq_tokenizer.pth",
    batch_size: int = 16, # Note: This now refers to the device_micro_batch_size
    lr: float = 1e-4,
    num_epochs: int = 10,
    codebook_size: int = 8192,
    chunk_size: int = 12000,
    num_workers: int = 8,
    update_loss_weight_every: int = 10,
    prefetch_factor: int = 128,
    val_ratio: float = 0.001,
    do_evaluate: bool = True,
    use_dynamic_commitment_weight: bool = True,
    commitment_weight: float = 1.0,
    commitment_weight_lr: float =0.01,
    commitment_weight_freeze_steps: int = 20000,
    commitment_weight_rpc: float = 1.0,
    codebook_diversity_loss_weight: float = 1.0,
    orthogonal_reg_weight: float = 1.0,
    loss_log_interval: int = 10,
    loss_csv_path: str = "train_loss.csv",
    use_wandb: bool = True,
    wandb_project: str = "nanopore_vq",
    wandb_name: str = "default_run",
    lr_scheduler_type: str = "cosine",
    warmup_steps: int = 500,
    warmup_start_factor: float = 1e-6,
    warmup_end_factor: float = 1.0,
    main_scheduler_end_factor: float = 1e-6,
    save_checkpoint_every_spoch: int = 500,
    evaluate_every_spoch: int = 10,
    checkpoint_path: Optional[str] = None,
    cnn_type: int = 0,
    init_codebook_path: Optional[str] = None,
    cnn_checkpoint_path: Optional[str] = None,
    freeze_cnn: int = 0,
    learnable_codebook: bool = False,
    global_batch_size: int = 256,
    device_micro_batch_size: int = 16, # Renamed for clarity, was 'batch_size' before
    mixed_precision: str = "bf16", # Options: "no", "fp16", "bf16", "fp8"
    gradient_clipping: float = 1.0, # Set to None to disable
    cpu: bool = False, # Set to True to force CPU training
    dataset_logic_chunk_size: int = 6000,
):
    """
    Training of Nanopore VQ tokenizer using Hugging Face Accelerate.

    Key features:
      - Distributed training handled by Accelerate (multi-GPU, multi-node).
      - Mixed precision training support (FP16, BF16).
      - Automatic gradient scaling for FP16.
      - Independent evaluation dataset via `evaluation_npy_dir`.
      - Checkpoint resume support (using Accelerate's state management).
      - Pre-trained CNN weight loading & freezing.
      - Initial codebook initialization (e.g., from Faiss).
      - WandB & CSV logging (main process only).
      - Learning rate scheduling with warmup.
      - Gradient accumulation integrated with Accelerate.

    ⚠️ NOTE ON DWA (Dynamic Weight Averager):
        The DWA module is used SOLELY for monitoring and logging purposes.
        It does NOT influence the actual loss computation or gradient updates.
        The training loss remains:
            total_loss = recon_loss + comit_loss * commitment_weight
        DWA weights are only recorded in logs/CSV/W&B for analysis.
    """
    if commitment_weight_rpc < 0.0001:
        commitment_weight_rpc = 0.0001

    print_training_args(
        train_npy_dir=train_npy_dir,
        model_type=model_type,
        evaluation_npy_dir=evaluation_npy_dir,
        output_model_path=output_model_path,
        lr=lr,
        num_epochs=num_epochs,
        codebook_size=codebook_size,
        chunk_size=chunk_size,
        num_workers=num_workers,
        update_loss_weight_every=update_loss_weight_every,
        prefetch_factor=prefetch_factor,
        val_ratio=val_ratio,
        do_evaluate=do_evaluate,
        commitment_weight=commitment_weight,
        commitment_weight_lr=commitment_weight_lr,
        codebook_diversity_loss_weight=codebook_diversity_loss_weight,
        orthogonal_reg_weight=orthogonal_reg_weight,
        loss_csv_path=loss_csv_path,
        use_wandb=use_wandb,
        use_dynamic_commitment_weight=use_dynamic_commitment_weight,
        commitment_weight_freeze_steps=commitment_weight_freeze_steps,
        commitment_weight_rpc=commitment_weight_rpc,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=warmup_steps,
        warmup_start_factor=warmup_start_factor,
        warmup_end_factor=warmup_end_factor,
        main_scheduler_end_factor=main_scheduler_end_factor,
        save_checkpoint_every_spoch=save_checkpoint_every_spoch,
        evaluate_every_spoch=evaluate_every_spoch,
        checkpoint_path=checkpoint_path,
        init_codebook_path=init_codebook_path,
        cnn_type=cnn_type,
        freeze_cnn= freeze_cnn,
        learnable_codebook = learnable_codebook,
        global_batch_size=global_batch_size,
        device_micro_batch_size=device_micro_batch_size,
        mixed_precision=mixed_precision,
        gradient_clipping=gradient_clipping,
        cpu=cpu,
        dataset_logic_chunk_size=dataset_logic_chunk_size,
    )


    # Calculate accumulation steps based on global and micro batch sizes
    effective_micro_batch = device_micro_batch_size * (1 if cpu else torch.cuda.device_count() if torch.cuda.is_available() else 1) # Estimate num_processes before accelerator init
    accumulation_steps = global_batch_size // effective_micro_batch

    if accumulation_steps == 0:
        raise ValueError(
            f"global_batch_size ({global_batch_size}) is too small for the current "
            f"device_micro_batch_size ({device_micro_batch_size}). "
            f"Minimum global_batch_size required is {effective_micro_batch}."
        )
    # 设置一个足够长的超时时间，比如 2 小时 (7200秒)
    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1800))
    # Initialize Accelerator
    # This handles device placement, distributed setup, mixed precision, and gradient accumulation automatically.
    accelerator = Accelerator(
        mixed_precision=None,
        cpu=cpu,
        gradient_accumulation_steps=accumulation_steps, # Pass the calculated steps to Acceleratoir
        log_with="wandb", # <-- Enable wandb integration
        project_dir="log", # <--- 添加这一行，提供一个目录给 accelerator 管理日志
    )
    # Initialise your wandb run, passing wandb parameters and any config information
    accelerator.init_trackers(
        project_name=wandb_project, 
        config={
                "device_micro_batch_size": device_micro_batch_size, # Changed from 'batch_size'
                "lr": lr,
                "cnn_type":cnn_type,
                "num_epochs": num_epochs,
                "codebook_size": codebook_size,
                "chunk_size": chunk_size,
                "dataset_logic_chunk_size":dataset_logic_chunk_size,
                "device_micro_batch_size":device_micro_batch_size,
                "global_batch_size":global_batch_size,
                "update_loss_weight_every": update_loss_weight_every,
                "commitment_weight": commitment_weight,
                "commitment_weight_lr": commitment_weight_lr,
                "commitment_weight_freeze_steps":commitment_weight_freeze_steps,
                "commitment_weight_rpc":commitment_weight_rpc,
                "codebook_diversity_loss_weight": codebook_diversity_loss_weight,
                "orthogonal_reg_weight": orthogonal_reg_weight,
                "world_size": accelerator.num_processes, # Changed from 'world_size'
                "mixed_precision": mixed_precision,
                "global_batch_size": global_batch_size,
        },
        init_kwargs={"wandb": {"entity": "jiaoshuaihit-hit","name":wandb_name}}
    )

    # Log accelerator info
    if accelerator.is_main_process:
        print(f"🚀 Accelerator initialized. Device: {accelerator.device}, Type: {accelerator.distributed_type}")
        print(f"   Number of processes: {accelerator.num_processes}")
        print(f"   Mixed Precision: {accelerator.mixed_precision}")
        print(f"   Global Batch Size: {global_batch_size}, Device Micro-Batch Size: {device_micro_batch_size}")
        print(f"   Gradient Accumulation Steps: {accumulation_steps}")


    # ========================
    # Data Loading
    # ========================
    # DataLoader's batch_size is now the micro-batch size per device/process.
    train_dataset = NanoporeSignalDataset(shards_dir=train_npy_dir,logic_chunk_size=dataset_logic_chunk_size,logic_chunk_overlap_size=100)
    # Accelerate provides a convenient way to create distributed samplers
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=device_micro_batch_size, # Micro-batch size per process
        shuffle=True, # Shuffling is handled by the sampler
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        drop_last=True
    )
    # Prepare dataloader with Accelerate (handles DistributedSampler internally)
    train_dataloader = accelerator.prepare(train_dataloader)

    # ========================
    # Evaluation Setup
    # ========================
    val_loader = None
    def evaluate_codebook_metrics():
        """Evaluate codebook usage, top-k concentration, and entropy on validation set."""
        if val_loader is None:
            return 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        # 关键：获取解包后的模型
        #unwrapped_model = accelerator.unwrap_model(model)
        #unwrapped_model.eval()
        model.eval()
        used_codes = set()
        token_counts = np.zeros(codebook_size, dtype=np.int64)
        total_tokens = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch # No need to call .to(device), Accelerate handles it
                # Ensure model input is correctly shaped if needed (e.g., adding batch dim)
                # Example: if x.shape is [seq_len, feat_dim] and model expects [batch, seq_len, feat_dim]
                # x = x.unsqueeze(0) # This adds the batch dimension
                # However, since we are using batches from DataLoader, x should already have a batch dimension
                # Assuming x shape is [batch_size, seq_len, feat_dim]
                
                # The model expects the correct shape; ensure your dataset/dataloader provides it.
                # If your raw data from dataset has shape [seq_len, feat_dim], DataLoader makes it [device_micro_batch_size, seq_len, feat_dim]
                # which should be fine for a model expecting [batch, seq_len, feat_dim].
                # 关键：确保输入数据类型与模型参数类型匹配
                #expected_dtype = next(unwrapped_model.parameters()).dtype
                #if x.dtype != expected_dtype:
                #    x = x.to(expected_dtype)
                print("---")  
                _, indices, _, _ = model(x) # Model input is handled by Accelerate
                print("+++")
                indices = indices.cpu().numpy().flatten()
                used_codes.update(indices.tolist())
                total_tokens += indices.size
                for idx in indices:
                    token_counts[idx] += 1

        usage_ratio = len(used_codes) / codebook_size
        top1_ratio = top3_ratio = top5_ratio = top7_ratio = top9_ratio = top10_ratio = 0.0
        entropy_val = 0.0
        max_entropy = np.log2(codebook_size)

        if total_tokens > 0:
            sorted_counts = np.sort(token_counts)[::-1]

            base_ratio = 1/codebook_size
            top1_ratio = sorted_counts[0] / total_tokens if len(sorted_counts) > 0 else 0.0
            top3_ratio = sorted_counts[2] / total_tokens if len(sorted_counts) > 2 else 0.0
            top5_ratio = sorted_counts[4] / total_tokens if len(sorted_counts) > 4 else 0.0
            top7_ratio = sorted_counts[6] / total_tokens if len(sorted_counts) > 6 else 0.0
            top9_ratio = sorted_counts[8] / total_tokens if len(sorted_counts) > 8 else 0.0

            top1_ratio = top1_ratio/base_ratio
            top3_ratio = top3_ratio/base_ratio
            top5_ratio = top5_ratio/base_ratio
            top7_ratio = top7_ratio/base_ratio
            top9_ratio = top9_ratio/base_ratio

            top10_ratio = float(sorted_counts[:min(9, codebook_size)].sum()) / total_tokens if len(sorted_counts) > 9 else 0.0
            top10_ratio = top10_ratio/base_ratio

            prob = token_counts / total_tokens
            nonzero_prob = prob[prob > 0]
            if nonzero_prob.size > 0:
                entropy_val = -np.sum(nonzero_prob * np.log2(nonzero_prob))
        #unwrapped_model.train()
        model.train()
        return (
            usage_ratio, total_tokens,
            top1_ratio, top3_ratio, top5_ratio, top7_ratio, top9_ratio, top10_ratio,
            entropy_val, max_entropy
        )

    def evaluate_codebook_metrics_v3_old():
        """Evaluate codebook usage, top-k concentration, and entropy on validation set."""
        if val_loader is None:
            return 0,0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        used_code_n = 0
        usage_ratio = 0.0
        total_tokens = 0
        top1_ratio = top3_ratio = top5_ratio = top7_ratio = top9_ratio = top10_ratio = 0.0
        entropy_val = 0.0
        max_entropy = np.log2(codebook_size) if codebook_size > 0 else 0.0
        global_recon_loss_val = 0.0
        global_comit_loss_val = 0.0

        # --- 新增：评估前的 Batch 数量同步校验 ---
        local_len = torch.tensor([len(val_loader)], device=accelerator.device)
        all_lens = accelerator.gather(local_len)
        
        if accelerator.is_main_process:
            counts = all_lens.tolist()
            print(f"\n[Eval Step {global_step}] Batch distribution: {counts}")
            if len(set(counts)) > 1:
                print(f"❌ CRITICAL WARNING: Inconsistent batch counts detected! Process may hang at gather().")
        # ---------------------------------------
        model.eval()
    
        # 存储局部结果的列表
        all_indices = []
        local_recon_loss = 0.0
        local_comit_loss = 0.0
        num_batches = 0
        # 初始化计数器 (在 GPU 上)
        token_counts = torch.zeros(codebook_size, device=accelerator.device)
        with torch.no_grad():
            # 1. 仅在主进程创建 tqdm 进度条，禁用其他进程的进度条
            # disable=not accelerator.is_main_process: 这是分布式训练的标配。它保证了只有 Rank 0 会打印那行进度条，其他 7 个进程默默干活。
            # leave=False: 验证集通常每隔一段时间跑一次，设置为 False 可以让验证完后的进度条消失，不污染你的训练日志。
            # 
            pbar = tqdm(
                val_loader,
                desc=f"Eval Step {global_step}",
                disable=not accelerator.is_main_process,
                leave=False # 评估完成后自动清除，保持终端整洁
            )
            for i, batch in enumerate(pbar):
                x = batch # 已被 accelerator 处理到正确设备
                # 使用原模型（分布式包装后的），获取当前卡上的结果
                recon, indices, _, loss_breakdown = model(x)
                recon_loss = F.mse_loss(recon, x)
                local_recon_loss += recon_loss.item()
                local_comit_loss += loss_breakdown.commitment.item()
                num_batches += 1
                # 直接累加计数，不存原始 indices
                flat_indices = indices.flatten()
                token_counts.scatter_add_(0, flat_indices, torch.ones_like(flat_indices, dtype=torch.float))
                # 建议：每 100 个 batch 同步一次，维持分布式系统的“心跳”
                #if i % 100 == 0:
                #    accelerator.wait_for_everyone()
        # 循环结束后，一次性同步计数器
        global_token_counts = accelerator.reduce(token_counts, reduction="sum")
        # --- 关键：聚合所有进程的数据 ---
        # 1. 聚合损失 (计算全局平均)
        # 将标量转换为 Tensor 以便 gather
        metrics = torch.tensor([local_recon_loss, local_comit_loss, float(num_batches)], device=accelerator.device)
        gathered_metrics = accelerator.gather(metrics).view(-1, 3) # [num_processes, 3]
        
        global_recon_loss = gathered_metrics[:, 0].sum() / gathered_metrics[:, 2].sum()
        global_comit_loss = gathered_metrics[:, 1].sum() / gathered_metrics[:, 2].sum()

        # 2. 聚合 Indices (用于计算 codebook usage 和 entropy)
        local_indices = torch.cat(all_indices)
        # 使用 gather_for_metrics 自动处理分布式采样可能产生的重复数据
        global_indices = accelerator.gather_for_metrics(local_indices)


        # --- 后期处理（仅在主进程计算统计量） ---
        if accelerator.is_main_process:
            indices_np = global_indices.cpu().numpy()
            total_tokens = indices_np.size
            
            # 计算 Codebook 统计
            token_counts = np.bincount(indices_np, minlength=codebook_size)
            used_code_n = np.count_nonzero(token_counts)
            usage_ratio = used_code_n / codebook_size
            top1_ratio = top3_ratio = top5_ratio = top7_ratio = top9_ratio = top10_ratio = 0.0
            entropy_val = 0.0
            max_entropy = np.log2(codebook_size)
            if total_tokens > 0:
                sorted_counts = np.sort(token_counts)[::-1]
                base_ratio = 1/codebook_size
                top1_ratio = sorted_counts[0] / total_tokens if len(sorted_counts) > 0 else 0.0
                top3_ratio = sorted_counts[2] / total_tokens if len(sorted_counts) > 2 else 0.0
                top5_ratio = sorted_counts[4] / total_tokens if len(sorted_counts) > 4 else 0.0
                top7_ratio = sorted_counts[6] / total_tokens if len(sorted_counts) > 6 else 0.0
                top9_ratio = sorted_counts[8] / total_tokens if len(sorted_counts) > 8 else 0.0

                top1_ratio = top1_ratio/base_ratio
                top3_ratio = top3_ratio/base_ratio
                top5_ratio = top5_ratio/base_ratio
                top7_ratio = top7_ratio/base_ratio
                top9_ratio = top9_ratio/base_ratio

                top10_ratio = float(sorted_counts[:min(9, codebook_size)].sum()) / total_tokens if len(sorted_counts) > 9 else 0.0
                top10_ratio = top10_ratio/base_ratio

                prob = token_counts / total_tokens
                nonzero_prob = prob[prob > 0]
                if nonzero_prob.size > 0:
                    entropy_val = -np.sum(nonzero_prob * np.log2(nonzero_prob))
            model.train()
            return (
                used_code_n, usage_ratio, total_tokens,
                top1_ratio, top3_ratio, top5_ratio, top7_ratio, top9_ratio, top10_ratio,
                entropy_val, max_entropy, global_recon_loss.item(), global_comit_loss.item()
            )
        else:
            model.train()
            return None
    def evaluate_codebook_metrics_v3():
        """Evaluate codebook usage, top-k concentration, and entropy on validation set."""
        if val_loader is None:
            return 0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # 初始化变量
        local_recon_loss = 0.0
        local_comit_loss = 0.0
        num_batches = 0
        # 在 GPU 上初始化计数器
        token_counts_gpu = torch.zeros(codebook_size, device=accelerator.device)

        # 校验 Batch 分布
        local_len = torch.tensor([len(val_loader)], device=accelerator.device)
        all_lens = accelerator.gather(local_len)
        if accelerator.is_main_process:
            print(f"\n[Eval Step {global_step}] Batch distribution: {all_lens.tolist()}")

        model.eval()
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Eval Step {global_step}", 
                        disable=not accelerator.is_main_process, leave=False)
            
            for i, batch in enumerate(pbar):
                x = batch
                recon, indices, _, loss_breakdown = model(x)
                
                recon_loss = F.mse_loss(recon, x)
                local_recon_loss += recon_loss.item()
                local_comit_loss += loss_breakdown.commitment.item()
                num_batches += 1
                
                # 核心优化：直接在 GPU 上统计，不需要 cat all_indices
                flat_indices = indices.flatten()
                token_counts_gpu.scatter_add_(0, flat_indices, torch.ones_like(flat_indices, dtype=torch.float))

        # --- 分布式聚合 ---
        # 1. 聚合计数器 (求和)
        global_counts_tensor = accelerator.reduce(token_counts_gpu, reduction="sum")
        
        # 2. 聚合损失
        metrics = torch.tensor([local_recon_loss, local_comit_loss, float(num_batches)], device=accelerator.device)
        gathered_metrics = accelerator.gather(metrics).view(-1, 3)

        # --- 后期处理（仅在主进程计算） ---
        if accelerator.is_main_process:
            # 计算全局平均 Loss
            total_batches_all = gathered_metrics[:, 2].sum()
            global_recon_loss = (gathered_metrics[:, 0].sum() / total_batches_all).item()
            global_comit_loss = (gathered_metrics[:, 1].sum() / total_batches_all).item()

            # 将计数器转到 CPU 供计算
            token_counts_np = global_counts_tensor.cpu().numpy()
            total_tokens = np.sum(token_counts_np)
            
            used_code_n = np.count_nonzero(token_counts_np)
            usage_ratio = used_code_n / codebook_size
            
            entropy_val = 0.0
            top1_ratio = top3_ratio = top5_ratio = top7_ratio = top9_ratio = top10_ratio = 0.0
            
            if total_tokens > 0:
                sorted_counts = np.sort(token_counts_np)[::-1]
                base_ratio = 1.0 / codebook_size
                
                # 计算 Top-N 集中度
                def get_ratio(rank):
                    return (sorted_counts[rank-1] / total_tokens) / base_ratio if len(sorted_counts) >= rank else 0.0

                top1_ratio = get_ratio(1)
                top3_ratio = get_ratio(3)
                top5_ratio = get_ratio(5)
                top7_ratio = get_ratio(7)
                top9_ratio = get_ratio(9)
                top10_ratio = (np.sum(sorted_counts[:10]) / total_tokens) / base_ratio if len(sorted_counts) >= 10 else 0.0

                # 计算熵
                prob = token_counts_np / total_tokens
                nz_prob = prob[prob > 0]
                entropy_val = -np.sum(nz_prob * np.log2(nz_prob))

            model.train()
            return (
                used_code_n, usage_ratio, int(total_tokens),
                top1_ratio, top3_ratio, top5_ratio, top7_ratio, top9_ratio, top10_ratio,
                entropy_val, np.log2(codebook_size), global_recon_loss, global_comit_loss
            )
        else:
            model.train()
            return None

    if do_evaluate and evaluation_npy_dir and os.path.isdir(evaluation_npy_dir) and val_ratio > 0: # Only setup eval on main process initially
        print(f"✅ Using independent evaluation dataset: {evaluation_npy_dir}")
        val_dataset = NanoporeSignalDataset(shards_dir=evaluation_npy_dir,logic_chunk_size=dataset_logic_chunk_size)
        actual_val_size = max(1, int(val_ratio * len(val_dataset)))
        np.random.seed(42)
        indices = np.random.choice(len(val_dataset), size=actual_val_size, replace=False)
        val_subset = torch.utils.data.Subset(val_dataset, indices)
        val_loader = DataLoader(
            val_subset,
            batch_size=device_micro_batch_size,
            shuffle=False,
            num_workers=max(2, num_workers // 2),
            pin_memory=True,
            drop_last=True
        )
        # 关键：必须 prepare，让 Accelerate 分发数据分片
        val_loader = accelerator.prepare(val_loader)
    # ========================
    # Model & Optimizer & Scheduler Preparation
    # ========================
    if model_type == 1:
        model = NanoporeVQEModel_V1(
            codebook_size=codebook_size,
            commitment_weight=commitment_weight,
            codebook_diversity_loss_weight=codebook_diversity_loss_weight,
            orthogonal_reg_weight=orthogonal_reg_weight,
            cnn_type=cnn_type,
            init_codebook_path=init_codebook_path,
            cnn_checkpoint_path = cnn_checkpoint_path,
            freeze_cnn = freeze_cnn,
            learnable_codebook=learnable_codebook
        )
    elif model_type == 2:
        model = NanoporeVQEModel_V2(
            codebook_size=codebook_size,
            commitment_weight=commitment_weight,
            codebook_diversity_loss_weight=codebook_diversity_loss_weight,
            orthogonal_reg_weight=orthogonal_reg_weight,
            cnn_type=cnn_type,
            init_codebook_path=init_codebook_path,
            cnn_checkpoint_path = cnn_checkpoint_path,
            freeze_cnn = freeze_cnn,
            learnable_codebook=learnable_codebook
        )
    elif model_type == 3:
        model = NanoporeVQEModel_V3(
            codebook_size=codebook_size,
            commitment_weight=commitment_weight,
            codebook_diversity_loss_weight=codebook_diversity_loss_weight,
            orthogonal_reg_weight=orthogonal_reg_weight,
            cnn_type=cnn_type,
            init_codebook_path=init_codebook_path,
            cnn_checkpoint_path = cnn_checkpoint_path,
            freeze_cnn = freeze_cnn,
            learnable_codebook=learnable_codebook,
        )
    else:
        print("error model type. exit")
        return 
    # No need to manually call .to(device) or wrap with DDP, Accelerate handles it
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
    )

    # Prepare model, optimizer, and scheduler with Accelerate
    model, optimizer = accelerator.prepare(model, optimizer)


    # Calculate total training steps correctly
    # len(train_dataloader) after prepare gives micro-batch count per epoch
    total_training_global_steps = num_epochs * len(train_dataloader) // accumulation_steps 

    # Scheduler setup (Accelerate requires it to be prepared after prepare(optimizer))
    scheduler = None
    # total_training_global_steps = int(len(train_dataloader) * num_epochs / global_batch_size) # <--- Remove or comment out old calculation

    if accelerator.is_main_process and lr_scheduler_type != "constant":
        print(f"📈 Using LR scheduler: {lr_scheduler_type}, warmup_steps={warmup_steps}")
        print(f"📈 Calculated total_training_global_steps: {total_training_global_steps}") # Optional: Add print to confirm
        print(f"📈 Calculated num_epochs: {num_epochs}") # Optional: Add print to confirm
        print(f"📈 Calculated accumulation_steps: {accumulation_steps}") # Optional: Add print to confirm
        print(f"调试信息:")
        print(f"  len(train_dataloader) = {len(train_dataloader)}")
        print(f"  accumulation_steps = {accumulation_steps}")
        print(f"  num_epochs = {num_epochs}")
        print(f"  计算的总步数 = {total_training_global_steps}")

    # 我们指定的warmup_steps是scheduler参数更新的次数，而不是真正的steps，所以

    if lr_scheduler_type != "constant":

        # 因为accelerate的调度和并行度有关，每次调度其实同时运行了多次
        # 所以要乘以rank数


        from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
        
        # Ensure warmup_steps does not exceed total steps to prevent errors
        actual_warmup_steps = min(warmup_steps, total_training_global_steps)
        main_steps = max(1, total_training_global_steps - actual_warmup_steps)

        scheduler_actual_warmup_steps = actual_warmup_steps * accelerator.num_processes
        scheduler_main_steps = main_steps * accelerator.num_processes

        warmup_scheduler = LinearLR(
            optimizer, 
            start_factor=warmup_start_factor, 
            end_factor=warmup_end_factor, 
            total_iters=scheduler_actual_warmup_steps
        )
        
        if lr_scheduler_type == "cosine":
            eta_min = lr * main_scheduler_end_factor # e.g., 5e-5 * 1e-6 = 5e-11
            main_scheduler = CosineAnnealingLR(
                optimizer, 
                T_max=scheduler_main_steps, 
                eta_min=eta_min
            )
        elif lr_scheduler_type == "linear":
            # Adjust end factor relative to warmup end factor for LinearLR continuity
            relative_end_factor = max(1e-8, min(1.0, main_scheduler_end_factor / warmup_end_factor)) if warmup_end_factor != 0 else main_scheduler_end_factor
            main_scheduler = LinearLR(
                optimizer, 
                start_factor=1.0, 
                end_factor=relative_end_factor, 
                total_iters=scheduler_main_steps
            )
        else:
            raise ValueError(f"Unsupported lr_scheduler_type: {lr_scheduler_type}")

        # Combine warmup and main scheduler
        scheduler = SequentialLR(
            optimizer, 
            schedulers=[warmup_scheduler, main_scheduler], 
            milestones=[scheduler_actual_warmup_steps]
        )
        
        # Prepare scheduler with Accelerate
        scheduler = accelerator.prepare(scheduler)
    # ========================
    # Resume from Checkpoint
    # ========================
    start_epoch = start_spoch = start_global_step = 0
    if checkpoint_path and os.path.isdir(checkpoint_path):
        # 1. 首先检查 metadata.json 是否存在且完整
        metadata = load_checkpoint_metadata(checkpoint_path)
        if metadata:
            # 定义恢复训练所必需的元数据键
            REQUIRED_METADATA_KEYS = {'epoch', 'spoch', 'global_step'}
            # 2. 检查 metadata 是否包含所有必需的键
            missing_keys = REQUIRED_METADATA_KEYS - metadata.keys()
            if not missing_keys:
                # 所有必需的键都存在
                try:
                    if accelerator.is_main_process:
                        print(f"📥 Loading checkpoint state and metadata from: {checkpoint_path}")
                    # 4. 解析所有必需的元数据信息
                    # 由于前面已确认键存在，这里可以直接访问
                    start_epoch = metadata['epoch'] + 1
                    start_spoch = metadata['spoch'] + 1
                    start_global_step = metadata['global_step']
                    model_type = metadata['model_type']
                    dynamic_commitment_weight = metadata.get('dynamic_commitment_weight', commitment_weight)
                    if accelerator.is_main_process:
                        print(f"✅ Successfully resumed from epoch {start_epoch}, spoch {start_spoch}, global_step {start_global_step}")
                    # 3. 加载加速器状态 (模型, 优化器等)
                    accelerator.load_state(checkpoint_path)
                    # 标记为加载成功
                    load_successful = True 
                except Exception as e:
                    # 加速器状态加载失败
                    if accelerator.is_main_process:
                        print(f"❌ Failed to load checkpoint state from {checkpoint_path}, despite valid metadata. Error: {e}. Starting from scratch.")
            else:
                # metadata.json 存在，但缺少必要的键
                if accelerator.is_main_process:
                    print(f"❌ metadata.json exists but is missing required keys: {missing_keys}. Cannot resume safely. Starting from scratch.")
        else:
            # metadata.json 文件不存在
            if accelerator.is_main_process:
                print(f"❌ metadata.json not found in {checkpoint_path}. Cannot verify checkpoint integrity. Starting from scratch.")
    else:
        if accelerator.is_main_process:
            print("⚠️ No checkpoint path provided or path is not a directory. Starting from scratch.") 




    if  accelerator.is_main_process: 
        print(f"Gradient Accumulation Steps: {accumulation_steps}")
        print(f"Num Processes (GPUs): {accelerator.num_processes}")
        print(f"actual_warmup_steps: {actual_warmup_steps}")

    # ========================
    # Gradient Accumulation Setup (Handled by Accelerator)
    # ========================
    # Calculation moved to before Accelerator init for logging
    # The accumulation_steps value is passed to Accelerator during initialization.
    # No further manual setup needed here.
    # Print configuration AFTER accelerator.prepare
    # ========================
    # Training Loop
    # ========================
    model.train()
    global_step = start_global_step
    spoch = start_spoch
    total_steps = len(train_dataloader) * num_epochs
    total_global_steps = total_training_global_steps
    total_spochs = total_steps // update_loss_weight_every
    # 在训练循环开始前
    if accelerator.is_main_process:
        print(f"\n=== 调度器详细调试 ===")
        print(f"配置值:")
        print(f"  lr: {lr}")
        print(f"  warmup_start_factor: {warmup_start_factor}")
        print(f"  warmup_end_factor: {warmup_end_factor}")
        print(f"  warmup_steps (配置): {warmup_steps}")
        print(f"  actual_warmup_steps (计算): {actual_warmup_steps}")
        
        # 检查调度器的 warmup 部分
        if hasattr(scheduler, '_schedulers'):
            warmup_sched = scheduler._schedulers[0]
            print(f"\nWarmup调度器:")
            print(f"  类型: {type(warmup_sched)}")
            print(f"  start_factor: {warmup_sched.start_factor}")
            print(f"  end_factor: {warmup_sched.end_factor}")
            print(f"  total_iters: {warmup_sched.total_iters}")
        
        print(f"\n初始学习率: {scheduler.get_last_lr()[0]:.6e}")
        print("=== 调试结束 ===\n")
    log_dict = {}

    wandb_total_tokens = 0
    wandb_eval_recon_loss = 0
    wandb_eval_comit_loss = 0
    wandb_codebook_usage = 0
    wandb_codebook_used  = 0
    wandb_codebook_top1_ratio = 0 
    wandb_codebook_top3_ratio = 0
    wandb_codebook_top5_ratio = 0
    wandb_codebook_top7_ratio = 0
    wandb_codebook_top9_ratio = 0
    wandb_codebook_top10_ratio = 0 
    wandb_codebook_entropy = 0
    wandb_codebook_max_entropy = 0


    if use_dynamic_commitment_weight:
        init_commitment_weight = commitment_weight
        dynamic_commitment_weight = init_commitment_weight
    else:
        init_commitment_weight = commitment_weight
        dynamic_commitment_weight = init_commitment_weight

    target_commitment_weight = init_commitment_weight 

    runtime_dict = {
        "target_commitment_weight":target_commitment_weight
    } 
    write_runtime_json(runtime_dict)

    train_start_time = time.time()
    for epoch in range(start_epoch, num_epochs):
        # Set epoch for the sampler if applicable (handled by Accelerate)
        train_dataloader.sampler.set_epoch(epoch) if hasattr(train_dataloader.sampler, 'set_epoch') else None

        # Zero grad at the beginning of an epoch (or more precisely, before the first accumulation cycle)
        # Accelerate's optimizer handles the zero_grad call internally when needed,
        # but for explicit control and clarity, especially with manual accumulation, it's good practice.
        # However, Accelerate often calls zero_grad automatically *before* the first backward() in a new step cycle.
        # The safest place is right before the first backward() call in a new accumulation cycle.
        # We'll handle this inside the loop based on the accumulation logic.
        # With accelerator.accumulate, zero_grad is handled automatically at the start of each accum cycle.
        for step, batch in enumerate(train_dataloader):
            # Use Accelerate's accumulate context manager
            with accelerator.accumulate(model):
                # 这里面的代码在每个 micro-batch 都会执行
                # 包括前向传播、反向传播
                # No need to manually move batch to device
                x = batch
                # Forward pass
                recon, indices, break_loss, loss_breakdown = model(x)
                recon_loss = F.mse_loss(recon, x)
                comit_loss = loss_breakdown.commitment
                diver_loss = loss_breakdown.codebook_diversity
                ortho_loss = loss_breakdown.orthogonal_reg
                # 💡 ACTUAL LOSS: Fixed weights. DWA is NOT applied here.
                total_loss = recon_loss + comit_loss * dynamic_commitment_weight

                # Scale the loss by the number of accumulation steps for averaging
                # Accelerate's backward function handles gradient scaling for mixed precision automatically
                # The loss is automatically scaled by 1/accumulation_steps inside the context if needed.
                accelerator.backward(total_loss) # Pass the unscaled loss

                # 梯度裁剪：必须在 step 之前，且在 accumulate 块内
                if accelerator.sync_gradients and gradient_clipping is not None:
                    accelerator.clip_grad_norm_(model.parameters(), gradient_clipping)
                # 只在梯度累积完成后执行优化器步骤
                #if accelerator.sync_gradients:
                # 执行梯度裁剪（如果启用）
                #if gradient_clipping is not None:
                #    accelerator.clip_grad_norm_(model.parameters(), gradient_clipping)
                # 使用 Accelerate 的方法执行优化器步骤
                optimizer.step() 
                scheduler.step()
                optimizer.zero_grad()
                
                     # 重要：这里不需要调用 optimizer.zero_grad()，Accelerate 会自动处理
            # Determine if it's time to update weights (based on effective steps)
            # We use accelerator.sync_gradients to know when the optimizer step was just performed
            
            # --- Post-update operations (logging, evaluation, checkpointing) ---
            if accelerator.sync_gradients:
                # These happen after a full parameter update
                global_step += 1 # Increment global step for every micro-step processed
                g_recon = recon_loss.item() # Get value from last micro-step
                g_comit = comit_loss.item()
                g_ortho = ortho_loss.item()
                g_diver = diver_loss.item()
                g_total = total_loss.item()

                # Log metrics (main process only)
                if accelerator.is_main_process:
                    current_lr = optimizer.param_groups[0]['lr']
                    # 为了避免log10(0)的问题，通常会给一个小的epsilon值
                    epsilon = 1e-8
                    # 计算log10值，确保非负值
                    g_recon_log10 = math.log10(g_recon + epsilon) if g_recon > 0 else math.log10(epsilon)
                    g_comit_log10 = math.log10(g_comit + epsilon) if g_comit > 0 else math.log10(epsilon)
                    g_recon_per_comit = g_recon/(g_comit + epsilon)
                    lr_log10 = math.log10(current_lr + epsilon)
                    expected_commitment_weight = 0
                    if use_dynamic_commitment_weight and global_step > commitment_weight_freeze_steps:
                        #expected_commitment_weight = target_commitment_weight
                        expected_commitment_weight = g_recon_per_comit/commitment_weight_rpc 
                        if dynamic_commitment_weight < expected_commitment_weight:
                            dynamic_commitment_weight += (expected_commitment_weight - dynamic_commitment_weight)*commitment_weight_lr

                    log_and_save(
                        epoch=epoch,
                        global_step=global_step,
                        total_epochs=num_epochs,
                        total_global_steps=total_global_steps,
                        train_start_time=train_start_time,
                        epoch_total_steps=len(train_dataloader),
                        avg_recon_loss=g_recon,
                        avg_total_loss=g_total,
                        avg_comit_loss=g_comit,
                        avg_diver_loss=g_diver,
                        avg_ortho_loss=g_ortho,
                        avg_rperc_ratio=g_recon_per_comit,
                        codebook_usage=wandb_codebook_usage, # Placeholder, updated later if eval runs
                        loss_csv_path=loss_csv_path,
                        lr=current_lr,
                        accelerator=accelerator # Pass accelerator instance
                    )
                    # Prepare log dict for WandB
                    log_dict.update({
                        "train/recon_loss": g_recon,
                        "train/recon_loss_log10": g_recon_log10,
                        "train/comit_loss": g_comit,
                        "train/comit_loss_log10": g_comit_log10,
                        "train/ortho_loss": g_ortho,
                        "train/diver_loss": g_diver,
                        "train/total_loss": g_total,
                        "comit/dynamic_commit_weight": dynamic_commitment_weight,
                        "comit/expected_commit_weight": expected_commitment_weight,
                        "train/recon_per_comit": g_recon_per_comit,
                        "evaluate/total_tokens": wandb_total_tokens, # Placeholder
                        "evaluate/recon_loss": wandb_eval_recon_loss, # Placeholder
                        "evaluate/comit_loss": wandb_eval_comit_loss, # Placeholder
                        "codebook/usage": wandb_codebook_usage, # Placeholder
                        "codebook/used_codes": wandb_codebook_used, # Placeholder
                        "codebook/entropy": wandb_codebook_entropy, # Placeholder
                        "codebook/max_entropy": wandb_codebook_max_entropy, # Placeholder
                        "topcode/top1_ratio": wandb_codebook_top1_ratio, # Placeholder
                        "topcode/top3_ratio": wandb_codebook_top3_ratio, # Placeholder
                        "topcode/top5_ratio": wandb_codebook_top5_ratio, # Placeholder
                        "topcode/top7_ratio": wandb_codebook_top7_ratio, # Placeholder
                        "topcode/top9_ratio": wandb_codebook_top9_ratio, # Placeholder
                        "topcode/topx_ratio": wandb_codebook_top10_ratio, # Placeholder
                        "learning_rate": current_lr,
                        "lr_log10": lr_log10,
                        "epoch": epoch + 1,
                        "global_step": global_step, # Useful for plotting against global steps
                    })
                    accelerator.log(log_dict, step=global_step) # <--- 添加这一行
                # --- Evaluation ---
                # Run evaluation based on spoch
                if global_step % evaluate_every_spoch == 0:
                    accelerator.wait_for_everyone() # Sync all processes before eval
                    try:
                        result = evaluate_codebook_metrics_v3()
                        if accelerator.is_main_process and result is not None:
                            (
                               wandb_codebook_used,
                               wandb_codebook_usage, 
                               wandb_total_tokens, 
                               wandb_codebook_top1_ratio, 
                               wandb_codebook_top3_ratio,
                               wandb_codebook_top5_ratio, 
                               wandb_codebook_top7_ratio, 
                               wandb_codebook_top9_ratio,
                               wandb_codebook_top10_ratio, 
                               wandb_codebook_entropy, 
                               wandb_codebook_max_entropy,
                               wandb_eval_recon_loss,
                               wandb_eval_comit_loss
                               ) = result
                            print(f"Effective Step {global_step} - Codebook Usage: {wandb_codebook_usage:.2%}")
                    except Exception as e:
                            print(f"Error during evaluation: {e}")

                    try:
                        runtime_json = read_runtime_json() 
                        if runtime_json is not None:
                            target_commitment_weight = runtime_json["target_commitment_weight"] 
                    except Exception as e:
                        print(f"Error during read runtime_json: {e}")

                    accelerator.wait_for_everyone() # Sync all processes after eval

                # Periodic checkpointing (main process only)
                if accelerator.is_main_process and global_step  % save_checkpoint_every_spoch == 0:
                    ckpt_path = f"{output_model_path}.step{global_step}.pth"
                    save_full_checkpoint(
                        path=ckpt_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler, # Pass scheduler state
                        epoch=epoch,
                        spoch=spoch,
                        global_step=global_step,
                        cnn_type=cnn_type,
                        model_type=model_type,
                        dynamic_commitment_weight=dynamic_commitment_weight,
                        accelerator=accelerator # Pass accelerator instance
                    )
                # --- End of if accelerator.sync_gradients block ---
                # --- End of with accelerator.accumulate(model) block ---
                # optimizer.step() and optimizer.zero_grad() are called automatically by Accelerate inside the context
                # when accelerator.sync_gradients becomes True (i.e., at the end of an accumulation cycle).


    # Final save (main process only)
    if accelerator.is_main_process:
        ckpt_path = f"{output_model_path}.final.pth"
        save_full_checkpoint(
            path=ckpt_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=num_epochs - 1,
            spoch=spoch,
            global_step=global_step,
            cnn_type=cnn_type,
            accelerator=accelerator
        )
    # Clean up Accelerator resources
    accelerator.end_training()


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Nanopore VQ Tokenizer with Accelerate")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Pass the configuration dictionary values to vqe_train
    # Added new Accelerate-specific arguments
    vqe_train(
        train_npy_dir=config.get("train_npy_dir"),
        model_type=config.get("model_type",1),
        evaluation_npy_dir=config.get("evaluation_npy_dir"),
        output_model_path=config.get("output_model_path", "nanopore_vq_tokenizer.pth"),
        lr=config.get("lr", 3e-4),
        num_epochs=config.get("num_epochs", 10),
        codebook_size=config.get("codebook_size", 8192),
        chunk_size=config.get("chunk_size", 12000),
        num_workers=config.get("num_workers", 8),
        val_ratio=config.get("val_ratio", 0.1),
        do_evaluate=config.get("do_evaluate", False),
        commitment_weight=config.get("commitment_weight", 0.25),
        commitment_weight_lr=config.get("commitment_weight_lr", 0.01),
        codebook_diversity_loss_weight=config.get("codebook_diversity_loss_weight", 0.0),
        orthogonal_reg_weight=config.get("orthogonal_reg_weight", 0.0),
        loss_csv_path=config.get("loss_csv_path", "train_loss.csv"),
        save_checkpoint_every_spoch=config.get("save_checkpoint_every_spoch", 10),
        loss_log_interval=config.get("loss_log_interval", 10), # Note: This logic was replaced by update_loss_weight_every
        checkpoint_path=config.get("checkpoint_path"),
        cnn_type=config.get("cnn_type", 1),
        init_codebook_path=config.get("init_codebook_path", ""),
        cnn_checkpoint_path=config.get("cnn_checkpoint_path", ""),
        freeze_cnn=config.get("freeze_cnn", 0),
        learnable_codebook=config.get("learnable_codebook", True),
        global_batch_size=config.get("global_batch_size", 256),
        device_micro_batch_size=config.get("device_micro_batch_size", 16), # Passed again for clarity
        # Accelerate specific arguments
        mixed_precision=config.get("mixed_precision", "bf16"), # Options: "no", "fp16", "bf16"
        gradient_clipping=config.get("gradient_clipping", 1.0), # Set to None to disable
        cpu=config.get("cpu", False), # Force CPU training
        # Add other parameters that might be in the YAML but not explicitly listed above
        update_loss_weight_every=config.get("update_loss_weight_every", 10),
        prefetch_factor=config.get("prefetch_factor", 128),
        use_wandb=config.get("use_wandb", True),
        use_dynamic_commitment_weight=config.get("use_dynamic_commitment_weight", True),
        commitment_weight_freeze_steps=config.get("commitment_weight_freeze_steps", 20000),
        commitment_weight_rpc=config.get("commitment_weight_rpc", 1),
        wandb_project=config.get("wandb_project", "nanopore_vq"),
        wandb_name=config.get("wandb_name", "default_run"),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        warmup_steps=config.get("warmup_steps", 100),
        warmup_start_factor=config.get("warmup_start_factor", 1e-5),
        warmup_end_factor=config.get("warmup_end_factor", 1.0),
        main_scheduler_end_factor=config.get("main_scheduler_end_factor", 1e-5),
        evaluate_every_spoch=config.get("evaluate_every_spoch", 10),
        dataset_logic_chunk_size=config.get("dataset_logic_chunk_size",6000)
    )

if __name__ == "__main__":
    main()
