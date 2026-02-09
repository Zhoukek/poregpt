import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import csv
import time
from typing import Optional
import yaml
import argparse


# 相对导入模块
from .dataset import NanoporeSignalDataset
from .cnn_model import NanoporeCNNModel


# ======================================================================================
# 辅助函数：打印训练配置
# ======================================================================================
def print_training_args(**kwargs):
    """以美观格式打印所有训练超参数"""
    from pprint import pformat
    print("\n" + "=" * 60)
    print(" 🚀 Starting CNN Autoencoder Training with the following configuration:")
    print("=" * 60)
    print(pformat(kwargs, width=100, sort_dicts=False))
    print("=" * 60 + "\n")


# ======================================================================================
# 辅助函数：保存完整训练状态（模型 + 优化器 + 随机状态等）
# ======================================================================================
def save_full_checkpoint(
    path: str,
    model,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    rank: int
):
    """仅在 rank=0 时保存完整 checkpoint，避免多进程写冲突"""
    if rank != 0:
        return

    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.module.state_dict(),  # DDP 包装后需 .module
        'optimizer_state_dict': optimizer.state_dict(),
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        'numpy_rng_state': np.random.get_state(),
    }
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, path)
    print(f"✅ Full checkpoint saved to {path}")


# ======================================================================================
# 辅助函数：记录训练日志并追加到 CSV
# ======================================================================================
def log_and_save(
    epoch: int,
    step: int,
    total_epochs: int,
    total_steps: int,
    epoch_start_time: float,
    epoch_total_steps: int,
    avg_recon_loss: float,
    lr: float,
    loss_csv_path: str,
):
    """打印当前训练进度，并将损失和学习率追加到 CSV 文件"""
    current_time = time.time()
    elapsed_seconds = current_time - epoch_start_time
    steps_done = step % epoch_total_steps or 1
    avg_time_per_step = elapsed_seconds / steps_done
    remaining_seconds = avg_time_per_step * max(0, epoch_total_steps - steps_done)

    def format_hms(seconds: float) -> str:
        seconds = int(seconds)
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"

    elapsed_str = format_hms(elapsed_seconds)
    remaining_str = format_hms(remaining_seconds)

    epoch_width = len(str(total_epochs))
    step_width = len(str(total_steps))

    print(
        f"[Epoch {epoch+1:>{epoch_width}}/{total_epochs} | "
        f"Step {step:>{step_width}}/{total_steps} | "
        f"{elapsed_str}<{remaining_str}] "
        f"Recon Loss: {avg_recon_loss:>8.6f} | "
        f"LR: {lr:>7.2e} |"
    )

    # 追加训练日志到 CSV
    with open(loss_csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, step, avg_recon_loss, lr])


# ======================================================================================
# 辅助函数：在验证集上评估模型
# ======================================================================================
def validate(model, val_loader, device):
    """
    在验证集上评估模型，返回平均重建损失（MSE）
    注意：此函数应在 model.eval() 模式下调用
    """
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            x = batch.to(device)  # [B, 1, T]
            recon = model(x)
            loss = F.mse_loss(recon, x)
            val_losses.append(loss.item())
    return np.mean(val_losses)


# ======================================================================================
# 主训练函数：支持多卡 DDP + 验证 + 日志 + 断点续训
# ======================================================================================
def cnn_train(
    npy_dir: str,
    output_model_path: str,
    batch_size: int = 16,
    lr: float = 1e-4,
    num_epochs: int = 10,
    chunk_size: int = 12000,
    num_workers: int = 8,
    prefetch_factor: int = 128,
    val_ratio: float = 0.1,               # ← 关键：验证集采样比例（即使有独立 val 路径也生效）
    val_dataset_path: Optional[str] = None,  # ← 可选：独立验证集目录
    do_evaluate: bool = True,
    loss_log_interval: int = 10,
    loss_csv_path: str = "cnn_train_loss.csv",
    use_wandb: bool = True,
    wandb_project: str = "nanopore_cnn",
    wandb_name: str = "default_cnn_run",
    lr_scheduler_type: str = "cosine",
    warmup_steps: int = 1000,
    warmup_start_factor: float = 1e-6,
    warmup_end_factor: float = 1.0,
    main_scheduler_end_factor: float = 1e-5,
    save_checkpoint_every_epoch: int = 1,
    checkpoint_path: Optional[str] = None,
    cnn_type: int = 1,
):
    """
    使用 DDP 多卡训练 Nanopore 信号的 CNN 自编码器，并在每个 epoch 后进行验证。
    
    核心逻辑：
      - 优先使用 val_dataset_path 作为验证数据源；
      - **但无论来源，都只取其中 val_ratio 比例的数据用于验证**；
      - 验证仅在 rank=0 执行，避免重复计算和 I/O 冲突；
      - 训练和验证损失均记录到 CSV 和 WandB。
    """
    # 打印所有训练参数（仅主进程）
    if torch.distributed.is_available():
        print_training_args(
            npy_dir=npy_dir,
            output_model_path=output_model_path,
            batch_size=batch_size,
            lr=lr,
            num_epochs=num_epochs,
            chunk_size=chunk_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            val_ratio=val_ratio,
            val_dataset_path=val_dataset_path,
            do_evaluate=do_evaluate,
            loss_csv_path=loss_csv_path,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_name=wandb_name,
            lr_scheduler_type=lr_scheduler_type,
            warmup_steps=warmup_steps,
            cnn_type=cnn_type,
            save_checkpoint_every_epoch=save_checkpoint_every_epoch,
        )

    # ==============================
    # 初始化分布式训练环境 (DDP)
    # ==============================
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_device_id = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_device_id)
    device = f"cuda:{local_device_id}"

    # ==============================
    # 初始化 WandB（仅 rank=0）
    # ==============================
    if rank == 0 and use_wandb:
        import wandb
        wandb.init(
            project=wandb_project,
            name=wandb_name,
            config={
                "batch_size": batch_size,
                "lr": lr,
                "num_epochs": num_epochs,
                "chunk_size": chunk_size,
                "cnn_type": cnn_type,
                "world_size": world_size,
                "val_ratio": val_ratio,
                "val_dataset_path": val_dataset_path,
            }
        )
    else:
        wandb = None

    # ==============================
    # 初始化日志文件（仅 rank=0）
    # ==============================
    if rank == 0:
        print(f"🚀 Using {world_size} GPUs.")
        print(f"📂 Train Data: {npy_dir}")
        if val_dataset_path:
            print(f"🔍 External val dataset path provided: {val_dataset_path}")
        else:
            print("🔍 No external val dataset; will sample from train data.")
        print(f"💾 Final model will be saved to: {output_model_path}")

        # 创建 CSV 文件并写入表头
        with open(loss_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'step', 'recon_loss', 'lr'])

    # ==============================
    # 构建训练数据集 + DataLoader
    # ==============================
    dataset = NanoporeSignalDataset(shards_dir=npy_dir)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        drop_last=True
    )
    # ==============================
    # [DEBUG] 打印一个 batch 的输入数据统计信息（仅 rank=0）
    # ==============================
    if rank == 0:
        print("\n🔍 [DEBUG] Inspecting first batch of training data...")
        for batch in dataloader:
            x_sample = batch[0]  # 取第一个样本: [1, T]
            print(f"  Shape: {x_sample.shape}")
            print(f"  Min: {x_sample.min().item():.4f}")
            print(f"  Max: {x_sample.max().item():.4f}")
            print(f"  Mean: {x_sample.mean().item():.4f}")
            print(f"  Std: {x_sample.std().item():.4f}")
            print(f"  First 20 values: {x_sample.flatten()[:20].cpu().numpy()}")
            print(f"  Last 20 values: {x_sample.flatten()[-20:].cpu().numpy()}")
            break  # 只看第一个 batch 的第一个样本
    print("✅ Debug inspection done.\n")
    # ==============================
    # 构建验证数据集（仅 rank=0）
    # ==============================
    val_loader = None
    if do_evaluate and rank == 0:
        # Step 1: 确定验证数据来源
        if val_dataset_path and os.path.isdir(val_dataset_path) and os.listdir(val_dataset_path):
            full_val_dataset = NanoporeSignalDataset(shards_dir=val_dataset_path)
            print(f"✅ Loaded external validation dataset ({len(full_val_dataset)} chunks).")
        else:
            full_val_dataset = NanoporeSignalDataset(shards_dir=npy_dir)
            print(f"⚠️ No valid external val dataset. Using training data as fallback.")

        # Step 2: 【关键逻辑】无论来源，都按 val_ratio 采样子集
        actual_val_size = max(1, int(val_ratio * len(full_val_dataset)))
        np.random.seed(42)  # 固定随机种子，确保实验可复现
        indices = np.random.choice(len(full_val_dataset), size=actual_val_size, replace=False)
        val_dataset = torch.utils.data.Subset(full_val_dataset, indices)

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,          # 验证时不打乱
            num_workers=max(2, num_workers // 2),
            pin_memory=True
        )
        print(f"📊 Validation set size after {val_ratio:.1%} sampling: {len(val_dataset)}")

    # ==============================
    # 构建模型、优化器、调度器
    # ==============================
    model = NanoporeCNNModel(cnn_type=cnn_type).to(device)
    model = DDP(model, device_ids=[local_device_id])

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 学习率调度器
    total_training_steps = len(dataloader) * num_epochs
    scheduler = None

    if rank == 0 and lr_scheduler_type != "constant":
        print(f"📈 LR Scheduler: {lr_scheduler_type}, warmup={warmup_steps}")

    if lr_scheduler_type != "constant":
        from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
        warmup_scheduler = LinearLR(optimizer, start_factor=warmup_start_factor, end_factor=warmup_end_factor, total_iters=warmup_steps)
        main_steps = max(1, total_training_steps - warmup_steps)

        if lr_scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(optimizer, T_max=main_steps)
        elif lr_scheduler_type == "linear":
            rel_factor = max(1e-8, min(1.0, main_scheduler_end_factor / warmup_end_factor))
            main_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=rel_factor, total_iters=main_steps)
        else:
            raise ValueError(f"Unsupported scheduler: {lr_scheduler_type}")

        scheduler = SequentialLR(optimizer, [warmup_scheduler, main_scheduler], milestones=[warmup_steps])

    # ==============================
    # 加载检查点（断点续训）
    # ==============================
    start_epoch = 0
    start_global_step = 0
    if checkpoint_path and rank == 0:
        if os.path.isfile(checkpoint_path):
            print(f"📥 Loading checkpoint: {checkpoint_path}")
        else:
            print(f"⚠️ Checkpoint not found. Training from scratch.")
            checkpoint_path = None

    # 广播加载标志到所有进程
    load_flag = torch.tensor([1 if checkpoint_path else 0], dtype=torch.int32, device=device)
    if rank == 0:
        load_flag[0] = int(os.path.isfile(checkpoint_path)) if checkpoint_path else 0
    dist.broadcast(load_flag, src=0)

    if load_flag.item() == 1:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if rank == 0:
            torch.set_rng_state(ckpt['rng_state'])
            if ckpt.get('cuda_rng_state') is not None:
                torch.cuda.set_rng_state(ckpt['cuda_rng_state'])
            np.random.set_state(ckpt['numpy_rng_state'])
            start_epoch = ckpt.get('epoch', -1) + 1
            start_global_step = ckpt.get('global_step', 0)
            print(f"✅ Resuming from epoch {start_epoch}")

    # ==============================
    # 主训练循环
    # ==============================
    global_step = start_global_step
    total_steps = len(dataloader) * num_epochs

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        sampler.set_epoch(epoch)  # 确保每个 epoch 打乱不同
        model.train()

        recon_losses = []

        for step, batch in enumerate(dataloader):
            global_step += 1
            x = batch.to(device)  # [B, 1, T]

            # 前向 + 损失
            recon = model(x)
            loss = F.mse_loss(recon, x)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新学习率
            if scheduler is not None:
                scheduler.step()

            recon_losses.append(loss.item())

            # 定期记录训练日志
            if (step + 1) % loss_log_interval == 0 or step == len(dataloader) - 1:
                avg_recon = np.mean(recon_losses)
                recon_losses.clear()

                # 多卡同步平均损失
                avg_tensor = torch.tensor(avg_recon, device=device)
                dist.all_reduce(avg_tensor, op=dist.ReduceOp.AVG)
                avg_recon = avg_tensor.item()

                if rank == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    log_and_save(
                        epoch=epoch,
                        step=global_step,
                        total_epochs=num_epochs,
                        total_steps=total_steps,
                        epoch_start_time=epoch_start_time,
                        epoch_total_steps=len(dataloader),
                        avg_recon_loss=avg_recon,
                        lr=current_lr,
                        loss_csv_path=loss_csv_path,
                    )

                    if use_wandb:
                        wandb.log({
                            "train/recon_loss": avg_recon,
                            "learning_rate": current_lr,
                            "epoch": epoch + 1,
                        }, step=global_step)

        # ==============================
        # ✅ 每个 epoch 结束后执行验证
        # ==============================
        if do_evaluate and rank == 0 and val_loader is not None:
            val_loss = validate(model.module, val_loader, device)  # 注意：用 .module 解包 DDP
            current_lr = optimizer.param_groups[0]['lr']

            # 打印验证结果
            print(
                f"[Epoch {epoch+1}/{num_epochs}] "
                f"✅ Val Recon Loss: {val_loss:>8.6f} | "
                f"LR: {current_lr:>7.2e}"
            )

            # 将验证结果写入 CSV（step 列用字符串 'validation' 标记）
            with open(loss_csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, 'validation', val_loss, current_lr])

            # 记录到 WandB
            if use_wandb:
                wandb.log({
                    "val/recon_loss": val_loss,
                    "epoch": epoch + 1,
                }, step=global_step)

        # 所有进程等待 rank=0 完成验证（避免 race condition）
        dist.barrier()

        # 定期保存 checkpoint
        if rank == 0 and (epoch + 1) % save_checkpoint_every_epoch == 0:
            ckpt_path = f"{output_model_path}.epoch{epoch+1}.pth"
            save_full_checkpoint(
                path=ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                rank=rank
            )

    # ==============================
    # 保存最终模型
    # ==============================
    if rank == 0:
        save_full_checkpoint(
            path=output_model_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=num_epochs - 1,
            global_step=global_step,
            rank=rank
        )
        print(f"✅ Final model saved to {output_model_path}")
        if use_wandb:
            wandb.finish()

    # 清理分布式环境
    dist.barrier()
    dist.destroy_process_group()
def main():
    # 定义一个简单的解析器，只用于获取 config 文件路径
    parser = argparse.ArgumentParser(description="Train Nanopore Signal CNN using a YAML config file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    args, _ = parser.parse_known_args() # 解析已知参数（主要是 --config），忽略其他可能传入的参数

    # 读取 YAML 配置文件
    config_file_path = args.config # 使用命令行传入的路径
    with open(config_file_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 从配置字典中提取参数，并使用 get 设置默认值
    # 从 data 部分提取
    npy_dir = config.get('data', {}).get('npy_dir', '')
    val_dataset_path = config.get('data', {}).get('val_dataset_path', None)

    # 从 training 部分提取
    output_model_path = config.get('training', {}).get('output_model_path', "demo_nanopore_vq_tokenizer.pth")
    batch_size = config.get('training', {}).get('batch_size', 16)
    lr = config.get('training', {}).get('lr', 3e-4)
    num_epochs = config.get('training', {}).get('num_epochs', 10)
    chunk_size = config.get('training', {}).get('chunk_size', 12000)
    num_workers = config.get('training', {}).get('num_workers', 8)
    val_ratio = config.get('training', {}).get('val_ratio', 0.1)
    loss_csv_path = config.get('training', {}).get('loss_csv_path', "train_loss.csv")
    loss_log_interval = config.get('training', {}).get('loss_log_interval', 10)
    checkpoint_path = config.get('training', {}).get('checkpoint_path', "checkpoint_nanopore_vq_tokenizer.pth")
    cnn_type = config.get('training', {}).get('cnn_type', 0)
    prefetch_factor = config.get('training', {}).get('prefetch_factor', 512)

    # 从 logging 部分提取
    do_evaluate = config.get('logging', {}).get('do_evaluate', False) # 默认为 False，与 argparse 的 store_true 行为不同
    use_wandb = config.get('logging', {}).get('use_wandb', True)
    wandb_project = config.get('logging', {}).get('wandb_project', 'nanopore_cnn')
    wandb_name = config.get('logging', {}).get('wandb_name', 'default_cnn_run')

    # 从 scheduler 部分提取
    lr_scheduler_type = config.get('scheduler', {}).get('lr_scheduler_type', 'cosine')
    warmup_steps = config.get('scheduler', {}).get('warmup_steps', 1000)
    warmup_start_factor = config.get('scheduler', {}).get('warmup_start_factor', 1e-6)
    warmup_end_factor = config.get('scheduler', {}).get('warmup_end_factor', 1.0)
    main_scheduler_end_factor = config.get('scheduler', {}).get('main_scheduler_end_factor', 1e-5)

    # 从 checkpointing 部分提取
    save_checkpoint_every_epoch = config.get('checkpointing', {}).get('save_checkpoint_every_epoch', 1)

    # 调用 cnn_train 函数
    cnn_train(
        npy_dir=npy_dir,
        output_model_path=output_model_path,
        batch_size=batch_size,
        lr=lr,
        num_epochs=num_epochs,
        chunk_size=chunk_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        val_ratio=val_ratio,
        val_dataset_path=val_dataset_path,
        do_evaluate=do_evaluate,
        loss_log_interval=loss_log_interval,
        loss_csv_path=loss_csv_path,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=warmup_steps,
        warmup_start_factor=warmup_start_factor,
        warmup_end_factor=warmup_end_factor,
        main_scheduler_end_factor=main_scheduler_end_factor,
        save_checkpoint_every_epoch=save_checkpoint_every_epoch,
        checkpoint_path=checkpoint_path,
        cnn_type=cnn_type
    )

if __name__ == "__main__":
    main()

