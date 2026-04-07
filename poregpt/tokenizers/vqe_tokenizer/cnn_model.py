# cnn_model.py
"""
纯卷积自编码器（Convolutional Autoencoder）用于 Nanopore 直接 RNA 信号预训练。

该模型仅包含 encoder 和 decoder，不涉及向量量化（VQ），用于第一阶段预训练。
预训练后的 encoder 权重将被加载到后续的 VQ 模型中，以提升训练稳定性。

架构特点：
    - 输入：[B, 1, T]，T 通常为 520（对应 130 bps × 4 kHz / 1000 × 1000 ≈ 520）
    - 总下采样率：5（cnn_type=0/1）或 12（cnn_type=2）
    - 感受野：≈33（type0/1）或 ≈65（type2）采样点
    - 输出重建信号，与输入对齐

支持三种 CNN 架构：
    - cnn_type=0: 大容量非对称模型（通道 1→64→128→256）
    - cnn_type=1: 小容量严格对称模型（通道 1→16→32→64）
    - cnn_type=2: 多阶段下采样模型（通道 1→64→64→128→128→512，总 stride=12）

注意：本模型设计为**确定性重建模型**，不包含随机操作或 VQ。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Tuple

class Conv1dWithMeanChannel(nn.Module):
    """
    Conv1d层，其中第一个输出通道（索引0）是输入信号在卷积核窗口内的均值。
    其余的输出通道由标准卷积操作生成。
    注意：此版本的 in_channels 固定为 1，并使用优化的均值计算方法。
    """
    def __init__(self, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(Conv1dWithMeanChannel, self).__init__()
        self.in_channels = 1  # 固定为 1
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        if out_channels <= 0:
            raise ValueError(f"out_channels 必须为正数，得到的是 {out_channels}")

        # 创建一个专门用于计算均值的卷积层
        # 权重初始化为 1/kernel_size，使得卷积结果为平均值
        # 偏置设为 0
        self.mean_conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,  # 只需要一个输出通道来存放均值
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False # 不需要偏置
        )
        # 将权重设置为 1/kernel_size
        with torch.no_grad():
            self.mean_conv.weight.fill_(1.0 / kernel_size)

        # 我们需要至少1个通道来存放均值。如果 out_channels > 1，
        # 对其余的 (out_channels - 1) 个通道执行标准卷积。
        self.use_standard_conv = out_channels > 1
        if self.use_standard_conv:
            # 为其余 (out_channels - 1) 个通道创建标准卷积层
            self.std_conv = nn.Conv1d(1, out_channels - 1, kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        """
        前向传播函数。

        Args:
            x (torch.Tensor): 输入张量，形状为 [Batch_Size, 1, Input_Length] (因为 in_channels 固定为 1)

        Returns:
            torch.Tensor: 输出张量，形状为 [Batch_Size, out_channels, Output_Length]
                          其中第一个通道是输入的局部均值。
        """
        # --- 计算局部均值 (优化版) ---
        # 直接使用预设权重的卷积层来计算均值
        # 该卷积层的权重为 [1/kernel_size, 1/kernel_size, ..., 1/kernel_size]
        # 卷积运算自动完成了求和与除法，得到均值
        mean_channel = self.mean_conv(x) # [B, 1, L_out]

        # --- 构造最终输出 ---
        if not self.use_standard_conv:
            # 如果只需要1个输出通道，则直接返回计算出的均值通道
            return mean_channel

        # --- 如果需要更多通道 ---
        # 对输入x执行标准卷积，生成其余的 (out_channels - 1) 个通道
        std_conv_out = self.std_conv(x) # [B, out_ch - 1, L_out]

        # 将计算出的均值通道（作为第一个）与标准卷积的结果通道拼接起来
        output = torch.cat([mean_channel, std_conv_out], dim=1) # [B, out_ch, L_out]
        return output

class Conv1dWithMeanAndThresholdChannels(nn.Module):
    """
    Conv1d层，其中：
    - 第0个输出通道是输入信号在卷积核窗口内的均值 (不可导)。
    - 第1个输出通道表示输入信号在卷积核窗口内是否包含超出阈值的值 (不可导)。
      逻辑：
      - 如果窗口内存在任何值 > upper_threshold (默认2.99)，则输出 +1。
      - 如果窗口内存在任何值 < lower_threshold (默认-2.99)，则输出 -1。
      - 否则（所有值都在 [lower_threshold, upper_threshold] 范围内），输出 0。
    - 其余的输出通道由标准卷积操作生成 (可导)。
    
    注意：此版本的 in_channels 固定为 1。
    """
    def __init__(self, out_channels, kernel_size, stride=1, padding=0, bias=True, 
                 upper_threshold=2.99, lower_threshold=-2.99):
        super(Conv1dWithMeanAndThresholdChannels, self).__init__()
        self.in_channels = 1  # 固定为 1
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold

        if out_channels <= 0:
            raise ValueError(f"out_channels 必须为正数，得到的是 {out_channels}")

        # 专门用于计算均值的卷积层 (固定权重，不可导)
        self.mean_conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,  # 只需要一个输出通道来存放均值
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False # 不需要偏置
        )
        # 将权重设置为 1/kernel_size
        with torch.no_grad():
            self.mean_conv.weight.fill_(1.0 / kernel_size)
        # 冻结 mean_conv 的参数
        for p in self.mean_conv.parameters():
            p.requires_grad = False

        # 专门用于阈值判断的逻辑 (不可导)
        # 不需要额外的可训练参数

        # 我们需要至少2个通道来存放均值和阈值判断。如果 out_channels > 2，
        # 对其余的 (out_channels - 2) 个通道执行标准卷积。
        self.use_standard_conv = out_channels > 2
        if self.use_standard_conv:
            # 为其余 (out_channels - 2) 个通道创建标准卷积层
            # 这个层的参数是可训练的 (requires_grad=True by default)
            self.std_conv = nn.Conv1d(1, out_channels - 2, kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        """
        前向传播函数。

        Args:
            x (torch.Tensor): 输入张量，形状为 [Batch_Size, 1, Input_Length] (因为 in_channels 固定为 1)

        Returns:
            torch.Tensor: 输出张量，形状为 [Batch_Size, out_channels, Output_Length]
                          其中第0个通道是局部均值 (不可导)，
                               第1个通道是阈值判断结果 (1, -1, 0) (不可导)，
                               其余通道是标准卷积结果 (可导)。
        """
        # --- 计算局部均值通道 (不可导) ---
        # 直接使用预设权重的卷积层来计算均值
        mean_channel = self.mean_conv(x) # [B, 1, L_out]
        # 由于 mean_conv 的参数已冻结，且其操作是线性的，mean_channel 本身仍是计算图的一部分。
        # 但我们希望它是固定的，所以 detach 它
        mean_channel = mean_channel.detach() # [B, 1, L_out]

        # --- 计算阈值判断通道 (不可导) ---
        with torch.no_grad(): # 确保内部操作不计入计算图
            batch_size, _, seq_len = x.shape
            
            if x.size(1) != 1:
                raise ValueError(f"Expected input channels to be 1, got {x.size(1)}")
                
            # 使用 unfold 创建滑动窗口视图
            x_padded = F.pad(x, (self.padding, self.padding), mode='constant', value=0) # 手动填充
            x_unfolded = x_padded.unfold(dimension=-1, size=self.kernel_size, step=self.stride) # 再展开
            B, C, L_out, K_size = x_unfolded.shape

            # 检查窗口内是否有值 > upper_threshold 或 < lower_threshold
            gt_mask = x_unfolded > self.upper_threshold
            lt_mask = x_unfolded < self.lower_threshold

            any_gt = gt_mask.any(dim=-1) # shape: (B, 1, L_out)
            any_lt = lt_mask.any(dim=-1) # shape: (B, 1, L_out)

            # 初始化输出张量
            threshold_channel = torch.zeros((B, 1, L_out), dtype=x.dtype, device=x.device)

            # 设置满足条件的位置
            threshold_channel[any_gt] = 1.0
            threshold_channel[any_lt & ~any_gt] = -1.0 # Set -1 only if not already set to +1

            # detach 操作确保梯度不会流经这个不可导的逻辑
            threshold_channel = threshold_channel.detach() # [B, 1, L_out]

        # --- 构造最终输出 ---
        if not self.use_standard_conv:
            # 如果只需要2个输出通道，则直接返回均值和阈值判断通道
            # 顺序：[mean_channel, threshold_channel]
            output = torch.cat([mean_channel, threshold_channel], dim=1) # [B, 2, L_out]
            return output

        # --- 如果需要更多通道 ---
        # 对输入x执行标准卷积，生成其余的 (out_channels - 2) 个通道
        std_conv_out = self.std_conv(x) # [B, out_ch - 2, L_out]

        # 将不可导的均值通道、不可导的阈值判断通道与可导的标准卷积结果通道拼接起来
        # 顺序：[mean_channel, threshold_channel, std_conv_out...]
        output = torch.cat([mean_channel, threshold_channel, std_conv_out], dim=1) # [B, out_ch, L_out]

        return output

class NanoporeCNNModel(nn.Module):
    """Nanopore 信号重建用纯卷积自编码器（无 VQ）。"""

    def __init__(self, cnn_type: Literal[0, 1, 2,3,4,5,6,7,8,9,10,11,12] = 1) -> None:
        super().__init__()

        if cnn_type not in (0, 1, 2,3,4,5,6,7,8,9,10,11,12):
            raise ValueError(f"`cnn_type` must be 0, 1 or 2,3,4,5 got {cnn_type}.")

        self.cnn_type: int = cnn_type
        if cnn_type == 0:
            self._build_cnn_type0()
            self.out_channels = 256
            self.stride = 5
            self.receptive_field = 33
            self.RF = 33
        elif cnn_type == 1:
            self._build_cnn_type1()
            self.out_channels = 64
            self.stride = 5
            self.receptive_field = 33
            self.RF = 33
        elif cnn_type == 2:
            self._build_cnn_type2()
            self.out_channels = 512
            self.stride = 12
            self.receptive_field = 65
            self.RF = 65
        elif cnn_type == 3:
            self._build_cnn_type3()
            self.out_channels = 64
            self.stride = 5
            self.receptive_field = 33
            self.RF = 33
        elif cnn_type == 4:
            self._build_cnn_type4()
            self.out_channels = 32
            self.stride = 5
            self.receptive_field = 33
            self.RF = 33
        elif cnn_type == 5:
            self._build_cnn_type5()
            self.out_channels = 16
            self.stride = 5
            self.receptive_field = 33
            self.RF = 33
        elif cnn_type == 6:
            self._build_cnn_type6()
            self.out_channels = 128
            self.stride = 5
            self.receptive_field = 33
            self.RF = 33
        elif cnn_type == 7:
            self._build_cnn_type7()
            self.out_channels = 256
            self.stride = 5
            self.receptive_field = 33
            self.RF = 33
        elif cnn_type == 8:
            self._build_cnn_type8()
            self.out_channels = 512
            self.stride = 5
            self.receptive_field = 33
            self.RF = 33
        elif cnn_type == 9:
            self._build_cnn_type9()
            self.out_channels = 512
            self.stride = 8
            self.receptive_field = 49
            self.RF = 49
        elif cnn_type == 10:
            self._build_cnn_type10()
            self.out_channels = 512
            self.stride = 3
            self.receptive_field = 81
            self.RF = 81
        elif cnn_type == 11:
            self._build_cnn_type11()
            self.out_channels = 512
            self.stride = 6
            self.receptive_field = 65
            self.RF = 65
        elif cnn_type == 12:
            self._build_cnn_type12()
            self.out_channels = 512
            self.stride = 4
            self.receptive_field = 49
            self.RF = 49




    # 现代 CNN  遵循“Conv → BN → Act” 的惯例。
    # nn.SiLU()  # 等价于 x * torch.sigmoid(x)
    # 关键在于：Batch Normalization（BN）的设计假设输入是“未激活”的线性特征。
    # Conv → BatchNorm → Activation（如 SiLU/ReLU）
    # BN 的作用是标准化“线性变换后的分布”
    # BN 的目的是消除 internal covariate shift，让每一层的输入分布稳定。
    # 它假设输入来自一个近似高斯分布的线性空间（即 Conv/Wx + b 的输出）。
    # 如果你先用 SiLU（或 ReLU）非线性扭曲了分布（比如把负值压向 0，造成偏态），再做 BN：
    # BN 要去 normalize 一个高度偏斜、非对称的分布
    # 效果大打折扣，甚至可能放大噪声
    # 
    

    def _build_cnn_type0(self) -> None:
        """构建大容量 encoder（1 → 64 → 128 → 256）"""
        self.encoder = nn.Sequential(
            # Layer 1: Conv -> BN -> SiLU
            nn.Conv1d(1, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),

            # Layer 2: Conv -> BN -> SiLU
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),

            # Layer 3: 下采样层
            nn.Conv1d(128, 256, kernel_size=25, stride=5, padding=12, bias=False),
            nn.BatchNorm1d(256),
            # 注意：最后一层可以不加激活函数，或者使用SiLU保持一致性
            # Bonito 用 Tanh 是因为它要 basecall，而你要 tokenize。
            # 它牺牲信息保真度换取 basecalling 稳定性；
            # 我们必须保留完整信号信息以支持 LLM 级建模。
        )
        """构建大容量 decoder（256 → 128 → 64 → 1）"""
        self.decoder = nn.Sequential(
            # ConvTranspose -> BN -> SiLU
            nn.ConvTranspose1d(
                in_channels=256,
                out_channels=128,
                kernel_size=25,
                stride=5,
                padding=12,
                output_padding=0,
                bias=False,
            ),
            nn.BatchNorm1d(128),
            nn.SiLU(),

            # Conv -> BN -> SiLU
            nn.Conv1d(128, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),

            # 最后一层：只卷积，不加BN和激活
            nn.Conv1d(64, 1, kernel_size=5, padding=2, bias=True),
        )

    def _build_cnn_type1(self) -> None:
        """构建小容量对称 encoder（1 → 16 → 32 → 64）"""
        self.encoder = nn.Sequential(
            # Layer 1
            nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(16),
            nn.SiLU(),

            # Layer 2
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.SiLU(),

            # Layer 3: 下采样层
            nn.Conv1d(32, 64, kernel_size=25, stride=5, padding=12, bias=False),
            nn.BatchNorm1d(64),
        )
        """构建小容量对称 decoder（64 → 32 → 16 → 1）"""
        self.decoder = nn.Sequential(
            # ConvTranspose -> BN -> SiLU
            nn.ConvTranspose1d(
                in_channels=64,
                out_channels=32,
                kernel_size=25,
                stride=5,
                padding=12,
                output_padding=0,
                bias=False,
            ),
            nn.BatchNorm1d(32),
            nn.SiLU(),

            # Conv -> BN -> SiLU
            nn.Conv1d(32, 16, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(16),
            nn.SiLU(),

            # 最后一层：只卷积，不加BN和激活
            nn.Conv1d(16, 1, kernel_size=5, padding=2, bias=True),
        )

    def _build_cnn_type2(self) -> None:
        """cnn_type=2: 多阶段下采样"""
        self.encoder = nn.Sequential(
            # Layer 1
            nn.Conv1d(1, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),

            # Layer 2
            nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),

            # Layer 3: stride=3
            nn.Conv1d(64, 128, kernel_size=9, stride=3, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),

            # Layer 4: stride=2
            nn.Conv1d(128, 128, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),

            # Layer 5: stride=2
            nn.Conv1d(128, 512, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(512),
        )
        """严格对称 decoder: 512 → 128 → 128 → 64 → 64 → 1，上采样顺序与 encoder 下采样逆序对应"""
        self.decoder = nn.Sequential(
            # Inverse of encoder Layer 5: 512 → 128, upsample ×2
            nn.ConvTranspose1d(512, 128, kernel_size=5, stride=2, padding=2, output_padding=0,bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),

            # Inverse of encoder Layer 4: 128 → 128, upsample ×2
            nn.ConvTranspose1d(128, 128, kernel_size=9, stride=2, padding=4, output_padding=0,bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),

            # Inverse of encoder Layer 3: 128 → 64, upsample ×3
            nn.ConvTranspose1d(128, 64, kernel_size=9, stride=3, padding=4, output_padding=0,bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),

            # Inverse of encoder Layer 2: 64 → 64
            nn.Conv1d(64, 64, kernel_size=5, padding=2,bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),

            # Inverse of encoder Layer 1: 64 → 1
            nn.Conv1d(64, 1, kernel_size=5,padding=2,bias=True)
        )

    def _build_cnn_type3(self) -> None:
        """构建 cnn_type=1 的 encoder：1 → 16 → 32 → 64（严格对称）
        Modified: First layer has the first channel as local mean.
        """
        self.encoder = nn.Sequential(
            # Layer 1: 1 → 16, 第一个通道(kernel_size=5区域内的均值)，其余15个通道来自标准卷积
            # 注意：调用时不再需要传入 in_channels，因为它已被固定为 1
            Conv1dWithMeanChannel(out_channels=16, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(16),
            nn.SiLU(),

            # Layer 2: 16 → 32
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.SiLU(),

            # Layer 3: 32 → 64, stride=5, RF=33
            nn.Conv1d(32, 64, kernel_size=25, stride=5, padding=12, bias=False),
            nn.BatchNorm1d(64),
        )
        """构建 cnn_type=1 的 decoder（严格对称：64 → 32 → 16 → 1）"""
        self.decoder = nn.Sequential(
            # Inverse of encoder Layer 3: 64 → 32
            nn.ConvTranspose1d(
                in_channels=64,
                out_channels=32,
                kernel_size=25,
                stride=5,
                padding=12,
                output_padding=0,
                bias=False,
            ),
            nn.BatchNorm1d(32),
            nn.SiLU(),

            # Inverse of encoder Layer 2: 32 → 16
            nn.Conv1d(32, 16, kernel_size=5, padding=2,bias=False),
            nn.BatchNorm1d(16),
            nn.SiLU(),
            # Inverse of encoder Layer 1: 16 → 1
            nn.Conv1d(16, 1, kernel_size=5, padding=2,bias=True)
        )

    def _build_cnn_type4(self) -> None:
        """构建 cnn_type=4 的 encoder：1 → 8 → 16 → 32（严格对称）
        Modified: First layer has the first channel as local mean.
        """
        self.encoder = nn.Sequential(
            # Layer 1: 1 → 8, 第一个通道(kernel_size=5区域内的均值)，其余7个通道来自标准卷积
            Conv1dWithMeanChannel(out_channels=8, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(8),
            nn.SiLU(),

            # Layer 2: 8 → 16
            nn.Conv1d(8, 16, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(16),
            nn.SiLU(),

            # Layer 3: 16 → 32, stride=5, RF=33
            nn.Conv1d(16, 32, kernel_size=25, stride=5, padding=12, bias=False),
            nn.BatchNorm1d(32),
        )
        """构建 cnn_type=1 的 decoder（严格对称：32 → 16 → 8 → 1）"""
        self.decoder = nn.Sequential(
            # Inverse of encoder Layer 3: 32 → 16
            nn.ConvTranspose1d(
                in_channels=32,
                out_channels=16,
                kernel_size=25,
                stride=5,
                padding=12,
                output_padding=0,
                bias=False,
            ),
            nn.BatchNorm1d(16),
            nn.SiLU(),

            # Inverse of encoder Layer 2: 16 → 8
            nn.Conv1d(16, 8, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(8),
            nn.SiLU(),
            
            # Inverse of encoder Layer 1: 8 → 1
            nn.Conv1d(8, 1, kernel_size=5, padding=2, bias=True)
        )

    def _build_cnn_type5(self) -> None:
        """构建 cnn_type=4 的 encoder：1 → 8 → 16 → 32（严格对称）
        Modified: First layer has the first channel as local mean.
        """
        self.encoder = nn.Sequential(
            # Layer 1: 1 → 8, 第一个通道(kernel_size=5区域内的均值)，其余7个通道来自标准卷积
            Conv1dWithMeanChannel(out_channels=4, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(4),
            nn.SiLU(),

            # Layer 2: 8 → 16
            nn.Conv1d(4, 8, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(8),
            nn.SiLU(),

            # Layer 3: 16 → 32, stride=5, RF=33
            nn.Conv1d(8, 16, kernel_size=25, stride=5, padding=12, bias=False),
            nn.BatchNorm1d(16),
        )
        """构建 cnn_type=1 的 decoder（严格对称：16 → 8 → 4 → 1）"""
        self.decoder = nn.Sequential(
            # Inverse of encoder Layer 3: 32 → 16
            nn.ConvTranspose1d(
                in_channels=16,
                out_channels=8,
                kernel_size=25,
                stride=5,
                padding=12,
                output_padding=0,
                bias=False,
            ),
            nn.BatchNorm1d(8),
            nn.SiLU(),

            # Inverse of encoder Layer 2: 16 → 8
            nn.Conv1d(8, 4, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(4),
            nn.SiLU(),
            
            # Inverse of encoder Layer 1: 8 → 1
            nn.Conv1d(4, 1, kernel_size=5, padding=2, bias=True)
        )


    def _build_cnn_type6(self) -> None:
        """构建 cnn_type=1 的 encoder：1 → 16 → 32 → 64（严格对称）
        Modified: First layer has the first channel as local mean.
        """
        self.encoder = nn.Sequential(
            # Layer 1: 1 → 16, 第一个通道(kernel_size=5区域内的均值)，其余15个通道来自标准卷积
            # 注意：调用时不再需要传入 in_channels，因为它已被固定为 1
            Conv1dWithMeanChannel(out_channels=32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.SiLU(),

            # Layer 2: 16 → 32
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),

            # Layer 3: 32 → 64, stride=5, RF=33
            nn.Conv1d(64, 128, kernel_size=25, stride=5, padding=12, bias=False),
            nn.BatchNorm1d(128),
        )
        """构建 cnn_type=1 的 decoder（严格对称：64 → 32 → 16 → 1）"""
        self.decoder = nn.Sequential(
            # Inverse of encoder Layer 3: 64 → 32
            nn.ConvTranspose1d(
                in_channels=128,
                out_channels=64,
                kernel_size=25,
                stride=5,
                padding=12,
                output_padding=0,
                bias=False,
            ),
            nn.BatchNorm1d(64),
            nn.SiLU(),

            # Inverse of encoder Layer 2: 32 → 16
            nn.Conv1d(64, 32, kernel_size=5, padding=2,bias=False),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            # Inverse of encoder Layer 1: 16 → 1
            nn.Conv1d(32, 1, kernel_size=5, padding=2,bias=True)
        )


    def _build_cnn_type7(self) -> None:
        """构建 cnn_type=1 的 encoder：1 → 16 → 32 → 64（严格对称）
        Modified: First layer has the first channel as local mean.
        """
        self.encoder = nn.Sequential(
            # Layer 1: 1 → 16, 第一个通道(kernel_size=5区域内的均值)，其余15个通道来自标准卷积
            # 注意：调用时不再需要传入 in_channels，因为它已被固定为 1
            Conv1dWithMeanChannel(out_channels=64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),

            # Layer 2: 16 → 32
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),

            # Layer 3: 32 → 64, stride=5, RF=33
            nn.Conv1d(128, 256, kernel_size=25, stride=5, padding=12, bias=False),
            nn.BatchNorm1d(256),
        )
        """构建 cnn_type=1 的 decoder（严格对称：64 → 32 → 16 → 1）"""
        self.decoder = nn.Sequential(
            # Inverse of encoder Layer 3: 64 → 32
            nn.ConvTranspose1d(
                in_channels=256,
                out_channels=128,
                kernel_size=25,
                stride=5,
                padding=12,
                output_padding=0,
                bias=False,
            ),
            nn.BatchNorm1d(128),
            nn.SiLU(),

            # Inverse of encoder Layer 2: 32 → 16
            nn.Conv1d(128, 64, kernel_size=5, padding=2,bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            # Inverse of encoder Layer 1: 16 → 1
            nn.Conv1d(64, 1, kernel_size=5, padding=2,bias=True)
        )

    def _build_cnn_type8(self) -> None:
        """构建 cnn_type=1 的 encoder：1 → 16 → 32 → 64（严格对称）
        Modified: First layer has the first channel as local mean.
        """
        self.encoder = nn.Sequential(
            # Layer 1: 1 → 16, 第一个通道(kernel_size=5区域内的均值)，其余15个通道来自标准卷积
            # 注意：调用时不再需要传入 in_channels，因为它已被固定为 1
            Conv1dWithMeanChannel(out_channels=128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),

            # Layer 2: 16 → 32
            nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(256),
            nn.SiLU(),

            # Layer 3: 32 → 64, stride=5, RF=33
            nn.Conv1d(256, 512, kernel_size=25, stride=5, padding=12, bias=False),
            nn.BatchNorm1d(512),
        )
        """构建 cnn_type=1 的 decoder（严格对称：64 → 32 → 16 → 1）"""
        self.decoder = nn.Sequential(
            # Inverse of encoder Layer 3: 64 → 32
            nn.ConvTranspose1d(
                in_channels=512,
                out_channels=256,
                kernel_size=25,
                stride=5,
                padding=12,
                output_padding=0,
                bias=False,
            ),
            nn.BatchNorm1d(256),
            nn.SiLU(),

            # Inverse of encoder Layer 2: 32 → 16
            nn.Conv1d(256, 128, kernel_size=5, padding=2,bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            # Inverse of encoder Layer 1: 16 → 1
            nn.Conv1d(128, 1, kernel_size=5, padding=2,bias=True)
        )
    
    # 末尾加SiLU,是为了送给VQ
    # 如果最后一层没有激活函数
    # 这可能是因为原始设计是用于纯自编码器的预训练，其目标是确定性地重建信号，此时保留原始的线性输出可能有助于保持信息的保真度。
    # 当模型用于 VQ 时，其角色发生了变化。它不再是独立的自编码器，而是 VQ 模型的一个组件，其输出是 VQ 的输入。
    # 此时，为了优化整个 VQ 系统，对编码器的输出进行一定的预处理（即通过一个激活函数）是合理且常见的。
    # VQ 的核心思想是将连续的潜在向量（z）映射到离散的嵌入向量（e）空间。这个过程依赖于计算输入向量与代码本（codebook）中各个嵌入向量的距离。
    # 如果编码器的输出 z 包含大量负值，这会使得代码本的训练变得更加困难。
    # 因为距离计算（通常是欧氏距离）会受到正负值混合的影响，使得代码本需要学习如何在正负值混合的空间中进行有效的聚类。
    # 激活函数的作用：
    # nn.SiLU(x) = x * sigmoid(x) 或 nn.ReLU(x) = max(0, x) 等激活函数可以将特征映射到一个非负或半非负的范围（SiLU 会将大的负值推向 0，i
    # 而 ReLU 直接将所有负值置为 0）。
    # 好的，我们可以通过具体的数值例子来直观地理解激活函数（如 ReLU 和 SiLU）如何改变特征分布，以及这对向量量化（VQ）的代码本（codebook）学习有何益处。
    # 核心概念回顾:
    # VQ 的核心是计算输入向量 z 和代码本中嵌入向量 e_k 之间的欧氏距离 ||z - e_k||²。最小距离决定了 z 被映射到哪个离散的嵌入 e_k
    # 例子 1：ReLU 激活函数的作用
    # 假设输入：一个非常简单的输入向量 z_before_relu = [-2.0, -0.5, 0.1, 1.0]
    # 应用 ReLU：ReLU(x) = max(0, x) 会将所有负值变为 0
    # z_after_relu = ReLU(z_before_relu) = [0.0, 0.0, 0.1, 1.0]
    # 对 VQ 的影响：
    # 特征空间的改变：原本在四维空间 [-2.0, -0.5, 0.1, 1.0] 附近寻找最佳匹配的嵌入向量，现在变成了在 [0.0, 0.0, 0.1, 1.0] 附近寻找。
    # 这个新点完全位于第一卦限（所有维度都非负）的边界上。
    # 简化代码本学习：代码本现在不需要学习如何在负值区域进行有效的聚类。
    # 它可以专注于学习一个在非负空间内有意义的离散表示集合。
    # 例如，代码本可以学习 [0.0, 0.0, 0.0, 0.0] 作为一个表示“静默”或“背景”的嵌入，而 [0.0, 0.0, 0.5, 1.0] 可能表示一种特定的信号模式。
    # 这使得代码本的嵌入向量更容易组织和学习。
    # 例子 2：SiLU 激活函数的作用
    # 假设输入：z_before_silu = [-2.0, -0.5, 0.1, 1.0]
    # 应用 SiLU：
    # SiLU(-2.0) = -2.0 * sigmoid(-2.0) = -2.0 * (1 / (1 + exp(2))) ≈ -2.0 * 0.12 ≈ -0.24
    # SiLU(-0.5) = -0.5 * sigmoid(-0.5) = -0.5 * (1 / (1 + exp(0.5))) ≈ -0.5 * 0.38 ≈ -0.19
    # SiLU(0.1) = 0.1 * sigmoid(0.1) = 0.1 * (1 / (1 + exp(-0.1))) ≈ 0.1 * 0.52 ≈ 0.05
    # SiLU(1.0) = 1.0 * sigmoid(1.0) = 1.0 * (1 / (1 + exp(-1))) ≈ 1.0 * 0.73 ≈ 0.73
    # z_after_silu = [-0.24, -0.19, 0.05, 0.73]
    # 对 VQ 的影响：
    # 软性处理负值：与 ReLU 的硬性截断不同，SiLU 对负值进行了“软性”处理。大的负值（如 -2.0）被显著削弱（变为 -0.24），而接近零或为正的值则被保留或轻微缩放。
    # 特征空间的“收缩”：SiLU 将整个特征空间向原点方向“拉近”。对于 VQ 来说，这意味着输入 z 的动态范围变小了，代码本需要覆盖的范围也相应缩小，这使得学习过程更稳定。
    # 保持信息：与 ReLU 彻底丢失负值信息不同，SiLU 保留了关于负值强度的信息（虽然削弱了），这可能对某些细微的信号模式识别是有用的。
    # 假设我们有一个简化的二维情况，代码本中有一个嵌入向量 e_k = [0.2, 0.3]。
    # 没有激活函数的输入 z1：z1 = [-1.0, 1.5]
    #   距离 d1 = ||z1 - e_k||² = ||[-1.0 - 0.2, 1.5 - 0.3]||² = ||[-1.2, 1.2]||² = 1.44 + 1.44 = 2.88
    # ● 经过 ReLU 的输入 z2：z2 = ReLU([-1.0, 1.5]) = [0.0, 1.5]
    # ○ 距离 d2 = ||z2 - e_k||² = ||[0.0 - 0.2, 1.5 - 0.3]||² = ||[-0.2, 1.2]||² = 0.04 + 1.44 = 1.48
    # ● 经过 SiLU 的输入 z3：z3 = SiLU([-1.0, 1.5]) ≈ [-0.27, 1.2]
    # ○ 距离 d3 = ||z3 - e_k||² = ||[-0.27 - 0.2, 1.2 - 0.3]||² = ||[-0.47, 0.9]||² = 0.22 + 0.81 = 1.03

    # 结论：
    # 原始的 z1 包含负值，导致它与嵌入 e_k 的距离非常远 (2.88)。
    # 经过 ReLU 或 SiLU 处理后，新的输入 z2 和 z3 与 e_k 的距离都显著减小 (1.48 和 1.03)。
    # 这意味着代码本学习到的嵌入向量 e_k 现在可以更容易地“吸引”和代表那些经过激活函数处理后的特征向量。
    # 这种“距离的规范化”使得 VQ 的学习过程更加高效和稳定，因为代码本不必在巨大的、充满负值的特征空间中艰难地寻找最优匹配。

    # 个更优的折衷方案：不在 CNN 的最后一层加激活函数，而是在 VQ 之前加。
    # 
    def _build_cnn_type9(self) -> None:
        """cnn_type=2: 多阶段下采样，总 stride=12，输出通道=512"""
        self.encoder = nn.Sequential(
            # Layer 1: 1 → 64, stride=1
            nn.Conv1d(1, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),

            # Layer 2: 64 → 64, stride=1
            nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),

            # Layer 3: 64 → 128, stride=2
            nn.Conv1d(64, 128, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),

            # Layer 4: 128 → 128, stride=2
            nn.Conv1d(128, 128, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),

            # Layer 5: 128 → 512, stride=2
            nn.Conv1d(128, 512, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(512),
        )
        """严格对称 decoder: 512 → 128 → 128 → 64 → 64 → 1，上采样顺序与 encoder 下采样逆序对应"""
        self.decoder = nn.Sequential(
            # Inverse of encoder Layer 5: 512 → 128, upsample ×2
            nn.ConvTranspose1d(512, 128, kernel_size=5, stride=2, padding=2, output_padding=1,bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),

            # Inverse of encoder Layer 4: 128 → 128, upsample ×2
            nn.ConvTranspose1d(128, 128, kernel_size=9, stride=2, padding=4, output_padding=1,bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),

            # Inverse of encoder Layer 3: 128 → 64, upsample ×2
            nn.ConvTranspose1d(128, 64, kernel_size=9, stride=2, padding=4, output_padding=1,bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            # Inverse of encoder Layer 2: 64 → 64
            nn.Conv1d(64, 64, kernel_size=5, padding=2,bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            # Inverse of encoder Layer 1: 64 → 1
            nn.Conv1d(64, 1, kernel_size=5,padding=2,bias=True)
        )
    def _build_cnn_type10(self) -> None:
        """构建 cnn_type=1 的 encoder：1 → 16 → 32 → 64（严格对称）
        Modified: First layer has the first channel as local mean.
        """
        self.encoder = nn.Sequential(
            # Layer 1: 1 → 16, 第一个通道(kernel_size=5区域内的均值)，其余15个通道来自标准卷积
            # 注意：调用时不再需要传入 in_channels，因为它已被固定为 1
            Conv1dWithMeanChannel(out_channels=128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),

            # Layer 2: 16 → 32
            nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(256),
            nn.SiLU(),

            # Layer 3: 32 → 64, stride=5, RF=33
            nn.Conv1d(256, 512, kernel_size=25, stride=3, padding=12, bias=False),
            nn.BatchNorm1d(512),
        )
        """构建 cnn_type=1 的 decoder（严格对称：64 → 32 → 16 → 1）"""
        self.decoder = nn.Sequential(
            # Inverse of encoder Layer 3: 64 → 32
            nn.ConvTranspose1d(
                in_channels=512,
                out_channels=256,
                kernel_size=25,
                stride=3,
                padding=12,
                output_padding=0,
                bias=False,
            ),
            nn.BatchNorm1d(256),
            nn.SiLU(),

            # Inverse of encoder Layer 2: 32 → 16
            nn.Conv1d(256, 128, kernel_size=5, padding=2,bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            # Inverse of encoder Layer 1: 16 → 1
            nn.Conv1d(128, 1, kernel_size=5, padding=2,bias=True)
        )

    def _build_cnn_type11(self) -> None:
        """cnn_type=2: 多阶段下采样，总 stride=12，输出通道=512"""
        self.encoder = nn.Sequential(
            # Layer 1: 1 → 64, stride=1
            nn.Conv1d(1, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),

            # Layer 2: 64 → 64, stride=1
            nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),

            # Layer 3: 64 → 128, stride=2
            nn.Conv1d(64, 128, kernel_size=9, stride=3, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),

            # Layer 4: 128 → 128, stride=2
            nn.Conv1d(128, 128, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),

            # Layer 5: 128 → 512, stride=2
            nn.Conv1d(128, 512, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(512),
        )
        """严格对称 decoder: 512 → 128 → 128 → 64 → 64 → 1，上采样顺序与 encoder 下采样逆序对应"""
        self.decoder = nn.Sequential(
            # Inverse of encoder Layer 5: 512 → 128, upsample ×2
            nn.ConvTranspose1d(512, 128, kernel_size=5, stride=1, padding=2, output_padding=0, bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),

            # Inverse of encoder Layer 4: 128 → 128, upsample ×2
            nn.ConvTranspose1d(128, 128, kernel_size=9, stride=2, padding=4, output_padding=1,bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),

            # Inverse of encoder Layer 3: 128 → 64, upsample ×2
            nn.ConvTranspose1d(128, 64, kernel_size=9, stride=3, padding=4, output_padding=2,bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            # Inverse of encoder Layer 2: 64 → 64
            nn.Conv1d(64, 64, kernel_size=5, padding=2,bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            # Inverse of encoder Layer 1: 64 → 1
            nn.Conv1d(64, 1, kernel_size=5,padding=2,bias=True)
        )

    def _build_cnn_type12(self) -> None:
        """cnn_type=2_stride4: 总 stride=4 (1,1,2,2,1)"""
        self.encoder = nn.Sequential(
            # Layer 1: stride=1, 提取基础波形特征
            nn.Conv1d(1, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),
        
            # Layer 2: stride=1, 进一步平滑
            nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),
        
            # Layer 3: stride=2, 第一次下采样
            nn.Conv1d(64, 128, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),
        
            # Layer 4: stride=2, 第二次下采样
            nn.Conv1d(128, 128, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),
        
            # Layer 5: stride=1, 高维特征融合 (512通道)
            nn.Conv1d(128, 512, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(512),
        )
        
        """严格对称 decoder"""
        self.decoder = nn.Sequential(
            # L5 Inverse: 512 -> 128, stride 1
            nn.Conv1d(512, 128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),
        
            # L4 Inverse: 128 -> 128, upsample x2
            nn.ConvTranspose1d(128, 128, kernel_size=9, stride=2, padding=4, output_padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),
        
            # L3 Inverse: 128 -> 64, upsample x2
            nn.ConvTranspose1d(128, 64, kernel_size=9, stride=2, padding=4, output_padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),
        
            # L2 Inverse: 64 -> 64
            nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),
        
            # L1 Inverse: 64 -> 1
            nn.Conv1d(64, 1, kernel_size=5, stride=1, padding=2, bias=True)
        )
        
    # 来自bonito的DNA R9.4.1 卷积层


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(f"Expected input shape [B, 1, T], got {x.shape}")

        z = self.encoder(x)
        recon = self.decoder(z)
        
        # 对齐长度
        target_len = x.shape[-1]
        current_len = recon.shape[-1]
        
        if current_len != target_len:
            diff = abs(current_len - target_len)
            if current_len > target_len:
                # 对称裁剪
                crop_left = diff // 2
                recon = recon[..., crop_left:crop_left + target_len]
            else:
                # 对称填充
                pad_left = diff // 2
                pad_right = diff - pad_left
                recon = F.pad(recon, (pad_left, pad_right))
        
        return recon
    
    # 在 cnn_model.py 的 NanoporeCNNModel 类中添加：
    # 输出是torch.Size([1, 64, 2400])
    # [batch, channel, time] 是 PyTorch 的标准格式（NCL），所以 [1, 64, 2400] 是完全正常且正确的。
    # 在 PyTorch 中，1D 卷积层 nn.Conv1d 的输入/输出形状是：
    # (N, C_in, L)  →  (N, C_out, L_out)
    # N: batch size
    # C: number of channels（特征维度）
    # L: sequence length（时间步/信号长度）
    # L: sequence length（时间步/信号长度）
    # 输入电信号通常是 [B, 1, signal_len]（单通道）
    # 经过 CNN 后，变成 [B, 64, T]，其中：
    # 64 是 feature channels（即你想要的“64维特征”）
    # T=2400 是下采样后的时间步数
    # 那为什么你觉得应该是 [1, 2400, 64]？
    # 因为你可能更习惯 Transformer / NLP / 机器学习中的常见格式：
    # (batch, sequence_length, feature_dim)
    # 比如 BERT、LSTM 输出通常是 [B, T, D]。
    # 但在 PyTorch 的 CNN 生态中（尤其是语音、信号处理），默认使用 channel-first（NCL） 格式，因为：
    # Conv1d、BatchNorm1d、MaxPool1d 等模块都假设通道在第1维（索引1）；
    # 这样设计对 GPU 内存访问更高效。

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input signal to latent representation."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation back to signal."""
        return self.decoder(z)
