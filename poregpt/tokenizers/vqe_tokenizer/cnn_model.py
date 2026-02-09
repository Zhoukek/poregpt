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

class NanoporeCNNModel(nn.Module):
    """Nanopore 信号重建用纯卷积自编码器（无 VQ）。"""

    def __init__(self, cnn_type: Literal[0, 1, 2,3,4,5,6,7,8,9] = 1) -> None:
        super().__init__()

        if cnn_type not in (0, 1, 2,3,4,5,6,7,8,9):
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
            self.stride = 12
            self.receptive_field = 65
            self.RF = 65


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

    def _build_cnn_type9(self) -> None:
        """cnn_type=2: 多阶段下采样，总 stride=12，输出通道=512"""
        self.encoder = nn.Sequential(
            # Layer 1: 1 → 64, stride=1
            Conv1dWithMeanChannel(out_channels=64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),

            # Layer 2: 64 → 64, stride=1
            nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),

            # Layer 3: 64 → 128, stride=3
            nn.Conv1d(64, 128, kernel_size=9, stride=3, padding=4, bias=False),
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
