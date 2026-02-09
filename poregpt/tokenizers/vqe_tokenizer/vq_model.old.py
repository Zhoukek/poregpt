import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize
from typing import Tuple, Dict
# 导入新的 CNN 模型
from .cnn_model import NanoporeCNNModel
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


class NanoporeVQModel(nn.Module):
    """
    Nanopore VQ Tokenizer for Direct RNA Sequencing (130 bps, 4 kHz)

    支持多种 CNN 架构配置，通过 `cnn_type` 切换：
        - cnn_type=0: 大容量非严格对称模型（默认）
        - cnn_type=1: 小容量严格对称模型（通道数 1→16→32→64）

    设计目标通用：
        - 感受野 ≈ 33 采样点（≈1 个 RNA 碱基）
        - 总下采样率 = 5×（每碱基 ≈6 个 tokens）
        - 输出 codebook_dim 维 latent，直接用于 VQ
        - Decoder 在 cnn_type=1 时严格对称于 encoder

    适用于：VQ tokenizer + LLM basecalling pipeline
    """

    def __init__(
        self,
        codebook_size: int = 8192,
        commitment_weight: float = 1.0,
        orthogonal_reg_weight: float = 1.0,
        codebook_diversity_loss_weight: float = 1.0,
        cnn_type: int = 0,
        learnable_codebook: bool= True,
        init_codebook_path: str = None,
        freeze_cnn: bool = False,
        cnn_checkpoint_path: str = None
    ):
        """
        初始化 NanoporeVQModel。

        Args:
            codebook_size (int): VQ 码本大小。
            codebook_dim (int): VQ 嵌入维度（即 encoder 最终输出通道数）。
            commitment_weight (float): VQ commitment loss 权重。
            orthogonal_reg_weight (float): 正交正则化权重。
            codebook_diversity_loss_weight (float): 码本多样性损失权重。
            cnn_type (int): CNN 架构类型。
                - 0: 默认大模型（1 → 64 → 128 → codebook_dim）
                - 1: 严格对称小模型（1 → 16 → 32 → 64），此时 codebook_dim 必须为 64
        """
        super().__init__()

        # 设置 codebook_dim 根据 cnn_type
        if cnn_type == 0:
            codebook_dim = 256
        elif cnn_type == 1:
            codebook_dim = 64
        elif cnn_type == 2:
            codebook_dim = 512  # 固定为 512，与你提供的结构一致
        elif cnn_type == 3:
            codebook_dim = 64  # 固定为 512，与你提供的结构一致
        elif cnn_type == 4:
            codebook_dim = 32  # 固定为 512，与你提供的结构一致
        elif cnn_type == 5:
            codebook_dim = 16  # 固定为 512，与你提供的结构一致
        elif cnn_type == 6:
            codebook_dim = 128  # 固定为 512，与你提供的结构一致
        elif cnn_type == 7:
            codebook_dim = 256  # 固定为 512，与你提供的结构一致
        elif cnn_type == 8:
            codebook_dim = 512  # 固定为 512，与你提供的结构一致
        elif cnn_type == 9:
            codebook_dim = 512  # 固定为 512，与你提供的结构一致
        else:
            raise ValueError(f"Unsupported cnn_type: {cnn_type}. Supported: 0, 1, or 2.")

        self.codebook_dim = codebook_dim
        self.cnn_type = cnn_type
        self.latent_dim = codebook_dim
        self.codebook_size = codebook_size
        print(f"codebook_dim:{codebook_dim}")
        # 构建 encoder 和 decoder
        if cnn_type == 0:
            self._build_encoder_type0()
            self._build_decoder_type0()
            self.cnn_stride = 5   # 总下采样率（仅最后一层 stride=5）
            self.RF = 33          # 感受野（采样点），对应 ~1 个 RNA 碱基
        elif cnn_type == 1:
            self._build_encoder_type1()
            self._build_decoder_type1()
            self.cnn_stride = 5   # 总下采样率（仅最后一层 stride=5）
            self.RF = 33          # 感受野（采样点），对应 ~1 个 RNA 碱基
        elif cnn_type == 2:
            self._build_encoder_type2()
            self._build_decoder_type2()
            self.cnn_stride = 12  # 1 * 1 * 3 * 2 * 2
            self.RF = 65  #
        elif cnn_type == 3:
            self._build_encoder_type3()
            self._build_decoder_type3()
            self.cnn_stride = 5   # 总下采样率（仅最后一层 stride=5）
            self.RF = 33          # 感受野（采样点），对应 ~1 个 RNA 碱基
        elif cnn_type == 4:
            self._build_encoder_type4()
            self._build_decoder_type4()
            self.cnn_stride = 5   # 总下采样率（仅最后一层 stride=5）
            self.RF = 33          # 感受野（采样点），对应 ~1 个 RNA 碱基
        elif cnn_type == 5:
            self._build_encoder_type5()
            self._build_decoder_type5()
            self.cnn_stride = 5   # 总下采样率（仅最后一层 stride=5）
            self.RF = 33          # 感受野（采样点），对应 ~1 个 RNA 碱基
        elif cnn_type == 6:
            self._build_encoder_type6()
            self._build_decoder_type6()
            self.cnn_stride = 5   # 总下采样率（仅最后一层 stride=5）
            self.RF = 33          # 感受野（采样点），对应 ~1 个 RNA 碱基
        elif cnn_type == 7:
            self._build_encoder_type7()
            self._build_decoder_type7()
            self.cnn_stride = 5   # 总下采样率（仅最后一层 stride=5）
            self.RF = 33          # 感受野（采样点），对应 ~1 个 RNA 碱基
        elif cnn_type == 8:
            self._build_encoder_type8()
            self._build_decoder_type8()
            self.cnn_stride = 5   # 总下采样率（仅最后一层 stride=5）
            self.RF = 33          # 感受野（采样点），对应 ~1 个 RNA 碱基
        elif cnn_type == 9:
            self._build_encoder_type9()
            self._build_decoder_type9()
            self.cnn_stride = 12  # 1 * 1 * 3 * 2 * 2
            self.RF = 65  #
        else:
            raise ValueError(f"Unsupported cnn_type: {cnn_type}. Supported: 0 or 1.")


        # ======================================================================
        # VECTOR QUANTIZATION (VQ)
        # ======================================================================
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if learnable_codebook == True:
            ema_update = False
        else:
            ema_update = True

        self.vq = VectorQuantize(
            dim=self.latent_dim,
            codebook_size=codebook_size,
            kmeans_init=True,
            kmeans_iters=10,
            decay=0.99,
            threshold_ema_dead_code=2,
            commitment_weight=commitment_weight,
            codebook_diversity_loss_weight=codebook_diversity_loss_weight,
            orthogonal_reg_weight=orthogonal_reg_weight,
            orthogonal_reg_max_codes=256,
            orthogonal_reg_active_codes_only=True,
            learnable_codebook=learnable_codebook,
            ema_update = ema_update,
        )
        
        # 如果有初始codebook路径，加载它
        if init_codebook_path:
            self._load_init_codebook(init_codebook_path)
        # 如果有CNN检查点路径，加载权重
        if cnn_checkpoint_path:
            self._load_cnn_weights(cnn_checkpoint_path, freeze_cnn)
 

        if rank == 0:
            self._print_vq_config()
   
    def _load_cnn_weights(self, cnn_checkpoint_path, freeze_cnn=False):
        """从检查点加载CNN权重"""
        try:
            import os
            import torch
            
            if not os.path.isfile(cnn_checkpoint_path):
                print(f"⚠️ CNN checkpoint文件不存在: {cnn_checkpoint_path}")
                return
            
            print(f"📥 从 {cnn_checkpoint_path} 加载CNN权重")
            
            # 加载检查点
            cnn_ckpt = torch.load(cnn_checkpoint_path, map_location='cpu',weights_only=False)
            cnn_state_dict = cnn_ckpt.get('model_state_dict', cnn_ckpt)
            
            # 如果权重有'module.'前缀，去掉它
            if list(cnn_state_dict.keys())[0].startswith('module.'):
                cnn_state_dict = {k.replace('module.', ''): v for k, v in cnn_state_dict.items()}
            
            # 只加载encoder和decoder的权重
            encoder_decoder_keys = [k for k in cnn_state_dict.keys() 
                                   if k.startswith(('encoder.', 'decoder.'))]
            
            if not encoder_decoder_keys:
                print(f"⚠️ 在checkpoint中未找到encoder/decoder权重")
                return
            
            # 获取当前模型状态
            model_state = self.state_dict()
            loaded_keys = []
            
            for key in encoder_decoder_keys:
                if key in model_state and cnn_state_dict[key].shape == model_state[key].shape:
                    model_state[key] = cnn_state_dict[key]
                    loaded_keys.append(key)
            
            # 加载权重
            self.load_state_dict(model_state, strict=False)
            print(f"✅ 加载了 {len(loaded_keys)} 个encoder/decoder参数")
            
            # 冻结参数（如果需要）
            if freeze_cnn:
                print("🔒 冻结encoder和decoder参数")
                for name, param in self.named_parameters():
                    if name.startswith(('encoder.', 'decoder.')):
                        param.requires_grad = False
            
        except Exception as e:
            print(f"❌ 加载CNN权重失败: {e}")


    # 在 vq_model.py 中修改 _load_init_codebook 函数
    def _load_init_codebook2(self, init_codebook_path):
        """从numpy文件加载初始codebook - 只写死第一维为1"""
        try:
            import numpy as np
            import os
            
            if not os.path.isfile(init_codebook_path):
                print(f"⚠️ Codebook文件不存在: {init_codebook_path}")
                return
            
            # 直接加载numpy文件
            init_codebook = np.load(init_codebook_path)
            
            # 打印原始形状
            print(f"📊 加载的codebook原始形状: {init_codebook.shape}")
            
            # 获取模型期望的形状
            expected_shape = self.vq._codebook.embed.shape
            print(f"📊 模型期望的形状: {expected_shape}")
            
            # 核心修复：如果numpy是2D形状 (N, D)，就变成3D (1, N, D)
            if len(init_codebook.shape) == 2:
                # 从2D (N, D) 变成3D (1, N, D)
                init_codebook = init_codebook[np.newaxis, :, :]
                print(f"✅ 自动转换: 2D -> 3D, 新形状: {init_codebook.shape}")
            elif len(init_codebook.shape) == 3:
                # 已经是3D，直接使用
                print(f"✅ Codebook已经是3D形状: {init_codebook.shape}")
            else:
                print(f"❌ 不支持的codebook维度: {len(init_codebook.shape)}D")
                return
            
            # 现在检查维度是否匹配
            if init_codebook.shape != expected_shape:
                print(f"⚠️ Codebook形状不匹配:")
                print(f"   模型期望: {expected_shape}")
                print(f"   实际得到: {init_codebook.shape}")
                
                # 尝试只比较后两个维度
                if init_codebook.shape[1:] == expected_shape[1:]:
                    print(f"✅ 后两个维度匹配，可以继续")
                    # 形状不匹配可能是因为第一维不同，我们直接复制数据
                    if isinstance(self.vq._codebook.embed, nn.Parameter):
                        with torch.no_grad():
                            # 直接复制数据，忽略第一维
                            self.vq._codebook.embed.data.copy_(torch.from_numpy(init_codebook).float())
                        print(f"✅ Codebook加载成功（忽略第一维差异）")
                    else:
                        self.vq._codebook.embed = torch.from_numpy(init_codebook).float()
                        print(f"✅ Codebook加载成功（忽略第一维差异）")
                    return
                else:
                    print(f"❌ 维度完全不匹配，无法加载")
                    return
            
            # 直接赋值（如果是buffer）或复制（如果是parameter）
            init_codebook_tensor = torch.from_numpy(init_codebook).float()
            
            if isinstance(self.vq._codebook.embed, nn.Parameter):
                with torch.no_grad():
                    self.vq._codebook.embed.data.copy_(init_codebook_tensor)
            else:
                # 如果是buffer，直接赋值
                self.vq._codebook.embed = init_codebook_tensor
            
            print(f"✅ 从 {init_codebook_path} 加载初始codebook成功")
            print(f"   最终形状: {init_codebook_tensor.shape}")
            print(f"   是否可学习: {isinstance(self.vq._codebook.embed, nn.Parameter)}")
            
        except Exception as e:
            print(f"❌ 加载初始codebook失败: {e}")
            import traceback
            traceback.print_exc()

    # 在 vq_model.py 中修改 _load_init_codebook 方法
    def _load_init_codebook(self, init_codebook_path):
        """从numpy文件加载初始codebook - 修复内存布局问题"""
        try:
            import numpy as np
            import os
            
            if not os.path.isfile(init_codebook_path):
                print(f"⚠️ Codebook文件不存在: {init_codebook_path}")
                return
            
            # 直接加载numpy文件
            init_codebook = np.load(init_codebook_path)
            print(f"📥 加载codebook: {init_codebook.shape}")
            
            # 如果形状是2D，添加一个维度变成3D
            if len(init_codebook.shape) == 2:
                init_codebook = init_codebook[np.newaxis, :, :]
                print(f"  -> 调整为3D: {init_codebook.shape}")
            
            # 转换为tensor - 使用与模型参数相同的设备
            device = self.vq._codebook.embed.device
            init_codebook_tensor = torch.from_numpy(init_codebook).float().to(device)
            
            # 关键修复：确保内存布局一致
            # 使用 contiguous() 确保内存连续
            init_codebook_tensor = init_codebook_tensor.contiguous()
            
            # 获取原始参数的引用
            embed_param = self.vq._codebook.embed
            
            # 如果是Parameter，直接修改data
            if isinstance(embed_param, nn.Parameter):
                with torch.no_grad():
                    # 确保目标也是连续的
                    embed_param.data = embed_param.data.contiguous()
                    # 复制数据
                    embed_param.data.copy_(init_codebook_tensor)
            else:
                # 如果是buffer，直接赋值但保持内存布局
                self.vq._codebook.embed = init_codebook_tensor.contiguous()
            
            print(f"✅ Codebook初始化成功")
            print(f"   最终形状: {self.vq._codebook.embed.shape}")
            print(f"   内存连续: {self.vq._codebook.embed.is_contiguous()}")
            
        except Exception as e:
            print(f"❌ 加载初始codebook失败: {e}")
            import traceback
            traceback.print_exc()


    def _print_vq_config(self) -> None:
        """打印 VQ 配置信息（仅 rank 0）"""
        print("Intialized VectorQuantize with the following hyperparameters:")
        print(f"  dim: {self.latent_dim}")
        print(f"  codebook_size: {self.codebook_size}")
        print(f"  kmeans_init: True")
        print(f"  kmeans_iters: 10")
        print(f"  decay: 0.99")
        print(f"  threshold_ema_dead_code: 2")
        print(f"  commitment_weight: {self.vq.commitment_weight}")
        print(f"  codebook_diversity_loss_weight: {self.vq.codebook_diversity_loss_weight}")
        print(f"  orthogonal_reg_weight: {self.vq.orthogonal_reg_weight}")
        print(f"  orthogonal_reg_max_codes: 256")
        print(f"  orthogonal_reg_active_codes_only: True")
        print(f"  cnn_type: {self.cnn_type}")
        print("-" * 60)

    # ────────────────────────────────────────────────
    # ENCODER BUILDERS
    # ────────────────────────────────────────────────

    def _build_encoder_type0(self) -> None:
        """构建 cnn_type=0 的 encoder：1 → 64 → 128 → latent_dim（如 256）"""
        self.encoder = nn.Sequential(
            # Layer 1: 超局部特征提取（无下采样）
            nn.Conv1d(1, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),

            # Layer 2: 局部上下文聚合（无下采样）
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),

            # Layer 3: 下采样 + 升维至 latent space（RF=33, stride=5）
            nn.Conv1d(128, self.latent_dim, kernel_size=25, stride=5, padding=12, bias=False),
            nn.BatchNorm1d(self.latent_dim),
        )

    def _build_encoder_type1(self) -> None:
        """构建 cnn_type=1 的 encoder：1 → 16 → 32 → 64（严格对称）"""
        self.encoder = nn.Sequential(
            # Layer 1: 1 → 16
            nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2, bias=False),
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
    def _build_encoder_type2(self) -> None:
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

            # Layer 3: 64 → 128, stride=3
            nn.Conv1d(64, 128, kernel_size=9, stride=3, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),

            # Layer 4: 128 → 128, stride=2
            nn.Conv1d(128, 128, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),

            # Layer 5: 128 → 512, stride=2
            nn.Conv1d(128, self.latent_dim, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(self.latent_dim),
        )
    def _build_encoder_type3(self) -> None:
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

    def _build_encoder_type4(self) -> None:
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
    def _build_encoder_type5(self) -> None:
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
    def _build_encoder_type6(self) -> None:
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
    def _build_encoder_type7(self) -> None:
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
    def _build_encoder_type8(self) -> None:
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
    def _build_encoder_type9(self) -> None:
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
            nn.Conv1d(128, self.latent_dim, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(self.latent_dim),
        )

    # ────────────────────────────────────────────────
    # DECODER BUILDERS
    # ────────────────────────────────────────────────

    def _build_decoder_type0(self) -> None:
        """构建 cnn_type=0 的 decoder（近似对称，高维 refine）"""
        self.decoder = nn.Sequential(
            # Upsample ×5: 逆操作 encoder 最后一层
            nn.ConvTranspose1d(
                in_channels=self.latent_dim,
                out_channels=128,
                kernel_size=25,
                stride=5,
                padding=12,
                output_padding=0,
                bias=False,
            ),
            nn.BatchNorm1d(128),
            nn.SiLU(),

            # Refine layer: 消除棋盘伪影
            nn.Conv1d(128, 64, kernel_size=5, padding=2,bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),

            # Final projection to raw signal
            nn.Conv1d(64, 1, kernel_size=5,padding=2,bias=True),
        )

    def _build_decoder_type1(self) -> None:
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
    def _build_decoder_type2(self) -> None:
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
    def _build_decoder_type3(self) -> None:
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
    def _build_decoder_type4(self) -> None:
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
    def _build_decoder_type5(self) -> None:
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
    def _build_decoder_type6(self) -> None:
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
    def _build_decoder_type7(self) -> None:
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
    def _build_decoder_type8(self) -> None:
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
    def _build_decoder_type9(self) -> None:
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


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        前向传播。

        Args:
            x (torch.Tensor): 输入信号，形状 [B, 1, T]

        Returns:
            recon (torch.Tensor): 重建信号，[B, 1, T]
            indices (torch.Tensor): VQ 离散 token，[B, T//5]
            loss (torch.Tensor): VQ 总损失（标量）
            loss_breakdown (dict): 损失分项（commitment, diversity, ortho...）
        """
        # Encode: [B, 1, T] → [B, C, T//5]
        z_continuous = self.encoder(x)

        # Permute for VQ: [B, C, N] → [B, N, C]
        z_permuted = z_continuous.permute(0, 2, 1)

        # Quantize
        z_quantized_permuted, indices, loss, loss_breakdown = self.vq(
            z_permuted, return_loss_breakdown=True
        )

        # Back to [B, C, N] for decoder
        z_quantized = z_quantized_permuted.permute(0, 2, 1)

        # Decode
        recon = self.decoder(z_quantized)

        # Length alignment: ensure recon length == input length
        target_len = x.shape[-1]
        current_len = recon.shape[-1]
        if current_len > target_len:
            recon = recon[..., :target_len]
        elif current_len < target_len:
            recon = F.pad(recon, (0, target_len - current_len))

        return recon, indices, loss, loss_breakdown
