import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize
from typing import Tuple, Dict
# 导入新的 CNN 模型
from .cnn_model import NanoporeCNNModel


# 导入局部注意力模块
from .local_attention import LocalTransformerEncoder


class NanoporeVQEModel_V3(nn.Module):
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
        codebook_decay: float = 0.99,
        codebook_emadc: int = 2,
        commitment_weight: float = 1.0,
        orthogonal_reg_weight: float = 1.0,
        codebook_diversity_loss_weight: float = 1.0,
        cnn_type: int = 0,
        learnable_codebook: bool= True,
        init_codebook_path: str = None,
        freeze_cnn: bool = False,
        cnn_checkpoint_path: str = None,
        # Transformer 参数
        nhead: int = 4,              # Transformer 头数
        num_layers: int = 4,         # Transformer 层数
        dim_feedforward: int = 1024, # FFN 维度
        dropout: float = 0.1,
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

        self.cnn_model = NanoporeCNNModel(cnn_type=cnn_type)
        
        d_model = self.cnn_model.out_channels *2  # 自动设置为CNN输出维度
        
        codebook_dim = d_model

        # 设置 codebook_dim 根据 cnn_type
        self.codebook_dim = codebook_dim
        self.cnn_type = cnn_type
        self.codebook_size = codebook_size
        self.cnn_stride = self.cnn_model.stride
        self.RF = self.cnn_model.RF


        print(f"codebook_dim:{codebook_dim}")


        # -----------------------------
        # 2. 添加线性投影层
        # -----------------------------
        # 定义从 CNN 输出维度 (d_model//2) 到 Transformer 维度 (d_model) 的线性层
        self.proj_in = nn.Linear(self.cnn_model.out_channels, d_model) # e.g., Linear(128 -> 256)


        # ======================================================================
        # VECTOR QUANTIZATION (VQ)
        # ======================================================================
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if learnable_codebook == True:
            ema_update = False
        else:
            ema_update = True


        # -----------------------------
        # 2. Transformer Context Modeler
        # -----------------------------
        self.d_model = d_model # 256
        
        dim_feedforward = d_model*2
        # 线性投影层: 将 CNN 特征 (128) 映射到 Transformer 维度 (256)
        # --- 新增：位置编码 ---
        self.max_position = 2048  # 预设一个足够大的上限
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_position, d_model))
        # 初始化（小的标准差有助于稳定训练开始阶段）
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)        
        # 构建 Transformer Encoder
        #encoder_layer = nn.TransformerEncoderLayer(
        #    d_model=d_model, 
        #    nhead=nhead, 
        #    dim_feedforward=dim_feedforward, 
        #    dropout=dropout,
        #    activation='gelu', # GELU 通常比 ReLU 在信号处理上效果稍好
        #    batch_first=True # 非常重要：保持 (Batch, Seq, Feature) 格式
        #)
        #self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 使用局部注意力的Transformer
        # 2. 直接实例化我们自定义的 Encoder，而不是传给 nn.TransformerEncoder
        self.transformer_encoder = LocalTransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            window_size=65,        # 1200 token 下，65 是非常合理的窗口
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        # --- 添加 VQ 后的降维层 ---
        self.vq_to_cnn_dim = nn.Linear(d_model, self.cnn_model.out_channels)
        # --- 添加结束 ---

        self.vq = VectorQuantize(
            dim=d_model,
            codebook_size=codebook_size,
            kmeans_init=True,
            kmeans_iters=10,
            decay=codebook_decay,
            threshold_ema_dead_code=codebook_emadc,
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
            print("预训练模型权重键 (前几个):", list(cnn_state_dict.keys())[:5])
            print("当前模型权重键 (前几个):", list(self.state_dict().keys())[:5])
            # 预训练模型权重键 (前几个): ['encoder.0.mean_conv.weight', 'encoder.0.std_conv.weight', 'encoder.1.weight', 'encoder.1.bias', 'encoder.1.running_mean']
            # 当前模型权重键 (前几个): ['cnn_model.encoder.0.mean_conv.weight', 'cnn_model.encoder.0.std_conv.weight', 'cnn_model.encoder.1.weight', 'cnn_model.encoder.1.bias', 'cnn_model.encoder.1.running_mean']
            # --- 添加以下逻辑 ---
            # 假设当前模型的 encoder 部分是通过 'cnn_model.encoder' 这个属性访问的
            # 我们需要将预训练权重的 'encoder.xxxx' 映射到 'cnn_model.encoder.xxxx'
            mapped_cnn_state_dict = {}
            for k, v in cnn_state_dict.items():
                if k.startswith('encoder.'): # 如果原始键是以 'encoder.' 开头
                    new_k = 'cnn_model.' + k # 将其映射为 'cnn_model.encoder.xxxx'
                    mapped_cnn_state_dict[new_k] = v
                else:
                    # 如果不是以 'encoder.' 开头（例如 decoder 或其他部分），可以选择跳过或也进行相应映射
                    pass # 或者继续处理其他部分，如果需要的话
                        # 只加载encoder和decoder的权重
                        # encoder_decoder_keys = [k for k in cnn_state_dict.keys() if k.startswith(('encoder.', 'decoder.'))]
             # 现在使用映射后的字典
            cnn_state_dict = mapped_cnn_state_dict

            # 原来的筛选逻辑现在应该能找到匹配项了
            # 注意这里也改为 'cnn_model.encoder.'
            encoder_decoder_keys = [k for k in cnn_state_dict.keys() if k.startswith(('cnn_model.encoder.'))]
            if not encoder_decoder_keys:
                print(f"⚠️ 在checkpoint中未找到encoder/decoder权重")
                return
           # --- 添加结束 ---

            # 获取当前模型状态
            model_state = self.state_dict()
            loaded_keys = []

            for key in encoder_decoder_keys:
                if key in model_state and cnn_state_dict[key].shape == model_state[key].shape:
                    print(f"加载参数:{key}")
                    model_state[key] = cnn_state_dict[key]
                    loaded_keys.append(key)

            # 加载权重
            self.load_state_dict(model_state, strict=False)
            #print(f"✅ 加载了 {len(loaded_keys)} 个encoder/decoder参数")
            print(f"✅ 加载了 {len(loaded_keys)} 个encoder参数")

            # 冻结参数（如果需要）
            freeze_cnt = 0
            if freeze_cnn:
                #print("🔒 冻结encoder和decoder参数")
                print("🔒 冻结encoder参数")
                for name, param in self.named_parameters():
                    #if name.startswith(('encoder.', 'decoder.')):
                    #if name.startswith(('encoder.')):
                    if name.startswith(('cnn_model.encoder.')):      # <- 修改为新的前缀
                        freeze_cnt +=1
                        param.requires_grad = False
                        print(f"冻结参数:{name}")
            print(f"✅ 冻结了 {freeze_cnt} 个encoder参数")
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
        print(f"  dim: {self.d_model}")
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


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        前向传播函数 (升级版：支持 CNN + Transformer 架构)。

        Args:
            x (torch.Tensor): 输入信号，形状 [B, 1, T]
                (例如: B=4, T=2560 -> )

        Returns:
            recon (torch.Tensor): 重建信号，[B, 1, T]
            indices (torch.Tensor): VQ 离散 token，[B, N] (N = T // 5)
            loss (torch.Tensor): VQ 总损失（标量）
            loss_breakdown (dict): 损失分项（commitment, diversity, ortho...）
        """

        # ======================================================================
        # 1. CNN 编码器 (特征提取)
        #    目标：将原始长序列压缩为较短的特征序列
        # ======================================================================
        # 输入 x: [B, 1, T]
        #   (B: Batch Size, 1: 单通道电信号, T: 信号长度 2560)
        z_cnn = self.cnn_model.encode(x)
        # 输出 z_cnn: [B, C_CNN, N]
        #   (C_CNN: CNN 特征维度 = 128, N: 序列长度 = T//5 = 512)
        #   例如:

        # ======================================================================
        # 2. 维度变换与投影 (Projection)
        #    目标：适配 Transformer 的输入维度 (128 -> 256)
        # ======================================================================
        # 2.1 变换维度顺序
        # PyTorch 的 CNN 输出是 [B, Channels, Length]
        # PyTorch 的 Transformer 要求 [B, Length, Features] (batch_first=True)
        z_permuted = z_cnn.permute(0, 2, 1)
        # z_permuted: [B, N, C_CNN] -> 例如:

        # 2.2 线性投影 (Projection)
        # 将 CNN 提取的 128 维特征，映射到 Transformer 的 256 维空间
        # 这对应架构中的 "Linear Projection: 128 -> 256"
        z_projected = self.proj_in(z_permuted)
        # z_projected: [B, N, D_Model]
        #   (D_Model: Transformer 模型维度 = 256)
        #   例如:
        # 这个张量就是你描述中的 h_proj / z_e (Encoder Output)
        
        # ======================================================================
        # 3. Transformer 上下文建模 (Context Modeling)
        #    目标：利用全局注意力机制增强特征表达能力
        # ======================================================================
        # 输入 z_projected: [B, N, 256]
        # --- 新增：注入位置编码 ---
        seq_len = z_projected.size(1)
        if seq_len > self.max_position:
            raise ValueError(f"输入序列长度 {seq_len} 超过了预设的最大位置编码长度 {self.max_position}")
        # 将位置编码加到特征上（利用广播机制 [1, 2048, 256] -> [B, 512, 256]）
        z_with_pos = z_projected + self.pos_embedding[:, :seq_len, :]
        # -------------------------
        # 修改输入，使用带位置信息的 z_with_pos
        z_transformed = self.transformer_encoder(z_with_pos)

        # 输出 z_transformed: [B, N, 256]
        # 特征维度保持 256 不变，但每个位置的特征都融合了全局上下文信息
        # 这个张量就是最终送入 VQ 的 z_e

        # ======================================================================
        # 4. 向量量化 (Vector Quantization - VQ)
        #    目标：将连续特征映射为离散 Token
        # ======================================================================
        # 注意：z_transformed 的形状已经是 [B, N, 256]，完美符合 VQ 输入要求
        # (不需要像 CNN 那样先 Permute，因为 Transformer 输出已经是 [B, N, C])

        # VQ 处理：
        # 1. 计算距离：找到 z_transformed 中每个向量在 Codebook 中最接近的索引
        # 2. 生成 Token：输出离散索引 indices
        # 3. 生成量化向量：输出可微的 z_quantized_permuted (用于反向传播)
        z_quantized_permuted, indices, loss, loss_breakdown = self.vq(
            z_transformed, # 输入连续特征
            return_loss_breakdown=True # 返回详细的损失分项
        )

        # z_quantized_permuted: [B, N, 256] (量化后的连续特征，用于 Decoder)
        # indices: [B, N] (离散的 Token ID，用于存储/下游任务)
        #   例如 indices:

        # ======================================================================
        # 4.1 降维 (Dimension Reduction after VQ)
        #    目标：将 VQ 输出的 d_model 维特征降维到 CNN 输出的 out_channels 维
        # ======================================================================
        z_quantized_cnn_dim = self.vq_to_cnn_dim(z_quantized_permuted)
        # z_quantized_cnn_dim: [B, N, self.cnn_model.out_channels] (例如: [B, 512, 128])


        # ======================================================================
        # 5. 解码器准备 (Decoder Preparation)
        #    目标：将特征格式转换回 CNN 解码器需要的格式
        # ======================================================================
        # Decoder (反卷积网络) 需要的格式是 [B, Channels, Length]
        z_quantized = z_quantized_cnn_dim.permute(0, 2, 1)
        # z_quantized: [B, 256, N] -> 例如:

        # ======================================================================
        # 6. 解码器 (Decoder - 信号重构)
        #    目标：将量化特征重构回原始信号空间
        # ======================================================================
        recon = self.cnn_model.decode(z_quantized)
        # recon: [B, 1, T_recon]
        #   (理论上 T_recon 应该等于 T，但为了防止反卷积导致的长度微小差异)

        # ======================================================================
        # 7. 长度对齐 (Length Alignment)
        #    目标：确保输出信号长度与输入完全一致
        # ======================================================================
        target_len = x.shape[-1]  # 输入信号的原始长度 (2560)
        current_len = recon.shape[-1] # 重构信号的当前长度

        if current_len > target_len:
            # 如果重构信号过长（通常由 Padding 引起），进行裁剪
            recon = recon[..., :target_len]
        elif current_len < target_len:
            # 如果重构信号过短，进行填充 (Pad)
            # F.pad 的参数是 (左填充, 右填充)，这里只在时间轴末尾填充
            recon = F.pad(recon, (0, target_len - current_len))

        return recon, indices, loss, loss_breakdown
