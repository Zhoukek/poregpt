import torch
# 关键修改：从 bonito.util 中显式导入 load_model
from bonito.util import load_model 

# 你要使用的模型名称
# model_dir = "./dorado_models/dna_r10.4.1_e8.2_400bps_hac@v5.0.0"
model_dir = "/mnt/zzbnew/rnamodel/model/bonito/dna_basic_0121"

# 指定设备并加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(model_dir, device=device)
model.eval()

print("模型加载成功！网络结构如下：")
print(model)

# import triton
# triton.runtime.autotuner.DEFAULT_NUM_WARPS = 8

# import os
# # 彻底禁用所有 Flash Attention + Triton 加速（关键修复）
# os.environ["FLASH_ATTENTION_FORCE_DISABLE"] = "1"
# os.environ["FLASH_ATTENTION_SKIP_TRITON"] = "1"
# os.environ["FLASH_ATTENTION_DISABLE_BACKWARD"] = "1"
# os.environ["TRITON_DISABLE"] = "1"          # 新增：禁用Triton
# os.environ["TORCH_FLASH_ATTN_SDP_ENABLED"] = "0" # 新增：禁用PyTorch自带Flash Attention

# import torch
# from bonito.util import load_model

# model_dir = "./dorado_models/dna_r10.4.1_e8.2_400bps_sup@v5.0.0"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = load_model(model_dir, device=device)
# model.float()  # 关键：将模型所有权重转为 Float32
# model.eval()

# # 存储 transformer_encoder 的输出
# transformer_output = None

# def hook_fn(module, input, output):
#     global transformer_output
#     transformer_output = output  # 保存输出
    
# # 注册 hook 到 transformer_encoder
# hook = model.encoder.transformer_encoder.register_forward_hook(hook_fn)

# # 准备输入数据（示例）
# # 注意：需要根据你的实际输入shape调整，通常是 (batch, channels, sequence_length)
# sample_input = torch.randn(1, 1, 5000).to(device).float() # batch=1, channels=1, seq_len=5000

# # sample_input_half = sample_input.half()  # 转为半精度
# # 前向传播
# with torch.no_grad():
#     output = model(sample_input)  # 这会触发 hook

# # 现在 transformer_output 包含了 transformer_encoder 的输出
# print(f"Transformer encoder output shape: {transformer_output.shape}")

# # 使用完毕后移除 hook
# hook.remove()


