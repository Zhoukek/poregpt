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

# 1. 模拟输入数据
# 纳米孔信号通常是 1 维的连续电流值
# 假设我们模拟 2000 个采样点的信号
# batch_size=1, channels=1, length=2000
batch_size = 1
channels = 1
signal_length = 2000 
dummy_input = torch.randn(batch_size, channels, signal_length).to(device)
model.float()

# 2. 推理
with torch.no_grad():
    # 这里的输出通常是一个命名元组或特定的张量
    outputs = model(dummy_input)

# 3. 解析输出
print("-" * 30)
print(f"输入形状: {dummy_input.shape}") # [1, 1, 2000]

# 这种模型（基于 bonito 框架）通常返回的是 log 概率或分数
# 形状通常为 [Time, Batch, State]
print(f"输出形状: {outputs.shape}") 
# 预期输出长度为 2000 / 5 = 400 左右
# 最后一维 4096 对应 LinearCRFEncoder 的 out_features

# 4. 查看部分数值（发射得分）
print("输出部分得分样例 (前 5 个时间点的局部数据):")
print(outputs[:5, 0, :10])


