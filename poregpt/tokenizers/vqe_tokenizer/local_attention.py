# local_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, window_size: int = 33, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.window_size = window_size
        self.batch_first = True  # 添加这一行

        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def create_local_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """创建局部注意力掩码"""
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)
        for i in range(seq_len):
            start_idx = max(0, i - self.window_size // 2)
            end_idx = min(seq_len, i + self.window_size // 2 + 1)
            mask[i, start_idx:end_idx] = True  # True表示可见
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.nhead, self.head_dim

        # Generate Q, K, V
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # [B, H, T, D]
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)  # [B, H, T, D]
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)  # [B, H, T, D]

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, T, T]

        # Create and apply local attention mask
        local_mask = self.create_local_mask(T, x.device)  # [T, T]
        local_mask = local_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
        attn_scores = attn_scores.masked_fill(~local_mask, float('-inf'))

        # Apply softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, H, T, T]
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attended = torch.matmul(attn_weights, v)  # [B, H, T, D]
        attended = attended.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, C]

        return self.out_proj(attended)

class LocalTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, window_size: int = 33,
                 dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.self_attn = LocalAttention(d_model, nhead, window_size, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, 
                src_mask=None, 
                src_key_padding_mask=None, 
                is_causal=None) -> torch.Tensor:
        """
        接受与标准TransformerEncoderLayer相同的参数
        """
        # Multi-head attention with local attention
        # 忽略传入的mask，因为我们使用固定的局部注意力
        attn_output = self.self_attn(src)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        # Feed forward
        ffn_output = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(ffn_output)
        src = self.norm2(src)
        return src

class LocalTransformerEncoder(nn.Module):
    """自定义的Transformer编码器，专门用于LocalAttention"""
    def __init__(self, encoder_layer: LocalTransformerEncoderLayer, num_layers: int):
        super().__init__()
        # 创建多个层的副本
        self.layers = nn.ModuleList([
            LocalTransformerEncoderLayer(
                d_model=encoder_layer.self_attn.d_model,
                nhead=encoder_layer.self_attn.nhead,
                window_size=encoder_layer.self_attn.window_size,
                dim_feedforward=encoder_layer.linear1.out_features,
                dropout=encoder_layer.dropout.p
            ) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        
    def forward(self, src: torch.Tensor, 
                mask=None, 
                src_key_padding_mask=None, 
                is_causal=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, 
                          src_key_padding_mask=src_key_padding_mask, 
                          is_causal=is_causal)
        return output
