import torch
import torch.nn as nn
import torch.nn.functional as F

class FlashAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Flash Attention Module
        Args:
            embed_dim: 输入嵌入的维度
            num_heads: 注意力头的数量
            dropout: Dropout 概率
        """
        super(FlashAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # 每个头的维度

        # 线性变换层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: 查询张量，形状为 (batch_size, seq_len, embed_dim)
            key: 键张量，形状为 (batch_size, seq_len, embed_dim)
            value: 值张量，形状为 (batch_size, seq_len, embed_dim)
            mask: 注意力掩码，形状为 (batch_size, 1, seq_len, seq_len)
        Returns:
            输出张量，形状为 (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = query.size()

        # 线性变换并分割为多头
        Q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Flash Attention 核心：块状计算
        # 初始化输出张量
        output = torch.zeros_like(Q)

        # 分块计算注意力
        block_size = 64  # 假设块大小为 64
        for i in range(0, seq_len, block_size):
            # 当前块的范围
            start = i
            end = min(i + block_size, seq_len)

            # 提取当前块
            Q_block = Q[:, :, start:end, :]  # (batch_size, num_heads, block_size, head_dim)
            K_block = K[:, :, :, :]         # (batch_size, num_heads, seq_len, head_dim)
            V_block = V[:, :, :, :]         # (batch_size, num_heads, seq_len, head_dim)

            # 计算注意力分数（缩放点积注意力）
            attn_scores = torch.matmul(Q_block, K_block.transpose(-2, -1)) / (self.head_dim ** 0.5)

            # 应用掩码（如果提供）
            if mask is not None:
                attn_scores = attn_scores.masked_fill(mask[:, :, start:end, :] == 0, float('-inf'))

            # 计算注意力权重
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # 加权求和
            attn_output = torch.matmul(attn_weights, V_block)  # (batch_size, num_heads, block_size, head_dim)

            # 将结果写入输出张量
            output[:, :, start:end, :] = attn_output

        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        # 输出线性变换
        output = self.out_proj(output)

        return output


# 测试 Flash Attention
if __name__ == "__main__":
    batch_size = 2
    seq_len = 128
    embed_dim = 64
    num_heads = 4

    # 输入张量
    query = torch.randn(batch_size, seq_len, embed_dim)
    key = torch.randn(batch_size, seq_len, embed_dim)
    value = torch.randn(batch_size, seq_len, embed_dim)

    # 掩码（可选）
    mask = torch.ones(batch_size, 1, seq_len, seq_len)

    # 初始化 Flash Attention
    flash_attention = FlashAttention(embed_dim, num_heads)

    # 前向传播
    output = flash_attention(query, key, value, mask)
    print("Output shape:", output.shape)  # 应为 (batch_size, seq_len, embed_dim)