import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttentionWithKVCache(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Multi-Head Attention with Key-Value Cache
        Args:
            embed_dim: 输入嵌入的维度
            num_heads: 注意力头的数量
            dropout: Dropout 概率
        """
        super(MultiHeadAttentionWithKVCache, self).__init__()
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

        # 缓存
        self.k_cache = None
        self.v_cache = None

    def forward(self, query, key, value, mask=None, use_cache=False):
        """
        Args:
            query: 查询张量，形状为 (batch_size, seq_len, embed_dim)
            key: 键张量，形状为 (batch_size, seq_len, embed_dim)
            value: 值张量，形状为 (batch_size, seq_len, embed_dim)
            mask: 注意力掩码，形状为 (batch_size, 1, seq_len, seq_len)
            use_cache: 是否使用键值缓存
        Returns:
            输出张量，形状为 (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = query.size()

        # 线性变换并分割为多头
        Q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 如果使用缓存，将当前的 K 和 V 追加到缓存中
        if use_cache:
            if self.k_cache is None or self.v_cache is None:
                self.k_cache = K
                self.v_cache = V
            else:
                self.k_cache = torch.cat([self.k_cache, K], dim=2)  # 在 seq_len 维度上拼接
                self.v_cache = torch.cat([self.v_cache, V], dim=2)
            K, V = self.k_cache, self.v_cache

        # 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 应用掩码（如果提供）
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # 计算注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权求和
        attn_output = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_len, head_dim)

        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        # 输出线性变换
        output = self.out_proj(attn_output)

        return output

    def reset_cache(self):
        """
        重置键值缓存
        """
        self.k_cache = None
        self.v_cache = None


# 测试 Multi-Head Attention with KV Cache
if __name__ == "__main__":
    batch_size = 2
    seq_len = 4
    embed_dim = 16
    num_heads = 4

    # 输入张量
    query = torch.randn(batch_size, seq_len, embed_dim)
    key = torch.randn(batch_size, seq_len, embed_dim)
    value = torch.randn(batch_size, seq_len, embed_dim)

    # 掩码（可选）
    mask = torch.ones(batch_size, 1, seq_len, seq_len)

    # 初始化 Multi-Head Attention with KV Cache
    mha_kv_cache = MultiHeadAttentionWithKVCache(embed_dim, num_heads)

    # 前向传播（不使用缓存）
    output = mha_kv_cache(query, key, value, mask, use_cache=False)
    print("Output without cache:", output.shape)

    # 前向传播（使用缓存）
    for t in range(seq_len):
        q_t = query[:, t:t+1, :]  # 模拟逐时间步输入
        k_t = key[:, t:t+1, :]
        v_t = value[:, t:t+1, :]
        output_t = mha_kv_cache(q_t, k_t, v_t, mask=None, use_cache=True)
        print(f"Output at time step {t}:", output_t.shape)

    # 重置缓存
    mha_kv_cache.reset_cache()