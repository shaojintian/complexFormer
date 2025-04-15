import torch
import time
from flashattention import FlashAttention
from mha import MultiHeadAttention

def benchmark_attention(attention_module, query, key, value, mask=None, num_iters=10):
    """
    基准测试函数，用于测量注意力模块的前向传播时间。
    Args:
        attention_module: 注意力模块（如 FlashAttention 或 MultiHeadAttention）。
        query: 查询张量，形状为 (batch_size, seq_len, embed_dim)。
        key: 键张量，形状为 (batch_size, seq_len, embed_dim)。
        value: 值张量，形状为 (batch_size, seq_len, embed_dim)。
        mask: 注意力掩码，形状为 (batch_size, 1, seq_len, seq_len)。
        num_iters: 测试迭代次数。
    Returns:
        平均前向传播时间（毫秒）。
    """
    # 确保模型在 GPU 上运行（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attention_module = attention_module.to(device)
    query, key, value = query.to(device), key.to(device), value.to(device)
    if mask is not None:
        mask = mask.to(device)

    # 预热 GPU（避免首次运行的初始化开销）
    for _ in range(5):
        _ = attention_module(query, key, value, mask)

    # 正式测试
    start_time = time.time()
    for _ in range(num_iters):
        _ = attention_module(query, key, value, mask)
    end_time = time.time()

    avg_time = (end_time - start_time) / num_iters * 1000  # 转换为毫秒
    return avg_time

if __name__ == "__main__":
    # 参数设置
    batch_size = 32
    seq_len = 512
    embed_dim = 128
    num_heads = 8
    num_iters = 50

    # 初始化输入张量
    query = torch.randn(batch_size, seq_len, embed_dim)
    key = torch.randn(batch_size, seq_len, embed_dim)
    value = torch.randn(batch_size, seq_len, embed_dim)
    mask = torch.ones(batch_size, 1, seq_len, seq_len)  # 可选掩码

    # 初始化注意力模块
    flash_attention = FlashAttention(embed_dim, num_heads)
    mha = MultiHeadAttention(embed_dim, num_heads)

    # 测试 Flash Attention
    flash_time = benchmark_attention(flash_attention, query, key, value, mask, num_iters)
    print(f"Flash Attention Average Time: {flash_time:.2f} ms")

    # 测试传统 MHA
    mha_time = benchmark_attention(mha, query, key, value, mask, num_iters)
    print(f"Multi-Head Attention Average Time: {mha_time:.2f} ms")

    # 比较性能
    speedup = mha_time / flash_time
    print(f"Flash Attention is {speedup:.2f}x faster than Multi-Head Attention.")