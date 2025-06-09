import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Optional, Tuple
from torch.amp import autocast 

# --- 日志记录设置 (保持不变) ---
# 确保日志目录存在
import os
if not os.path.exists('./logs'):
    os.makedirs('./logs')

logger = logging.getLogger(__name__)
# 避免重复添加处理器
if not logger.handlers:
    logger.addHandler(logging.FileHandler('./logs/complex_attention.log'))

# 模拟的配置加载函数
def load_config(path):
    class Config:
        def __init__(self):
            self.debug = True # 在此设置为True以便日志记录
    return Config()

logger.setLevel(logging.INFO) if load_config("./pretrain/config.yaml").debug else logger.setLevel(logging.WARNING)

__all__ = ["ComplexMultiHeadAttentionV2"]

# --- 重构后的辅助模块 ---

class EulerTransform(nn.Module):
    """
    将输入张量通过欧拉变换分解为幅度和相位。
    """
    def __init__(self, d_k: int):
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even for Euler transform"
        self.d_k_half = d_k // 2
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch_size, num_heads, seq_len, d_k)
        real_part, imag_part = torch.split(x, self.d_k_half, dim=-1)
        
        # 为数值稳定性添加 epsilon
        magnitude: torch.Tensor = torch.sqrt(real_part**2 + imag_part**2 + 1e-9) 
        phase: torch.Tensor = torch.atan2(imag_part, real_part)
        return magnitude, phase

class RelativePosEncoding(nn.Module):
    """
    计算相对位置编码分量 (Delta P)，支持KV缓存。
    """
    def __init__(self, d_k_half: int, freqs: torch.Tensor):
        super().__init__()
        self.d_k_half = d_k_half
        # 将传入的频率张量注册为缓冲区
        self.register_buffer('g', freqs)

    def forward(self, query_len: int, key_len: int, past_key_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        计算相对位置编码。
        
        Args:
            query_len (int): 当前查询序列的长度。
            key_len (int): 总的键序列长度 (包括过去的键)。
            past_key_len (int): 过去键序列的长度 (KV缓存的长度)。
            device: 张量所在的设备。
            dtype: 张量的数据类型。

        Returns:
            torch.Tensor: 相对位置编码张量，形状为 (query_len, key_len, d_k_half)。
        """
        # q_pos 是查询token的绝对位置
        q_pos = torch.arange(past_key_len, past_key_len + query_len, device=device, dtype=torch.float).unsqueeze(1)
        # k_pos 是键token的绝对位置
        k_pos = torch.arange(0, key_len, device=device, dtype=torch.float).unsqueeze(0)
        
        relative_distance = q_pos - k_pos

        # self.g 是一个1D张量，形状为 (d_k_half)
        # 调整 freqs 的形状以便广播
        freqs = self.g.to(device=device, dtype=dtype).view(1, 1, self.d_k_half)

        # 最终的 delta_p 形状: (query_len, key_len, d_k_half)
        delta_p = relative_distance.unsqueeze(-1) * freqs
        return delta_p

class ComplexMultiHeadAttentionV2(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        
        assert self.d_k % 2 == 0, "d_k (d_model/num_heads) must be even for Euler transform"
        self.d_k_half = self.d_k // 2

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # 实例化真正的辅助模块
        self.euler_transform = EulerTransform(self.d_k)
        
        self.delta_params = nn.Parameter(torch.randn(self.num_heads, self.d_k_half))
        self.bias_params = nn.Parameter(torch.zeros(self.num_heads, self.d_k_half))

        self.register_buffer('g', self._get_frequencies())
        self.relative_pos_encoding = RelativePosEncoding(self.d_k_half, self.g)
        
        self.dropout = nn.Dropout(0)
        self.softmax = nn.Softmax(dim=-1)

        self.query_chunk_size = 8
        logger.info(f"ComplexAttention initialized with query_chunk_size: {self.query_chunk_size}")
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, ComplexMultiHeadAttentionV2):
            nn.init.normal_(module.delta_params, mean=0.0, std=1.0 / math.sqrt(module.d_k_half))
            nn.init.constant_(module.bias_params, 0.0)

    def _get_frequencies(self, max_len: int = 5000) -> torch.Tensor:
        freqs: torch.Tensor = 10000 ** (-torch.arange(0, self.d_k // 2, dtype=torch.float) * 2 / self.d_k)
        return freqs
        
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            hidden_states (torch.Tensor): 输入张量，形状 (B, QL, D)。在自注意力中，q, k, v都源于此。
            attention_mask (Optional[torch.Tensor]): 注意力掩码。
            past_key_value (Optional[Tuple[torch.Tensor, torch.Tensor]]): 包含过去key和value的元组 (past_key, past_value)。
            use_cache (bool): 是否使用并返回KV缓存。
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
            - output: 注意力层输出，形状 (B, QL, D)。
            - attention_weights: 注意力权重，形状 (B, H, QL, KL)。
            - present_key_value: 更新后的KV缓存，如果 use_cache 为 True。
        """
        # 在自注意力中，q, k, v 都来自于当前的 hidden_states
        q, k, v = hidden_states, hidden_states, hidden_states
        
        batch_size, query_len, _ = q.shape
        
        Q_proj = self.W_q(q).view(batch_size, query_len, self.num_heads, self.d_k).transpose(1, 2)
        K_proj = self.W_k(k).view(batch_size, query_len, self.num_heads, self.d_k).transpose(1, 2)
        V_proj = self.W_v(v).view(batch_size, query_len, self.num_heads, self.d_v).transpose(1, 2)

        past_key_len = 0
        if past_key_value is not None:
            past_key, past_value = past_key_value
            past_key_len = past_key.shape[2]
            K_proj = torch.cat([past_key, K_proj], dim=2)
            V_proj = torch.cat([past_value, V_proj], dim=2)

        key_len = K_proj.shape[2]
        
        present_key_value = (K_proj, V_proj) if use_cache else None

        magnitude_q_full, phase_q_full = self.euler_transform(Q_proj)
        magnitude_k, phase_k = self.euler_transform(K_proj)

        delta_adapted = self.delta_params.view(1, self.num_heads, 1, 1, self.d_k_half)
        bias_adapted = self.bias_params.view(1, self.num_heads, 1, 1, self.d_k_half)
        
        delta_p_full = self.relative_pos_encoding(query_len, key_len, past_key_len, q.device, q.dtype)
        delta_p_expanded_full = delta_p_full.unsqueeze(0).unsqueeze(0)

        effective_query_chunk_size = self.query_chunk_size if self.query_chunk_size is not None else query_len
        num_q_chunks = (query_len + effective_query_chunk_size - 1) // effective_query_chunk_size
        
        scores_chunks = []
        phase_k_unsqueezed = phase_k.unsqueeze(2)

        with autocast('cuda', dtype=torch.bfloat16):
            for i in range(num_q_chunks):
                q_start = i * effective_query_chunk_size
                q_end = min((i + 1) * effective_query_chunk_size, query_len)
                
                magnitude_q_chunk = magnitude_q_full[:, :, q_start:q_end, :]
                phase_q_chunk = phase_q_full[:, :, q_start:q_end, :]
                
                as_chunk = phase_q_chunk.unsqueeze(3) - phase_k_unsqueezed
                adapt_as_chunk = delta_adapted * as_chunk + bias_adapted
                
                delta_p_expanded_chunk = delta_p_expanded_full[:, :, q_start:q_end, :, :]
                combined_phase_chunk = adapt_as_chunk + delta_p_expanded_chunk
                cos_combined_phase_chunk = torch.cos(combined_phase_chunk)
                
                scores_current_chunk = torch.einsum(
                    'bhqd, bhkd, bhqkd -> bhqk',
                    magnitude_q_chunk,
                    magnitude_k,
                    cos_combined_phase_chunk
                )
                scores_chunks.append(scores_current_chunk)

            scores = torch.cat(scores_chunks, dim=2) if num_q_chunks > 1 else scores_chunks[0]

            if attention_mask is not None:
                # 支持布尔或浮点型掩码
                if attention_mask.dtype == torch.bool:
                    additive_mask = torch.where(attention_mask, 0.0, torch.finfo(scores.dtype).min)
                else:
                    additive_mask = attention_mask
                
                # 确保掩码维度正确，并与KV缓存的长度对齐
                if additive_mask.dim() == 2:
                    additive_mask = additive_mask.unsqueeze(0).unsqueeze(0)
                elif additive_mask.dim() == 3:
                    additive_mask = additive_mask.unsqueeze(1)
                
                # 从掩码中选择与当前查询和总键长度对应的部分
                # 假设 attention_mask 覆盖了所有可能的位置
                mask_slice = additive_mask[:, :, -query_len:, :key_len]
                scores = scores + mask_slice

            attention_weights = self.softmax(scores)
            attention_weights = self.dropout(attention_weights)

            output = torch.matmul(attention_weights, V_proj)
            output = output.transpose(1, 2).contiguous().view(batch_size, query_len, self.d_model)
            output = self.W_o(output)

            if torch.isnan(output).any():
                logger.error("Output contains NaN after W_o")
                raise ValueError("Output contains NaN")
            
        return output, attention_weights, present_key_value

if __name__ == '__main__':
    batch_size = 2
    seq_len = 32
    d_model = 256
    num_heads = 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ComplexMultiHeadAttentionV2(d_model, num_heads).to(device)
    model.eval() # 关闭dropout等

    print("--- 1. Testing Prefill (KV Cache Creation) ---")
    # 模拟一个有padding的输入序列
    prompt_len = 30
    hidden_states_prefill = torch.randn(batch_size, prompt_len, d_model, device=device)
    # 创建一个掩码，其中最后5个token是padding (值为0)
    prefill_mask = torch.ones(batch_size, 1, prompt_len, prompt_len, device=device, dtype=torch.float)
    prefill_mask[:, :, :, -5:] = -torch.inf

    with torch.no_grad():
        output_prefill, _, past_key_value = model(
            hidden_states=hidden_states_prefill,
            attention_mask=prefill_mask,
            use_cache=True
        )

    print(f"Prefill output shape: {output_prefill.shape}")
    assert output_prefill.shape == (batch_size, prompt_len, d_model)
    print(f"Past key shape: {past_key_value[0].shape}")
    assert past_key_value[0].shape == (batch_size, num_heads, prompt_len, d_model // num_heads)
    print(f"Past value shape: {past_key_value[1].shape}")
    assert past_key_value[1].shape == (batch_size, num_heads, prompt_len, d_model // num_heads)
    print("Prefill test PASSED.\n")

    print("--- 2. Testing Decode (KV Cache Usage) ---")
    # 模拟解码一步，输入只有一个新token
    hidden_states_decode = torch.randn(batch_size, 1, d_model, device=device)
    
    # 在解码时，新token可以关注所有过去的token，所以通常不需要特定的掩码
    # 如果需要处理padding，掩码逻辑会更复杂，但这里简化处理
    
    with torch.no_grad():
        output_decode, _, new_past_key_value = model(
            hidden_states=hidden_states_decode,
            attention_mask=None, # 在解码步通常为None
            past_key_value=past_key_value, # 使用上一轮的缓存
            use_cache=True
        )

    new_seq_len = prompt_len + 1
    print(f"Decode output shape: {output_decode.shape}")
    assert output_decode.shape == (batch_size, 1, d_model)
    print(f"New past key shape: {new_past_key_value[0].shape}")
    assert new_past_key_value[0].shape == (batch_size, num_heads, new_seq_len, d_model // num_heads)
    print(f"New past value shape: {new_past_key_value[1].shape}")
    assert new_past_key_value[1].shape == (batch_size, num_heads, new_seq_len, d_model // num_heads)
    print("Decode test PASSED.")