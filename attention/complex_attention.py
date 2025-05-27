import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Optional
from torch.cuda.amp import autocast

logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler('./logs/complex_attention.log')) #确保日志能输出
from utils import load_config
logger.setLevel(logging.INFO) if load_config("./pretrain/config.yaml").debug else logger.setLevel(logging.WARNING)# 设置日志级别

__all__ = ["ComplexMultiHeadAttentionV2"]

class EulerTransform(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k_half = d_k // 2
    def forward(self, x):
        # x: (B, H, SeqLen, DK)
        # Dummy: just split
        return x[..., :self.d_k_half], x[..., self.d_k_half:]

class RelativePosEncoding(nn.Module):
    def __init__(self, d_k_half):
        super().__init__()
        self.d_k_half = d_k_half
    def forward(self, ql, kl, device):
        # Dummy: return zeros
        return torch.zeros(ql, kl, self.d_k_half, device=device)

class ComplexMultiHeadAttentionV2(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads # Assuming d_k == d_v for simplicity here
        
        # Ensure d_k is even for Euler transform splitting
        assert self.d_k % 2 == 0, "d_k (d_model/num_heads) must be even for Euler transform"
        self.d_k_half = self.d_k // 2

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False) # Output projection

        self.euler_transform = EulerTransform(self.d_k) # Replace with actual
        
        # Learnable parameters for adaptive integration
        # Shape: (H, DK/2)
        self.delta_params = nn.Parameter(torch.randn(self.num_heads, self.d_k_half))
        self.bias_params = nn.Parameter(torch.zeros(self.num_heads, self.d_k_half))

        self.relative_pos_encoding = RelativePosEncoding(self.d_k_half) # Replace with actual
        
        self.dropout = nn.Dropout(0)
        self.softmax = nn.Softmax(dim=-1) # To apply after scores

        self.query_chunk_size = 8
        logger.info(f"ComplexAttention initialized with query_chunk_size: {self.query_chunk_size}")

        self.register_buffer('g', self._get_frequencies()) # 新方式

    def _get_frequencies(self, max_len: int = 5000) -> torch.Tensor:
        # Calculate frequencies (g_i) for positional encoding
        # Shape: (d_k_half)
        freqs: torch.Tensor = 10000 ** (-torch.arange(0, self.d_k // 2, dtype=torch.float) * 2 / self.d_k)
        return freqs # Will be unsqueezed later for broadcasting

    def euler_transform(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch_size, num_heads, seq_len, d_k)
        # or (batch_size * num_heads, seq_len, d_k) if preferred for intermediate steps
        
        # Split into real and imag parts along the last dimension
        # Each part has shape (..., d_k / 2)
        real_part, imag_part = torch.split(x, self.d_k // 2, dim=-1)
        
        magnitude: torch.Tensor = torch.sqrt(real_part**2 + imag_part**2 + 1e-9) # Add epsilon for stability
        phase: torch.Tensor = torch.atan2(imag_part, real_part)
        # magnitude, phase shape: (batch_size, num_heads, seq_len, d_k / 2)
        return magnitude, phase

    def relative_pos_encoding(self, query_len: int, key_len: int, device: torch.device) -> torch.Tensor:
        """
        Calculates a component of relative positional encoding (Delta P).
        This specific formulation suggests it might be part of a scheme like
        the one in Transformer-XL or variants, where relative distances are scaled by frequencies.
        Output shape: (query_len, key_len, d_k_half)
        """
        # Block 1: Calculate Relative Distances
        # m: (query_len, 1), n: (1, key_len)
        # Ensure float type for subsequent multiplication with frequencies.
        m = torch.arange(query_len, device=device, dtype=torch.float32).unsqueeze(1)
        n = torch.arange(key_len, device=device, dtype=torch.float32).unsqueeze(0)
        # relative_distance: (query_len, key_len)
        relative_distance = m - n

        # Block 2: Prepare Frequency Tensor
        # self.g is assumed to be a 1D tensor of shape (d_k_half).
        d_k_half = self.d_k // 2 # Or use self.d_k_half if it's an attribute

        # Ensure self.g is on the correct device and reshape for broadcasting.
        # freqs shape: (1, 1, d_k_half)
        # Note: self.g.to(device) is important if self.g might not be on the target device.
        # If self.g is a registered buffer and always on the correct device, this might be simplified.
        freqs = self.g.to(device,dtype=torch.bfloat16).view(1, 1, d_k_half)

        # Block 3: Calculate Delta P (Positional Encoding Component)
        # Expand relative_distance to (query_len, key_len, 1) for broadcasting.
        # Then multiply with freqs (1, 1, d_k_half)
        # Resulting delta_p shape: (query_len, key_len, d_k_half)
        delta_p = relative_distance.unsqueeze(-1) * freqs

        return delta_p 

        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        q: (batch_size, query_len, d_model)  - 输入的查询序列
        k: (batch_size, key_len, d_model)    - 输入的键序列
        v: (batch_size, value_len, d_model)  - 输入的值序列
        mask: (batch_size, 1, query_len, key_len) or (batch_size, query_len, key_len) - 可选的注意力掩码
        """
        batch_size: int = q.size(0)
        query_len: int = q.size(1)
        key_len: int = k.size(1)
        value_len: int = v.size(1)
        
        # For this specific complex attention, QL, KL, VL are usually same in self-attention context
        # but the formulas allow for QL != KL for cross-attention score part
        # However, V must have same length as K for standard attention (value_len == key_len)
        # The original assert was too strict if we imagine cross-attention scenarios for scores,
        # but for self-attention it's fine. Let's assume self-attention for now based on original assert.
        assert query_len == key_len == value_len, \
            f"Query, Key, and Value must have the same length in this self-attention setup. " \
            f"Got QL={query_len}, KL={key_len}, VL={value_len}"
        
        if torch.isnan(q).any() or torch.isnan(k).any() or torch.isnan(v).any():
            logger.error("Input contains NaN before W_q, W_k, W_v")
            raise ValueError("Input contains NaN")
        
        logger.info(f"Input shapes: q={q.shape}, k={k.shape}, v={v.shape}")

        Q_proj: torch.Tensor = self.W_q(q).view(batch_size, query_len, self.num_heads, self.d_k).transpose(1, 2)
        K_proj: torch.Tensor = self.W_k(k).view(batch_size, key_len, self.num_heads, self.d_k).transpose(1, 2)
        V_proj: torch.Tensor = self.W_v(v).view(batch_size, value_len, self.num_heads, self.d_v).transpose(1, 2)
        # Q_proj, K_proj shapes: (B, H, QL/KL, DK)
        # V_proj shape: (B, H, VL, DV)

        magnitude_q_full, phase_q_full = self.euler_transform(Q_proj) # (B, H, QL, DK/2)
        magnitude_k, phase_k = self.euler_transform(K_proj)           # (B, H, KL, DK/2)

        delta_adapted = self.delta_params.view(1, self.num_heads, 1, 1, self.d_k_half)
        bias_adapted = self.bias_params.view(1, self.num_heads, 1, 1, self.d_k_half)
        
        # Precompute relative positional encoding for all pairs if not chunking too finely
        # If QL is chunked, delta_p needs to be sliced or computed per chunk.
        # For simplicity, compute full delta_p and slice it.
        delta_p_full = self.relative_pos_encoding(query_len, key_len, q.device) # (QL, KL, DK/2)
        delta_p_expanded_full = delta_p_full.unsqueeze(0).unsqueeze(0)          # (1, 1, QL, KL, DK/2)

        # --- Chunked computation for scores ---
        # Determine actual chunk size for query_len
        current_query_chunk_size = self.query_chunk_size
        if current_query_chunk_size is None or current_query_chunk_size >= query_len:
            # No chunking or chunk size covers the whole query_len
            effective_query_chunk_size = query_len
            num_q_chunks = 1
        else:
            effective_query_chunk_size = current_query_chunk_size
            num_q_chunks = (query_len + effective_query_chunk_size - 1) // effective_query_chunk_size
        
        logger.debug(f"Using query_chunk_size: {effective_query_chunk_size}, num_q_chunks: {num_q_chunks}")

        scores_chunks = []
        # phase_k_unsqueezed for broadcasting with phase_q_chunk
        phase_k_unsqueezed = phase_k.unsqueeze(2) # (B, H, 1, KL, DK/2)

        with autocast(dtype=torch.bfloat16):
            for i in range(num_q_chunks):
                q_start = i * effective_query_chunk_size
                q_end = min((i + 1) * effective_query_chunk_size, query_len)
                
                # Slice Q-related tensors for the current chunk
                magnitude_q_chunk = magnitude_q_full[:, :, q_start:q_end, :] # (B, H, chunk_QL, DK/2)
                phase_q_chunk = phase_q_full[:, :, q_start:q_end, :]         # (B, H, chunk_QL, DK/2)
                
                # Calculate as_ for the current chunk
                # phase_q_chunk (B,H,chunk_QL,DK/2) -> unsqueeze(3) -> (B,H,chunk_QL,1,DK/2)
                # phase_k_unsqueezed (B,H,1,KL,DK/2)
                # as_chunk shape: (B, H, chunk_QL, KL, DK/2)
                as_chunk: torch.Tensor = phase_q_chunk.unsqueeze(3) - phase_k_unsqueezed

                # adapt_as_chunk shape: (B, H, chunk_QL, KL, DK/2)
                adapt_as_chunk: torch.Tensor = delta_adapted * as_chunk + bias_adapted
                
                # Slice delta_p_expanded for the current chunk
                # delta_p_expanded_full: (1, 1, QL, KL, DK/2)
                delta_p_expanded_chunk = delta_p_expanded_full[:, :, q_start:q_end, :, :] # (1,1,chunk_QL,KL,DK/2)

                # combined_phase_chunk shape: (B, H, chunk_QL, KL, DK/2)
                combined_phase_chunk: torch.Tensor = adapt_as_chunk + delta_p_expanded_chunk
                cos_combined_phase_chunk = torch.cos(combined_phase_chunk)

                # scores_current_chunk shape: (B, H, chunk_QL, KL)
                
                scores_current_chunk = torch.einsum('bhqd, bhkd, bhqkd -> bhqk',
                                            magnitude_q_chunk,    # (B, H, chunk_QL, DK_half)
                                            magnitude_k,          # (B, H, KL, DK_half) - full K used
                                            cos_combined_phase_chunk # (B, H, chunk_QL, KL, DK_half)
                                            )
                scores_chunks.append(scores_current_chunk)

            if num_q_chunks == 1:
                scores = scores_chunks[0]
            else:
                scores = torch.cat(scores_chunks, dim=2) # Concatenate along the query_len dimension

            # --- End of chunked computation ---

            if torch.isnan(scores).any():
                logger.error("Scores contains NaN after complex_attention_score logic (chunked or not)")
                raise ValueError("Scores contains NaN")

            # Apply mask (if provided)
            if mask is not None:
                # Ensure mask has 4 dimensions: (B, 1, QL, KL) or (B, H, QL, KL)
                if mask.dim() == 3: # (B, QL, KL)
                    mask = mask.unsqueeze(1) # (B, 1, QL, KL)
                if mask.dim() == 2:
                    mask = mask.unsqueeze(1).unsqueeze(1)
                # Mask should be (B, 1, QL, KL) or (B, H, QL, KL)
                # Scores are (B, H, QL, KL)
                # Mask values are typically 0 for positions to attend to, and -inf (or large negative) for masked positions.
                # Or, if mask is boolean (True for masked), convert it.
                # Assuming mask uses -inf for masked positions:
                assert mask.dim()==4, f"Mask should have 4 dimensions (B, 1, QL, KL) or (B, H, QL, KL), got {mask.dim()} dimensions"
                scores = scores + mask # Broadcasting if mask is (B, 1, QL, KL)

            # Apply softmax
            
            attention_weights = self.softmax(scores) # (B, H, QL, KL)
            
            # Apply dropout to attention weights
            attention_weights = self.dropout(attention_weights)

            # Multiply by V
            # V_proj is (B, H, VL, DV). Since VL==KL for self-attention, this is (B, H, KL, DV)
            # attention_weights is (B, H, QL, KL)
            # output should be (B, H, QL, DV)
            output = torch.matmul(attention_weights, V_proj) # (B, H, QL, DV)

            # Concatenate heads and project
            # transpose back to (B, QL, H, DV) then reshape to (B, QL, H*DV = D_MODEL)
            #print(f"Before transpose: output.shape = {output.shape}, output.numel() = {output.numel()}")
            output = output.transpose(1, 2).contiguous().view(batch_size, query_len, self.d_model)
            output = self.W_o(output) # Final linear projection

            if torch.isnan(output).any():
                logger.error("Output contains NaN after W_o")
                raise ValueError("Output contains NaN")
            
        return output
if __name__ == '__main__':
    batch_size = 2
    seq_len_q = 512# Query length
    seq_len_kv = 512 # Key/Value length
    d_model = 256
    num_heads = 8

    # Test case: Different Q and K/V lengths (like in encoder-decoder attention, but this is self-attn module)
    # For this module as self-attention, seq_len_q should typically equal seq_len_kv
    # If we want to test cross-attention, the logic for relative_pos_encoding and mask might need adjustment,
    # but the core vectorized score calculation should still work if QL, KL, VL are handled.
    # Let's assume self-attention for now where QL=KL=VL for simplicity of testing this module.
    # If using for cross-attention, ensure value_len in V matches key_len in K for matmul.
    # The current error RuntimeError: The size of tensor a (9) must match the size of tensor b (10) at non-singleton dimension 2
    # likely comes from mask (key_len=10) and scores (key_len=9).

    # To reproduce the error from the user:
    # query_len = 9
    # key_len_for_k_and_mask = 10 # This will cause a mismatch if k is based on a different length

    # Let's simulate the scenario where k has length 9, but mask implies length 10
    # This means scores will have key_len 9, but mask will target key_len 10.
    
    actual_key_len_for_k = seq_len_kv # This will be k.size(1)
    mask_implied_key_len = seq_len_q

    q_tensor = torch.randn(batch_size, seq_len_q, d_model)
    k_tensor = torch.randn(batch_size, seq_len_kv, d_model) # K has length 9
    v_tensor = torch.randn(batch_size, seq_len_kv, d_model) # V also has length 9 (must match K for matmul)

    # Mask is True for valid tokens. Shape (B, 1, QL, KL_mask)
    # Here, KL_mask is 10, but K's actual length is 9. This is the problem source.
    attention_mask = torch.ones(batch_size, 1, seq_len_q, mask_implied_key_len, dtype=torch.bool)
    # Make some parts of the mask False (padding) to test masking
    if mask_implied_key_len > 0:
        attention_mask[:, :, :, -1] = False # Last key position is padding for all queries

    logger.info("--- Testing ComplexMultiHeadAttentionV2Vectorized ---")
    cmha_vec = ComplexMultiHeadAttentionV2(d_model, num_heads)

    # Test with inputs that would cause the original error (scores key_len != mask key_len)
    # To do this correctly with the vectorized version, the K projection and scores
    # will naturally have key_len = actual_key_len_for_k.
    # The mask, if its key_len dimension is mask_implied_key_len, will cause an error
    # during the `scores.masked_fill(mask == 0, ...)` if their key_len dims don't match.

    try:
        print(f"Input q shape: {q_tensor.shape}")
        print(f"Input k shape: {k_tensor.shape}")
        print(f"Input v shape: {v_tensor.shape}")
        print(f"Input attention_mask shape: {attention_mask.shape}")
        output_vec = cmha_vec(q_tensor, k_tensor, v_tensor, mask=attention_mask)
        print("Vectorized Output shape:", output_vec.shape)
    except RuntimeError as e:
        print(f"Vectorized version caught RuntimeError: {e}")
        print("This error is expected if actual_key_len_for_k (from k_tensor) which determines scores.shape[-1] "
              "does not match mask_implied_key_len (from attention_mask) which determines mask.shape[-1].")
        # In the call scores.masked_fill(mask == 0, ...):
        # scores will have shape (B, H, QL, actual_key_len_for_k)
        # mask (after unsqueeze if needed) will be (B, 1 or H, QL, mask_implied_key_len)
        # If actual_key_len_for_k != mask_implied_key_len, this will error.

    print("\n--- Testing with MATCHING key lengths for K and Mask ---")
    # Now test with consistent key lengths
    consistent_key_len = seq_len_q
    q_tensor_c = torch.randn(batch_size, seq_len_q, d_model)
    k_tensor_c = torch.randn(batch_size, consistent_key_len, d_model)
    v_tensor_c = torch.randn(batch_size, consistent_key_len, d_model) # V length must match K length
    attention_mask_c = torch.ones(batch_size, 1, seq_len_q, consistent_key_len, dtype=torch.bool)
    if consistent_key_len > 1:
        attention_mask_c[:, :, :, -2:] = False # Mask last two key positions

    try:
        print(f"Input q_c shape: {q_tensor_c.shape}")
        print(f"Input k_c shape: {k_tensor_c.shape}")
        print(f"Input v_c shape: {v_tensor_c.shape}")
        print(f"Input attention_mask_c shape: {attention_mask_c.shape}")
        output_vec_c = cmha_vec(q_tensor_c, k_tensor_c, v_tensor_c, mask=attention_mask_c)
        print("Vectorized Consistent Output shape:", output_vec_c.shape)
        assert output_vec_c.shape == (batch_size, seq_len_q, d_model)
        print("Test with matching key lengths PASSED.")
    except Exception as e:
        print(f"Vectorized version with matching key lengths FAILED: {e}")


    print("\n--- Testing without mask ---")
    try:
        output_no_mask = cmha_vec(q_tensor_c, k_tensor_c, v_tensor_c, mask=None)
        print("Vectorized No Mask Output shape:", output_no_mask.shape)
        assert output_no_mask.shape == (batch_size, seq_len_q, d_model)
        print("Test without mask PASSED.")
    except Exception as e:
        print(f"Vectorized version without mask FAILED: {e}")