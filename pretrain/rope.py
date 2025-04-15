import torch
import math

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x, freqs):
    cosθ = freqs.cos()  # (seq_len, dim//2)
    sinθ = freqs.sin()  # (seq_len, dim//2)
    rotated_x = (x * cosθ) + (rotate_half(x) * sinθ)
    return rotated_x


