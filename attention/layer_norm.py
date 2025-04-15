import torch
from torch import nn

class LayerNorm(nn.Module):
    def __init__(self, embedding_dim, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(embedding_dim))
        self.b_2 = nn.Parameter(torch.zeros(embedding_dim))
        self.eps = eps

    def forward(self, x):
        # x shape: (batch_size, seq_len, embedding_dim)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x = (x-mean)/(std + self.eps)
        # Apply the learnable parameters
        norm = self.a_2 * norm + self.b_2

        return norm
