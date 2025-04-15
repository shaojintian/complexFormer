import torch

from torch import nn

class BatchNorm(nn.Module):
    def __init__(self, embedding_dim, eps=1e-6):
        self.gamma = nn.Parameter(torch.ones(embedding_dim))
        self.beta = nn.Parameter(torch.zeros(embedding_dim))
        self.eps = eps

    def forward(self, x):
        # x shape: (batch_size, seq_len, embedding_dim)
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        x = (x - mean) / (std + self.eps)
        norm = self.gamma * x + self.beta

        return norm