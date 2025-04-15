import torch
from torch import nn

class CrossAttention(nn.Module):
    def __init__(self, embedding_dim,dropout=0.1):
        self.W_Q = nn.Linear(embedding_dim, embedding_dim)
        self.W_K = nn.Linear(embedding_dim, embedding_dim)
        self.W_V = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, y):
        # x shape: (batch_size, seq_len_x, embedding_dim)
        # y shape: (batch_size, seq_len_y, embedding_dim)
        Q = self.W_Q(x)
        K = self.W_K(y)
        V = self.W_V(y)
        # Attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2,-1)) / (self.embedding_dim ** 0.5)
        attention_weights = self.softmax(attention_scores)
        attention_weights = self.dropout(attention_weights)

        # Context vector
        context_vector = torch.matmul(attention_weights, V)

        # Output
        output = self.out_proj(context_vector)
        return output
        