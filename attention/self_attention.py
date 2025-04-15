import torch
from torch import nn

class SelfAttention(nn.Module):
    def __init__(self,embedding_dim):
        super(SelfAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.query_proj = nn.Linear(embedding_dim, embedding_dim)
        self.key_proj = nn.Linear(embedding_dim, embedding_dim)
        self.value_proj = nn.Linear(embedding_dim, embedding_dim)

        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):  
        # x shape: (batch_size, seq_len, embedding_dim)
        batch_size, seq_len, _ = x.size()

        # Linear projections
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)

        # Attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2,-1)) / (self.embedding_dim ** 0.5)
        attention_weights = self.softmax(attention_scores)

        atten_output = torch.matmul(attention_weights, V) # shape: (batch_size, seq_len, embedding_dim)
        # Linear output projection
        atten_output = self.out_proj(atten_output)
        return atten_output
