import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self,embed_dim,num_heads,dropout = 0.1):
        """
        Multi-head attention module.
        Q = (batch,seq_len,embed_dim)
        """
        super(MultiHeadAttention,self).__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        #1.linear transformations for Q,K,V
        self.q_proj = nn.Linear(embed_dim,embed_dim)
        self.k_proj = nn.Linear(embed_dim,embed_dim)   
        self.v_proj = nn.Linear(embed_dim,embed_dim)

        self.out_proj = nn.Linear(embed_dim,embed_dim)
    def forward(self,query,key,value,mask = None):
        batch_size = query.size(0)
        seq_len = query.size(1)

        Q = self.q_proj(query).view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        K = self.k_proj(key).view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)       
        V = self.v_proj(value).view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)

        #Q,K,V = [batch_size,num_heads,seq_len,head_dim]

        #2.scaled dot-product attention
        attn_weights = torch.matmul(Q,K.transpose(-2,-1))/(self.head_dim**0.5)

        #3softmax
        attn_weights = torch.softmax(attn_weights,dim=-1)

        #4.attention
        attention = torch.matmul(attn_weights,V)     

        #5.concat multi head attention=[batch_size,num_heads,seq_len,head_dim]
        attention_output = attention.transpose(1,2).contiguous().view(batch_size,seq_len,self.embed_dim)
        #6.attention = [batch_size,seq_len,embed_dim]

        #7.linear transformation
        output = self.out_proj(attention_output)
        #8.output = [batch_size,seq_len,embed_dim]
        return output


if __name__ == "__main__":
    btch_size = 2
    seq_len = 10
    embed_dim = 512
    num_heads = 8
    mha = MultiHeadAttention(embed_dim,num_heads)
    query = torch.randn(btch_size,seq_len,embed_dim)
    key = torch.randn(btch_size,seq_len,embed_dim) 
    value = torch.randn(btch_size,seq_len,embed_dim)

    output = mha(query,key,value)
    print("Output shape:",output.shape)
    # Output shape: torch.Size([batch_size, seq_len, embed_dim])

