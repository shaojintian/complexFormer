import torch
import torch.nn as nn
import torch.nn.functional as F
import torchviz 
from torch.utils.tensorboard import SummaryWriter
import yaml
import argparse

class MyDecoderOnlyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config['vocab_size'], config['hidden_dim'])
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(config)]
        )
        self.linear = nn.Linear(config["hidden_dim"], config["vocab_size"])
        self.softmax = nn.Softmax(dim=-1)
        

    def forward(self, input_ids, attention_mask, labels=None):
        x = self.embedding(input_ids)
        for block in self.transformer_blocks:
            x = block(x, attention_mask=~attention_mask)
        x = self.linear(x)
        x = self.softmax(x)
        return x



class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config["hidden_dim"], config["intermediate_size"])
        self.linear3 = nn.Linear(config["intermediate_size"], config["hidden_dim"])
        self.linear2 = nn.Linear(config["hidden_dim"], config["intermediate_size"])
        self.activation = nn.SiLU()  # or any other activation function
        #self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        x = self.linear1(x) * self.activation(self.linear2(x))
        x = self.linear3(x)
        #x = self.dropout(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = 1e-5

    def forward(self, x):
        return x * (torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True)) * self.weight + self.eps)
    

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        #TODO:optimize mha
        self.attention = nn.MultiheadAttention(config["hidden_dim"], config["num_attention_heads"], dropout=config["dropout"])
        self.ffn = FFN(config)
        self.rmsnorm = RMSNorm(config["hidden_dim"])

    def forward(self, x, attention_mask):
        x = self.attention(x, x, x, attn_mask=attention_mask)[0]
        x = self.rmsnorm(x)
        x = self.ffn(x)
        return x
    

class MlutiLatentAtten(nn.Module):
    def __size__(self):
        pass
    def __init__(self, config,use_cache=True,drooutp=0.01):
        super().__init__()

        self.config = config
        assert config["hidden_dim"] % config["num_attention_heads"] == 0, "hidden_dim must be divisible by num_attention_heads"
        self.q_proj = nn.Linear(config["hidden_dim"], config["hiddendim"])
        self.k_proj = nn.Linear(config["hidden_dim"], config["hidden_dim"])
        self.v_proj = nn.Linear(config["hidden_dim"], config["hidden_dim"])
        self.out_proj = nn.Linear(config["hidden_dim"], config["hidden_dim"])
        
        self.embed_dim = config["hidden_dim"] / config["num_attention_heads"]
        self.softmax = nn.Softmax(dim=-1)
        
        self.use_cache = use_cache
        self.k_cache = nn.Parameter(torch.zeros(config["num_attention_heads"], config["hidden_dim"]), requires_grad=False)
        self.v_cache = nn.Parameter(torch.zeros(config["num_attention_heads"], config["hidden_dim"]), requires_grad=False)
    def forward(self, x, attention_mask):
        # x shape: (batch_size, seq_len, hidden_dim)

        batch_size, seq_len, hidden_dim = x.size()
        q = self.q_proj(x).view(batch_size, seq_len, self.config["num_attention_heads"], self.embed_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.config["num_attention_heads"], self.embed_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.config["num_attention_heads"], self.embed_dim).transpose(1, 2)
        # q, k, v shape: (batch_size, num_attention_heads, seq_len, embed_dim)
        from rotary_embedding_torch import RotaryEmbedding
        rope = RotaryEmbedding(dim=self.embed_dim, base=10000)
        q = rope(x, seq_len=seq_len)
        k = rope(x, seq_len=seq_len)

        if self.use_cache:
            k = torch.cat([self.k_cache, k], dim=1)
            v = torch.cat([self.v_cache, v], dim=1)
            self.k_cache = k
            self.v_cache = v


        attn_weights = self.softmax(torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(x.size(-1), dtype=torch.float32)))
        attn_weights = attn_weights * attention_mask

        atten_output = torch.matmul(attn_weights, v)
        #atten_output shape: (batch_size, num_attention_heads, seq_len, embed_dim)
        atten_output = atten_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        atten_output = self.out_proj(atten_output)
        return atten_output
        attn_output = torch.matmul(attn_weights, v)
    

def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    args = argparser.parse_args()

    model = MyDecoderOnlyModel(config=load_config(args.config))
    #model = [batch_size,seq_len]
    print(model(torch.randint(0, 10000, (1, 10)), torch.ones((1, 1)).bool()).shape)
    m = model(torch.randint(0, 10000, (1, 10)), torch.ones((1, 1)).bool()).shape
    #[batch_size, seq_len, vocab_size]
    torchviz.make_dot(model)
    # writer = SummaryWriter()
    # writer.add_graph(model, (torch.randint(0, 10000, (1, 10)), torch.ones((1, 1)).bool()))

    # writer.close()
    