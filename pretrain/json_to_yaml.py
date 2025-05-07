import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchviz
from torch.utils.tensorboard import SummaryWriter
import yaml
import argparse
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig, AutoModel
import logging
import hydra
import torch.utils.checkpoint as checkpoint
from typing import Optional, Dict, Any
from rotary_embedding_torch import RotaryEmbedding
from rich import traceback
traceback.install(show_locals=True)

logger: logging.Logger = logging.getLogger(__name__)
logger.propagate = False
logger.addHandler(logging.FileHandler("logs/model.log"))

class CustomConfig(PretrainedConfig):
    model_type: str = "autodiffusion"

    def __init__(self,
                 vocab_size: int = 30522,
                 hidden_dim: int = 512,
                 intermediate_size: int = 1024,
                 max_seq_len: int = 512,
                 n_layers: int = 8,
                 num_attention_heads: int = 8,
                 dropout: float = 0.0,
                 **kwargs: Any):
        super().__init__(**kwargs)
        self.vocab_size: int = vocab_size
        self.hidden_dim: int = hidden_dim
        self.intermediate_size: int = intermediate_size
        self.max_seq_len: int = max_seq_len
        self.n_layers: int = n_layers
        self.num_attention_heads: int = num_attention_heads
        self.dropout: float = dropout
        self.debug : bool = kwargs.get("debug", False)


class MyDecoderOnlyModel(PreTrainedModel):
    config_class = CustomConfig
    #

    def __init__(self, config: CustomConfig):
        super().__init__(config)
        self.config: CustomConfig = config
        self.embedding: nn.Embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.head_dim = config.hidden_dim // config.num_attention_heads
        self.transformer_blocks: nn.ModuleList = nn.ModuleList(
            [TransformerBlock(config)]
        )
        self.linear: nn.Linear = nn.Linear(config.hidden_dim, config.vocab_size)
        self.softmax: nn.Softmax = nn.Softmax(dim=-1)

        #
        # 添加旋转位置编码
        self.rope = RotaryEmbedding(dim=self.head_dim)
        self.position_embedding: nn.Embedding = nn.Embedding(config.max_seq_len, config.hidden_dim)
        
        self.gradient_checkpointing = False
        self.debug = config.debug
        self._togger_loger()
    def forward(self, input_ids: Tensor, attention_mask: Tensor, labels: Optional[Tensor] = None, use_checkpoint: bool = False,token_type_ids = None) -> Tensor:
        #
        if torch.isnan(input_ids).any() or torch.isinf(input_ids).any():
            raise ValueError("Tensor contains NaN or Inf values.")

        if (input_ids >= self.config.vocab_size).any() or (input_ids < 0).any():
            raise ValueError("input_ids contain values outside the valid range.")
        seq_len: int = input_ids.size(1)
        device = next(self.parameters()).device
        #logger.info(f"Input IDs shape: {input_ids.device}")
        # 生成位置索引
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
        # 获取位置编码
        position_ids = position_ids.to(device)
        #logger.info(f"Position IDs shape: {position_ids.device}")
        assert position_ids.device == self.position_embedding.weight.device, "position_ids and position_embedding must be on the same device."
        position_embeddings = self.position_embedding(position_ids).to(device)
        #logger.info(f"Position Embeddings shape: {position_embeddings.device}")

        # 将位置编码添加到输入嵌入
        ## 确保设备一致性
        position_embeddings = position_embeddings.to(device)
        token_embeddings = self.embedding(input_ids).to(device)
        #
        assert token_embeddings.device == position_embeddings.device, "input_ids and position_embedding must be on the same device."
        logger.info(f"Token Embeddings shape: {token_embeddings.device}")
        #logger.info(f"Token Embeddings shape: {token_embeddings.shape}")
        logger.info(f"Position Embeddings shape: {position_embeddings.device}")
        x: Tensor = token_embeddings + position_embeddings
        #logger.info(f"Input shape: {x.shape}")
        #
        # 应用旋转位置编码
        # x = x.view(batch_size, seq_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        # x = self.rope(x, seq_len=seq_len)  # 应用 RoPE
        # x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)  # 恢复形状

        for block in self.transformer_blocks:
            if self.gradient_checkpointing:
                def create_custom_forward(module):
                    # 这个函数用来捕获外部变量
                    def custom_forward(*inputs): # inputs[0] 会是 x
                        # captured_key_padding_mask 是从外面捕获的 mask
                        return module(inputs[0], padding_mask=captured_key_padding_mask)
                    return custom_forward

                captured_key_padding_mask = ~attention_mask # 捕获当前的 mask
                layer_outputs = checkpoint.checkpoint(
                    create_custom_forward(block), # 包装后的函数
                    x,                          # 只有 x (需要梯度) 直接传入
                    use_reentrant=False,
                    preserve_rng_state=True # 保证 dropout 等随机性一致
                )
                x = layer_outputs
                logger.info(f"Transformer block output shape with ckpt: {x.shape}")
            else:
                x = block(x, padding_mask=~attention_mask)
                logger.info(f"Transformer block output shape no ckpt: {x.shape}")
        logger.info(f"Transformer block output shape: {x.shape}")
        x = self.linear(x)
        logger.info(f"Linear layer output shape: {x.shape}")
        x = self.softmax(x)
        logger.info(f"Softmax output shape: {x.shape}")
       # assert x.shape == (self.config.batch_size, self.config.max_seq_len, self.config.vocab_size), f"Output shape mismatch{self.config.batch_size},{self.config.max_seq_len}, {self.config.vocab_size} is  {x.shape}"
        return x

    def gradient_checkpointing_enable(self,**kwargs):
        self.gradient_checkpointing = True
    
    def _togger_loger(self):
        if self.debug == False:
            logger.setLevel(logging.WARNING)
class FFN(nn.Module):
    def __init__(self, config: CustomConfig):
        super().__init__()
        self.linear1: nn.Linear = nn.Linear(config.hidden_dim, config.intermediate_size)
        self.linear3: nn.Linear = nn.Linear(config.intermediate_size, config.hidden_dim)
        self.linear2: nn.Linear = nn.Linear(config.hidden_dim, config.intermediate_size)
        self.activation: nn.Module = nn.SiLU()  # or any other activation function

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear1(x) * self.activation(self.linear2(x))
        x = self.linear3(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.weight: nn.Parameter = nn.Parameter(torch.ones(hidden_size))
        self.eps: float = 1e-5

    def forward(self, x: Tensor) -> Tensor:
        return x * (torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True)) * self.weight + self.eps)


class TransformerBlock(nn.Module):
    def __init__(self, config: CustomConfig):
        super().__init__()
        self.attention: nn.MultiheadAttention = nn.MultiheadAttention(
            config.hidden_dim, config.num_attention_heads, dropout=config.dropout, batch_first=True
        )
        self.ffn: FFN = FFN(config)
        self.rmsnorm: RMSNorm = RMSNorm(config.hidden_dim)

    def forward(self, x: Tensor, padding_mask: Tensor) -> Tensor:
        residual: Tensor = x
        logger.info(f"TransformerBlock Input shape: {x.shape}")
        seq_len: int = x.shape[1]
        causal_mask: Tensor = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)

        def atten_forward(x: Tensor) -> Tensor:
            return self.attention(x, x, x, attn_mask=causal_mask)[0]
        
        x = atten_forward(x)
        x = self.rmsnorm(x + residual)

        residual = x
        x = self.ffn(x) 
        x = self.rmsnorm(x + residual)
        return x


class MlutiLatentAtten(nn.Module):
    def __init__(self, config: CustomConfig, use_cache: bool = True, dropout: float = 0.01):
        super().__init__()
        self.config: CustomConfig = config
        assert config.hidden_dim % config.num_attention_heads == 0, "hidden_dim must be divisible by num_attention_heads"
        self.q_proj: nn.Linear = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.k_proj: nn.Linear = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.v_proj: nn.Linear = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.out_proj: nn.Linear = nn.Linear(config.hidden_dim, config.hidden_dim)

        self.embed_dim: float = config.hidden_dim / config.num_attention_heads
        self.softmax: nn.Softmax = nn.Softmax(dim=-1)

        self.use_cache: bool = use_cache
        self.k_cache: nn.Parameter = nn.Parameter(torch.zeros(config.num_attention_heads, config.hidden_dim), requires_grad=False)
        self.v_cache: nn.Parameter = nn.Parameter(torch.zeros(config.num_attention_heads, config.hidden_dim), requires_grad=False)

    def forward(self, x: Tensor, attention_mask: Tensor) -> Tensor:
        batch_size, seq_len, hidden_dim = x.size()
        q: Tensor = self.q_proj(x).view(batch_size, seq_len, self.config.num_attention_heads, self.embed_dim).transpose(1, 2)
        k: Tensor = self.k_proj(x).view(batch_size, seq_len, self.config.num_attention_heads, self.embed_dim).transpose(1, 2)
        v: Tensor = self.v_proj(x).view(batch_size, seq_len, self.config.num_attention_heads, self.embed_dim).transpose(1, 2)

        from rotary_embedding_torch import RotaryEmbedding
        rope = RotaryEmbedding(dim=self.embed_dim, base=10000)
        q = rope(q, seq_len=seq_len)
        k = rope(k, seq_len=seq_len)

        if self.use_cache:
            k = torch.cat([self.k_cache, k], dim=1)
            v = torch.cat([self.v_cache, v], dim=1)
            self.k_cache = k
            self.v_cache = v

        scale: Tensor = torch.sqrt(torch.tensor(x.size(-1), dtype=torch.float32, device=x.device))
        attn_weights: Tensor = self.softmax(torch.matmul(q, k.transpose(-2, -1)) / scale)
        attn_weights = attn_weights * attention_mask

        atten_output: Tensor = torch.matmul(attn_weights, v)
        atten_output = atten_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        atten_output = self.out_proj(atten_output)
        return atten_output


def load_config(config_path: str) -> CustomConfig:
    with open(config_path, 'r') as f:
        config_dict: Dict[str, Any] = yaml.safe_load(f)
    return CustomConfig(**config_dict)


@hydra.main(config_path='.', config_name="config.yaml")
def main(config: Dict[str, Any]) -> None:
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str, default='./pretrain/config.yaml', help='Path to the config file')
    args = argparser.parse_args()

    AutoConfig.register("autodiffusion", CustomConfig)
    AutoModel.register(CustomConfig, MyDecoderOnlyModel)

    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model: MyDecoderOnlyModel = MyDecoderOnlyModel(config=load_config(args.config)).to(device)
    test_model(model,config)

    model.gradient_checkpointing_enable()
    model.save_pretrained(config.model.save_dir)
    logger.info(f"Model saved to {config.model.save_dir}")

     # --- Calculate Parameters ---
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad) # Count only trainable
    logger.info(f"Trainable model parameters: {num_params/1e9:,}B")


def test_model(model: MyDecoderOnlyModel,config) -> None:
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Testing model...{device}")

    input_ids: Tensor = torch.randint(0, config.vocab_size, (config.batch_size, config.max_seq_len)).to(device)
    attention_mask: Tensor = torch.ones((1, 10)).bool().to(device)
    labels: Tensor = torch.randint(0, config.vocab_size, (config.batch_size, config.max_seq_len)).to(device)
    output: Tensor = model(input_ids, attention_mask=attention_mask, labels=labels, use_checkpoint=True).to(device)
    logger.info(f"Model output shape: {output.shape}")
    assert output.shape == (config.batch_size, config.max_seq_len, config.vocab_size), "Output shape mismatch"

    # 使用损失函数
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output.view(-1, config.vocab_size), labels.view(-1))  # 确保形状匹配
    logger.info(f"Loss: {loss.dim()}")

if __name__ == '__main__':
    main()