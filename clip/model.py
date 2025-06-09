import torch
from torch import nn
from torch.functional import F
from transformers import AutoTokenizer, AutoModel,ViTPreTrainedModel
import logging
from transformers import ViTModel, ViTPreTrainedModel, ViTConfig,CLIPConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling # ViTModel 的输出类型
from PIL import Image
import hydra 
from datasets import load_dataset
logger: logging.Logger = logging.getLogger(__name__)

class CLIPModel(nn.Module):
    def __init__(
        self,
        config:CLIPConfig
    ):
        super().__init__()
        self.image_encoder = ImageEncoder.from_pretrained(
            config.image.encoder.pretrained_model_name,
            cache_dir=config.image.encoder.cache,
        embedding_dim=512, # 假设我们想将输出投影到 512 维
    )#(batch_size, 3, H, W) 
        self.text_encoder = TextEncoder(config=config)
        self.image_projection = ProjectionHead(embedding_dim=config.image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=config.text_embedding)
        self.temperature = config.temperature

    def forward(self, batch):
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"]) #[B,C,H,W]
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )#[B,S,D_text] 
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features) #[B,D] CLS token
        text_embeddings = self.text_projection(text_features) #[B,D] EOS token

        # Calculating the Loss

        #prediction
        logits = (text_embeddings @ image_embeddings.T) / self.temperature #nxn
        

        #true
        images_similarity = image_embeddings @ image_embeddings.T 
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    

class ImageEncoder(ViTPreTrainedModel):
    def __init__(self, config: ViTConfig, embedding_dim: int = None, add_pooling_layer: bool = True):
        """
        图像编码器，基于 ViT (Vision Transformer)。

        参数:
            config (ViTConfig): ViT 模型的配置对象。
            embedding_dim (int, optional):
                如果提供，将在 ViT 的输出之上添加一个线性层，将特征投影到这个维度。
                如果为 None，则直接使用 ViT 的池化输出 (通常是 CLS token 的表示)。
            add_pooling_layer (bool):
                ViTModel 默认会有一个池化层 (取 CLS token)。
                这个参数在这里主要是为了概念上的清晰，因为 ViTModel 内部已经处理了。
        """
        super().__init__(config)
        self.config = config
        self.vit = ViTModel(config, add_pooling_layer=add_pooling_layer) # ViTModel 是核心

        self.projection = None
        if embedding_dim is not None and embedding_dim > 0:
            self.projection = nn.Linear(config.hidden_size, embedding_dim)
            # 你也可以在这里添加其他的层，比如归一化层
            # self.projection = nn.Sequential(
            #     nn.Linear(config.hidden_size, embedding_dim),
            #     nn.LayerNorm(embedding_dim) # 可选
            # )
            self.output_embedding_dim = embedding_dim
        else:
            self.output_embedding_dim = config.hidden_size


    def forward(
        self,
        pixel_values: torch.Tensor = None, # (batch_size, num_channels, height, width)
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
    ) -> torch.Tensor | BaseModelOutputWithPooling: # 返回的类型取决于是否做了投影
        """
        前向传播。

        参数:
            pixel_values (torch.Tensor):
                图像的像素值，通常由 ViTFeatureExtractor (或 ViTImageProcessor) 预处理得到。
                形状: (batch_size, num_channels, height, width)。
            output_attentions (bool, optional): 是否返回所有自注意力层的注意力权重。
            output_hidden_states (bool, optional): 是否返回所有隐藏层的状态。
            return_dict (bool, optional): 是否返回一个 BaseModelOutputWithPooling 对象而不是元组。

        返回:
            torch.Tensor:
                如果 self.projection 被定义，则返回投影后的图像嵌入，形状为 (batch_size, embedding_dim)。
                否则，返回 ViT 模型的池化器输出 (通常是 CLS token 的表示)，形状为 (batch_size, config.hidden_size)。
            BaseModelOutputWithPooling: (如果 return_dict=True 且没有投影)
                ViT 模型的标准输出。
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        # 1. 通过 ViT 模型获取特征
        # ViTModel 的输出是一个 BaseModelOutputWithPooling 对象 (如果 return_dict=True)
        # 或者是一个包含 last_hidden_state, pooler_output, hidden_states, attentions 的元组
        vit_outputs: BaseModelOutputWithPooling = self.vit(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True, # 强制使用 dict 以方便访问 pooler_output
        )

        # 2. 获取池化后的输出 (通常是 CLS token 的表示)
        # pooler_output 的形状是 (batch_size, config.hidden_size)
        pooled_output = vit_outputs.pooler_output

        # 3. (可选) 应用投影层
        if self.projection is not None:
            image_embeddings = self.projection(pooled_output)
            if not return_dict: # 如果调用者不想要字典，并且我们做了投影
                return (image_embeddings,) + (vit_outputs.last_hidden_state, vit_outputs.hidden_states, vit_outputs.attentions)
            # 如果做了投影，通常我们只关心最终的 image_embeddings
            # 为了简单起见，如果做了投影，我们就直接返回这个 tensor
            # 或者你可以构建一个新的 Output 对象
            return image_embeddings # (batch_size, embedding_dim)
        else:
            # 如果没有投影层，并且调用者要求字典，则返回 ViT 的原始输出
            if return_dict:
                return vit_outputs
            # 否则返回元组，其中第一个元素是池化输出
            return (pooled_output,) + (vit_outputs.last_hidden_state, vit_outputs.hidden_states, vit_outputs.attentions)



class TextEncoder: # 通常我们不直接继承 AutoTokenizer 来创建自定义编码器类
    def __init__(self, config,pretrained_model_name_or_path="bert-base-uncased"):
        """
        初始化文本编码器，加载预训练的 AutoTokenizer。

        参数:
            pretrained_model_name_or_path (str):
                预训练模型的名称 (例如 "bert-base-uncased")
                或包含模型配置和权重的本地路径。
        """
        # AutoTokenizer.from_pretrained() 会返回一个具体的 Tokenizer 实例
        # 例如 BertTokenizerFast, GPT2TokenizerFast 等
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path,cache_dir=config.tokenrizer.cache)
        logger.info(f"Tokenizer '{pretrained_model_name_or_path}' loaded successfully.")
        logger.info(f"Underlying tokenizer type: {type(self.tokenizer)}")

    def encode(self, text, **kwargs):
        """
        使用加载的分词器对文本进行编码。
        kwargs 可以传递给 tokenizer 的 __call__ 方法，例如 max_length, padding, truncation等。
        """
        if not isinstance(text, (str, list)):
            raise ValueError("Input text must be a string or a list of strings.")

        # tokenizer() 是 __call__ 方法的简写
        return self.tokenizer(text, **kwargs) #[batch_size,sql_length]

    def decode(self, token_ids, **kwargs):
        """
        使用加载的分词器对 token IDs 进行解码。
        """
        return self.tokenizer.decode(token_ids, **kwargs)

    # 你可以根据需要添加更多的方法，例如获取词汇表大小等
    def get_vocab_size(self):
        return self.tokenizer.vocab_size

    def get_special_tokens_mask(self, *args, **kwargs):
        return self.tokenizer.get_special_tokens_mask(*args, **kwargs)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        初始化 ProjectionHead。

        参数:
            input_dim (int): 输入特征的维度 (例如，编码器的输出维度)。
            hidden_dim (int): 投影头隐藏层的维度。
            output_dim (int): 输出投影特征的维度。
        """
        super().__init__() # 调用父类 nn.Module 的构造函数

        # 定义网络层
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim) # 适用于 (batch_size, features) 的输入
        self.silu = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # 有些实现可能更简单，例如只有一层线性层，或者没有BN
        # self.projection = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, output_dim)
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        定义前向传播。

        参数:
            x (torch.Tensor): 输入特征，形状通常为 (batch_size, input_dim)。

        返回:
            torch.Tensor: 投影后的特征，形状为 (batch_size, output_dim)。
        """
        x = self.fc1(x)
        # BatchNorm1d 期望 (N, C) 或 (N, C, L)，这里 x 是 (N, hidden_dim)
        if x.ndim > 2 and x.shape[1] == self.bn.num_features: # 如果输入是序列 (N, L, C) -> (N, C, L)
             x = x.permute(0, 2, 1)
             x = self.bn(x)
             x = x.permute(0, 2, 1)
        elif x.ndim == 2: # (N, C)
            x = self.bn(x)
        else: # 如果维度不匹配BN，可以选择跳过或报错
            # print("Warning: Skipping BatchNorm in ProjectionHead due to input dimensions.")
            pass


        x = self.silu(x)
        x = self.fc2(x)
        return x



@hydra.main(
    config_path=".",
    config_name="config",
    version_base=None,
)
def main(config):
    model = CLIPModel(CLIPConfig())
    test_model(model)

def test_model(model):
    # 测试模型的前向传播
    batch = {
        "image": torch.randn(2, 3, 224, 224),  # 假设输入是一个批次的图像
        "input_ids": torch.randint(2, 1000, (2, 20)),  # 假设输入是一个批次的文本
        "attention_mask": torch.ones(2, 20)  # 假设输入是一个批次的注意力掩码
    }
    loss = model(batch)
    print("Loss:", loss.item())

if __name__ ==  "__main__":
    main()