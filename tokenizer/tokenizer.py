import torch
import torch.nn as nn
import sentencepiece as spm
from typing import List

class TokenEmbedder:
    def __init__(self, model_path: str, embed_dim: int = 256):
        """
        初始化分词器与嵌入层
        :param model_path: SentencePiece 模型路径 (如 'tokenizer/model.model')
        :param embed_dim: Embedding 维度
        """
        # 加载分词器
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        
        # 创建嵌入层
        self.vocab_size = self.sp.get_piece_size()
        self.embedding = nn.Embedding(self.vocab_size, embed_dim)
        
    def encode_text(self, text: str) -> torch.Tensor:
        """
        生成文本的 Embedding 向量
        :param text: 输入文本 (如 "这是一个测试")
        :return: (seq_len, embed_dim) 的 Tensor
        """
        # 分词并转换为 ID
        tokens = self.sp.encode_as_ids(text)
        # 转换为 Tensor
        token_ids = torch.LongTensor(tokens)
        # 生成 Embeddings
        embeddings = self.embedding(token_ids)
        return embeddings
    
    def encode_as_ids(self, text: str) -> List[int]:
        """
        将文本转换为 ID 列表
        :param text: 输入文本 (如 "这是一个测试")
        :return: ID 列表
        """
        return self.sp.encode_as_ids(text)

# 示例用法
if __name__ == "__main__":
    embedder = TokenEmbedder(model_path="./tokenizer/gogpt_60k.model", embed_dim=512)
    embeddings = embedder.encode_text("这是一个测试"),
    input_ids = embedder.encode_as_ids("这是一个测试")
    print(f"Input IDs: {input_ids}")  # 输出: [1, 2, 3, 4]
    print(f"Embeddings shape: {embeddings.shape}")  # 输出: torch.Size([7, 256])