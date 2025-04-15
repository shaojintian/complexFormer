from sentencepiece import SentencePieceProcessor
import sentencepiece as spm

# 加载训练好的模型
sp = spm.SentencePieceProcessor()
sp.load('./tokenizer/gogpt_60k.model')

# 分词示例
text = "这是一个测试句子This is a test sentence."
tokens = sp.encode_as_pieces(text)
ids = sp.encode_as_ids(text)

print("Tokens:", tokens)
# 输出: ['▁这是', '一个', '测试', '句子', '<en>', 'This', '▁is', '▁a', '▁test', '▁sentence', '.']

print("IDs:", ids)

print("Vocabulary Size:", sp.get_piece_size())