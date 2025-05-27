from transformers import AutoTokenizer, AutoModelForCausalLM # 或者其他 AutoModel 类型
import yaml
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM # Or other AutoModel types
import torch # For torch_dtype if specified

# --- Configuration Loading ---
def load_config(config_path="config.yaml"):
    """Loads configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    return config_data

config = load_config("../pretrain/config.yaml")  # Adjust path as needed

# 1. 加载预训练的分词器 (和模型，如果需要的话)
tokenizer_name = config.tokenizer_name # 或者 "bert-base-uncased", "meta-llama/Llama-2-7b-hf", 等
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,cache_dir="./tokenizer/cache", trust_remote_code=True)

# 如果你的分词器没有 pad_token，而你后续的任务可能需要，最好现在加上
if tokenizer.pad_token is None:
    print("Tokenizer does not have a pad_token. Adding one based on eos_token.")
    # 常见做法是将 pad_token 设置为 eos_token
    # 或者添加一个新的 '[PAD]' token: tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token


# 2. 定义并添加新的特殊 token
new_special_tokens = ["<think>", "</think>"] # 你可以添加一个或多个
# 使用 add_special_tokens。它接受一个字典，键是 'additional_special_tokens'，值是包含新 token 字符串的列表。
num_added_toks = tokenizer.add_special_tokens({'additional_special_tokens': new_special_tokens})

print(f"原始词汇表示例 (eos_token): {tokenizer.eos_token} -> ID: {tokenizer.eos_token_id}")
print(f"添加了 {num_added_toks} 个新的特殊 token(s).")

# 验证新 token 是否已添加
for token_str in new_special_tokens:
    token_id = tokenizer.convert_tokens_to_ids(token_str)
    print(f"新 Token: '{token_str}' -> ID: {token_id}")

print(f"分词器词汇表大小现在为: {len(tokenizer)}") # 或者 tokenizer.vocab_size

# --- 如果你需要将此分词器与模型一起使用 ---
# 3. 调整模型词嵌入层的大小
# 只有在你加载了模型并且打算用这个修改过的分词器来训练或使用它时，才需要这一步。
try:
    # 假设你使用的是一个用于生成任务的模型
    model = AutoModelForCausalLM.from_pretrained("./model",
        torch_dtype="auto",  # 或者 torch.bfloat16, torch.float16
        device_map="auto",   # 自动使用 GPU 如果可用
        trust_remote_code=True
    )
    print(f"模型原始词嵌入层大小: {model.get_input_embeddings().weight.size(0)}")

    model.resize_token_embeddings(len(tokenizer))
    # 新添加的 token embeddings 通常会被随机初始化，或者基于现有 embeddings 的平均值等。
    # 在微调过程中，这些新的 embeddings 会被学习。

    print(f"模型调整后词嵌入层大小: {model.get_input_embeddings().weight.size(0)}")
    assert model.get_input_embeddings().weight.size(0) == len(tokenizer) # 验证大小是否一致
except Exception as e:
    print(f"未加载或调整模型: {e}")
    print("如果你只是想修改分词器用于数据预处理，则不需要加载和调整模型。")
    model = None # 明确表示模型未处理
# --- 模型调整结束 ---


# 4. 测试分词效果
text_with_new_tokens = "Let's think step by step. <think>This is my thought process.</think> So the answer is 42."
encoded_input = tokenizer.encode(text_with_new_tokens)
decoded_output = tokenizer.decode(encoded_input)

print(f"\n原始文本: {text_with_new_tokens}")
print(f"编码后的 IDs: {encoded_input}")
print(f"解码后的文本: {decoded_output}")

# 检查新 token 是否被正确编码为一个单独的 ID
think_start_id = tokenizer.convert_tokens_to_ids("<think>")
think_end_id = tokenizer.convert_tokens_to_ids("</think>")

if think_start_id in encoded_input and think_end_id in encoded_input:
    print("新 token <think> 和 </think> 被正确编码为单个 ID。")
else:
    print("警告: 新 token 未按预期编码！")

# 5. 保存修改后的分词器 (和模型，如果已修改)
save_directory = config.tokenizer_cache # 例如 "path/to/save/directory"
tokenizer.save_pretrained(save_directory)
print(f"\n修改后的分词器已保存到: {save_directory}")

if model:
    model.save_pretrained(save_directory)
    print(f"修改后的模型已保存到: {save_directory}")

# 后续可以这样加载:
# tokenizer_loaded = AutoTokenizer.from_pretrained(save_directory)
# if model:
#     model_loaded = AutoModelForCausalLM.from_pretrained(save_directory)