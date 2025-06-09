import torch
import torch.nn as nn
from model import Qwen3ForCausalLM
from safetensors.torch import load_file
from transformers import AutoTokenizer,AutoConfig
from attention import ComplexMultiHeadAttentionV2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载权重函数，支持忽略不匹配层
def load_safetensors_weights(model, path):
    state_dict = load_file(path)
    model_keys = set(model.state_dict().keys())
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
    print("未加载的参数:", missing_keys)
    print("多余的参数:", unexpected_keys)




# 模拟文本生成函数（示例）
def generate_text(model, prompt):
    for _, layer in enumerate(model.model.layers):
        layer.self_attn = ComplexMultiHeadAttentionV2(d_model=model.config.hidden_size, num_heads=model.config.num_attention_heads)
    model.to(device)
    model.eval()
    with torch.no_grad():
        tokenizer = AutoTokenizer.from_pretrained("./qwen/", trust_remote_code=True,local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
        )
        inputs = tokenizer([text], return_tensors='pt').to(device)
        generate_ids = model.generate(**inputs, max_new_tokens=32768)
        text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return text

# 主流程示范
if __name__ == "__main__":
    config = AutoConfig.from_pretrained("./qwen/config.json", trust_remote_code=True, local_files_only=True)
    model = Qwen3ForCausalLM(config)
    load_safetensors_weights(model, "./qwen/model.safetensors")

    # 假设输入是(batch_size=1, feature_dim=768)的随机张量
    dummy_input = "Please tell me some information about China"
    text = generate_text(model, dummy_input)
    print("生成的text:", text)


    