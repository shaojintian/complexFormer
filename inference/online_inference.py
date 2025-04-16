import torch



# 加载模型和分词器
model = torch.load("./pretrain/model.pt")
tokenizer = torch.load("./pretrain/tokenizer.pt")


input_text = "The future of AI is"
inputs = tokenizer(input_text, return_tensors="pt")
input_ids = inputs.input_ids
past_key_values = None  # 初始化缓存

for _ in range(50):
    outputs = model(
        input_ids=input_ids[:, -1:] if past_key_values else input_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    next_token_logits = outputs.logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1)
    input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
    past_key_values = outputs.past_key_values  # 更新缓存
    print(tokenizer.decode(next_token), end="", flush=True)