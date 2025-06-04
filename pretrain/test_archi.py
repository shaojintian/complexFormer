from transformers import AutoModel
model = AutoModel.from_pretrained("shaojintian/complex_attention_0.5B", trust_remote_code=True)
print(model.__class__)