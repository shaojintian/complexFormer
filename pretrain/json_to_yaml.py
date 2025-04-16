import yaml, json

with open("./pretrain/config.yaml", "r") as yf:
    data = yaml.safe_load(yf)
with open("./pretrain/config.json", "w") as jf:
    json.dump(data, jf, indent=2, ensure_ascii=False) 