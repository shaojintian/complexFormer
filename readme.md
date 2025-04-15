单卡pretrain 24GB

预训练
1.  数据集dataset 
from modelscope.msdatasets import MsDataset

2. tokenizer 
sentencepiece


3.  pretrain模型my_decoder_only_model
decoder-only 
8x transformer block 
    mla+flash att + resnet
    rmsnorm
    ffn(swiGLU)
    rmsnorm
linear
softmax

4.参数估计显存占有
weight: fp32(4bytes)
gradient: eq = weight
opetimizer(momentum+variance): eq = weight x2
cuda kernel:1.3GB

sum = weight + gradient + optimizer + cuda kernel = 

5.pretraining
    loss: cross entropy 
    optimizer: adamw
    scheduler: cosine
    lr: 1e-4
    batch size: 8
    epoch: one epoch technique
    warmup: 0.1
    mixed-priceision train

6. posttraing
RLHF(DPO)
    reward-model

7. evaluation
MMLU HELM
