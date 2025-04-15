import torch
from torch.utils.tensorboard import SummaryWriter
from utils.utils import get_gpu_memory,WarmupDecayScheduler  # 工具函数定义见下文
from my_decoder_only_model import MyDecoderOnlyModel as BigModel

# 初始化 TensorBoard Writer (日志保存到 runs/ 目录)
writer = SummaryWriter(log_dir='runs/exp1')

def train():
    model = BigModel().half().cuda()
    optimizer = torch.optim.AdamW(model.parameters())
    # 总训练步数（假设每个 epoch 有 1000 步）
    total_steps = 1000
    warmup_steps = 200  # 前 20% 步数预热
    
    # 初始化调度器（选择 'cosine' 或 'linear'）
    scheduler = WarmupDecayScheduler(
        optimizer, 
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        base_lr=1e-4,
        decay_type='cosine'
    )
    
    input = 
    target = 

    # 训练循环
    for step in range(total_steps):  # 假设训练 10 个 epoch
        
        #todo input左移一位即target
        
        output = model(input)
        loss = torch.nn.CrossEntropyLoss()(output.view(-1, output.size(-1)), target.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        writer.add_scalar('Loss/train', loss.item(), step)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], step)
        writer.add_scalar('GPU memory', get_gpu_memory(), step)


if __name__ == "__main__":
    train()
    writer.close()  # 关闭写入器