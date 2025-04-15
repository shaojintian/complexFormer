import yaml

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_gpu_memory():
    import torch
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() 


from torch.optim import Optimizer
import math

class WarmupDecayScheduler:
    """预热 + 衰减调度器（支持线性/余弦衰减）"""
    def __init__(self, 
        optimizer: Optimizer, 
        warmup_steps: int, 
        total_steps: int, 
        base_lr: float,
        decay_type: str = 'cosine'  # 'cosine' 或 'linear'
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.decay_type = decay_type
        self.current_step = 0
        
        # 初始学习率设为0
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = 0.0
    
    def step(self):
        self.current_step += 1
        # 预热阶段
        if self.current_step <= self.warmup_steps:
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        # 衰减阶段
        else:
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            if self.decay_type == 'cosine':
                lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))
            elif self.decay_type == 'linear':
                lr = self.base_lr * (1.0 - progress)
            else:
                raise ValueError(f"Unsupported decay type: {self.decay_type}")
        
        # 更新所有参数组的学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr