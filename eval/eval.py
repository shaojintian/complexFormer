'''
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
'''
# 这是一个注释，说明该文件的灵感来源于 GitHub 上的 ML-GSAI/SMDM 项目代码。

import sys
# 导入 sys 模块，它提供了访问由 Python 解释器使用或维护的变量和与解释器强烈交互的函数。

from accelerate import Accelerator,DeepSpeedPlugin
# 导入 accelerate 库，这是一个由 Hugging Face 开发的库，用于简化 PyTorch 在不同硬件（如多GPU、TPU）上的分布式训练和推理。
import torch
# 导入 PyTorch 库，一个开源的机器学习框架。
import re
# 导入 re 模块，用于正则表达式操作，通常用于文本模式匹配。
from pathlib import Path
# 从 pathlib 模块导入 Path 类，用于以面向对象的方式处理文件系统路径。
import random
# 导入 random 模块，用于生成伪随机数。
import numpy as np
# 导入 NumPy 库，并使用别名 np，它是 Python 中科学计算的基础包，尤其擅长处理大型多维数组和矩阵。
import torch.nn.functional as F
# 从 PyTorch 的 torch.nn 模块导入 functional 子模块，并使用别名 F。它包含了许多神经网络相关的函数，如激活函数、损失函数等。
from datasets import Dataset
# 从 datasets 库导入 Dataset 类，这是 Hugging Face 的一个库，用于方便地加载和处理各种数据集。
from lm_eval.__main__ import cli_evaluate
# 从 lm_eval 库的 __main__ 模块导入 cli_evaluate 函数。lm_eval (Language Model Evaluation Harness) 是一个用于评估语言模型的框架。cli_evaluate 可能是其命令行评估工具的入口点。
from lm_eval.api.instance import Instance
# 从 lm_eval.api.instance 模块导入 Instance 类。这可能代表评估任务中的一个单独的评估样本或请求。
from lm_eval.api.model import LM
# 从 lm_eval.api.model 模块导入 LM 基类。自定义模型评估类需要继承这个基类。
from lm_eval.api.registry import register_model
# 从 lm_eval.api.registry 模块导入 register_model 装饰器。用于将自定义模型注册到 lm_eval 框架中，使其可以被框架发现和使用。
from tqdm import tqdm
# 从 tqdm 库导入 tqdm 函数，用于在循环中显示进度条，方便跟踪长时间运行的任务。
import hydra
# 导入 hydra 库，这是一个用于优雅地配置复杂应用程序的框架，通常用于管理机器学习实验的配置。
# 从当前项目（或同级目录）的 pretrain.py 文件中导入名为 generate 的函数。这个函数很可能与模型的文本生成逻辑有关。
from transformers import AutoTokenizer, AutoModel,AutoConfig
from pretrain import ComplexFormerModel,CustomConfig
from omegaconf import OmegaConf, DictConfig
import torch.optim as optim
# 从 transformers 库导入 AutoTokenizer 和 AutoModel 类。这是 Hugging Face 的核心库，提供了大量的预训练模型和分词器。
# AutoTokenizer: 根据模型名称或路径自动加载合适的分词器。
# AutoModel: 根据模型名称或路径自动加载合适的模型架构。

def set_seed(seed):
# 定义一个名为 set_seed 的函数，接收一个参数 seed。
    torch.manual_seed(seed)
    # 为 PyTorch 在 CPU 上的所有操作设置随机种子。
    random.seed(seed)
    # 为 Python 内置的 random 模块设置随机种子。
    np.random.seed(seed)
    # 为 NumPy 的随机数生成器设置随机种子。

    torch.backends.cudnn.deterministic = True
    # 设置 PyTorch 的 cuDNN 后端为确定性模式。如果为 True，cuDNN 将选择确定性的算法，这有助于实验的可复现性，但可能会牺牲一些性能。
    torch.backends.cudnn.benchmark = False
    # 禁用 PyTorch 的 cuDNN 基准测试模式。如果为 True，cuDNN 会在开始时运行一些基准测试来选择最快的卷积算法，但这可能导致不确定性。禁用它以确保可复现性。


# 使用 hydra 装饰器。
# config_path='../pretrain': 指定配置文件的相对路径。Hydra 会在 ../pretrain 目录下查找配置文件。
# config_name="config.yaml": 指定主配置文件的名称。Hydra 会加载 config.yaml。
# 这个装饰器会将函数的参数（这里是 ComplexFormerEvalHarness 的 __init__ 中的 config）替换为从配置文件加载的配置对象。
@register_model("complex_former_dist")
# 使用 lm_eval 的 register_model 装饰器。
# "complex_former_dist": 将这个类注册到 lm_eval 框架中，并命名为 "complex_former_dist"。这样在运行 lm_eval 时，可以通过这个名字来指定使用这个模型进行评估。
#@hydra.main(config_path='/mnt/afs/intern/fangwenhan/jintian/NightMare6B/pretrain', config_name="config.yaml")
class ComplexFormerEvalHarness(LM):
# 定义一个名为 ComplexFormerEvalHarness 的类，它继承自 lm_eval.api.model.LM 类。
# 这意味着此类需要实现 LM 基类定义的一些方法，以便 lm_eval 框架可以调用它们进行模型评估。
    def __init__(
        self,
        model_path='',
        # 模型路径，默认为空字符串。
        mask_id=126336,
        # [MASK] 标记的 token ID，默认为 126336。
        max_length=4096,
        # 模型能处理的最大序列长度，默认为 4096。
        batch_size=32,
        # 批处理大小，默认为 32。
        mc_num=128,
        # 蒙特卡洛估计的迭代次数，默认为 128。
        is_check_greedy=True,
        # 是否检查贪婪解码的标志，默认为 True。用于某些评估指标（如 LAMBADA）。
        # 文档中建议设为 False，因为 ComplexFormer 论文中评估的指标不需要此功能，设为 False 可以加速评估。
        cfg=0.,
        # Classifier-Free Guidance (CFG) 的缩放因子，默认为 0.0 (表示不使用CFG)。
        steps=1024,
        # 生成过程的步数，默认为 1024。（可能与 diffusion-like 模型相关）
        gen_length=1024,
        # 生成文本的长度，默认为 1024。
        block_length=1024,
        # 块长度，默认为 1024。（可能与模型的特定处理方式相关）
        remasking='low_confidence',
        # 重掩码策略，默认为 'low_confidence'。（可能与模型的迭代生成过程相关）
        device="cuda",
        # 指定模型运行的设备，默认为 "cuda" (GPU)。
        config_path="/mnt/afs/intern/fangwenhan/jintian/NightMare6B/pretrain/config.yaml",
        # Hydra 传递过来的配置对象，默认为 None。
        **kwargs,
        # 接收其他未明确定义的关键字参数。
    ):
        '''
        Args:
            model_path:  model path.
            mask_id: The token id of [MASK] is 126336.
            max_length: the max sequence length.
            batch_size: mini batch size.
            mc_num: Monte Carlo estimation iterations
            is_check_greedy: For certain metrics like LAMBADA, the evaluation requires the model to verify whether the answer
                             is generated through greedy sampling conditioned on the prompt (note that this differs from conditional
                             generation). We implement this verification through the suffix_greedy_prediction() function, which
                             returns a True/False judgment used for accuracy calculation.
                             When is_check_greedy is set to True, the lm-evaluation-harness library automatically invokes this function.
                             However, since none of the metrics in the ComplexFormer paper (https://arxiv.org/abs/2502.09992) require this functionality,
                             we recommend setting is_check_greedy to False. This configuration causes suffix_greedy_prediction() to return False
                             by default, significantly accelerating the evaluation process.
            cfg_scale: Unsupervised classifier-free guidance scale. (应为 cfg，与参数名 cfg 对应)
        '''
        # 这是类的初始化方法的文档字符串，解释了各个参数的含义。
        
        super().__init__()
        # 调用父类 (LM) 的构造函数。
        self.config = OmegaConf.load(config_path)
        if self.config is None:
            raise ValueError("no config")
        ds_plugin = DeepSpeedPlugin(
            # zero_stage=self.config.deepspeed.zero_stage,
            # gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
        )
        ds_plugin.hf_ds_config.config['train_micro_batch_size_per_gpu'] = 1

        # Initialize Accelerate with DeepSpeed plugin

        accelerator = Accelerator(
            mixed_precision=self.config.training.mixed_precision,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            deepspeed_plugin=ds_plugin,
        )
        # 创建一个 accelerate.Accelerator 对象，用于处理分布式环境。
        if accelerator.num_processes > 1:
        # 检查是否在多进程/多GPU环境下运行 (例如，通过 `accelerate launch` 启动)。
            self.accelerator = accelerator
            # 如果是，则将 accelerator 对象存储为实例属性。
        else:
            self.accelerator = None
            # 如果不是，则将 self.accelerator 设为 None。

        model_kwargs = {}
        # 初始化一个空字典，用于存储加载模型时可能需要的额外参数。
        if self.accelerator is not None:
        # 如果使用了 accelerate (即多GPU环境)。
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})
            # 更新 model_kwargs，设置 device_map。这告诉 Hugging Face Transformers 如何将模型分布到不同的设备上。
            # {'': f'{self.accelerator.device}'} 通常表示将整个模型加载到 accelerate 分配的当前进程的设备上。
        AutoConfig.register("ComplexFormer", CustomConfig)
        AutoModel.register(CustomConfig,ComplexFormerModel )
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, cache_dir=self.config.eval.checkpoint_path,local_files_only=False,**model_kwargs)
        # 使用 AutoModel 从指定路径加载预训练模型。
        # model_path: 模型的路径或 Hugging Face Hub 上的模型名称。
        # trust_remote_code=True: 允许加载模型时执行模型仓库中自定义的代码。需要谨慎使用。
        # torch_dtype=torch.bfloat16: 指定模型加载时使用的数据类型为 bfloat16，可以节省内存并可能加速计算，尤其在支持 bfloat16 的硬件上。
        # cache_dir=self.config.eval.checkpoint_path: 指定模型下载和缓存的目录，路径从 Hydra 配置中读取。
        # local_files_only=False: 允许从 Hugging Face Hub 下载模型文件，如果为 True，则只查找本地缓存。
        # **model_kwargs: 传递之前准备好的额外参数，如 device_map。
        self.model.eval()
        # 将模型设置为评估模式。这会关闭 dropout 等只在训练时使用的层。

        self.device = torch.device(device)
        # 根据传入的 device 参数 (默认为 "cuda") 创建一个 PyTorch 设备对象。
        if self.accelerator is not None:
        # 如果使用了 accelerate。
            optimizer = optim.AdamW(
                self.model.parameters(),  # 将您的模型参数传递给优化器
                lr=self.config.training.learning_rate,
                 # 可以添加其他的优化器参数
            )
            self.model = self.accelerator.prepare(self.model,optimizer)
            # 使用 accelerator.prepare 方法包装模型。这个方法会自动处理模型在分布式环境下的必要设置 (如 DDP 封装)。
            self.device = torch.device(f'{self.accelerator.device}')
            # 更新 self.device 为 accelerate 分配给当前进程的设备。
            self._rank = self.accelerator.local_process_index
            # 获取当前进程在节点内的本地排名 (rank)。
            self._world_size = self.accelerator.num_processes
            # 获取参与分布式运行的总进程数 (world size)。
        else:
            self.model = self.model.to(device)
            # 如果不使用 accelerate (单设备)，则直接将模型移动到指定的设备。

        self.mask_id = mask_id
        # 存储 MASK token 的 ID。
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name, trust_remote_code=True,cache_dir=self.config.tokenizer.cache,local_files_only=False)
        # 使用 AutoTokenizer 从指定路径加载分词器。
        # model_path: 分词器的路径或名称。
        # trust_remote_code=True: 允许加载分词器时执行仓库中的自定义代码。
        # cache_dir=self.config.tokenizer.cache: 指定分词器下载和缓存的目录，路径从 Hydra 配置中读取。
        # local_files_only=False: 允许从 Hub 下载。

        self.mc_num = mc_num
        # 存储蒙特卡洛估计的迭代次数。
        self.batch_size = int(batch_size)
        # 存储批处理大小，并确保是整数。
        assert mc_num % self.batch_size == 0
        # 断言：确保蒙特卡洛迭代次数能被批大小整除。这是为了方便后续的批处理计算。
        self.sampling_eps = 0.
        # 采样 epsilon，默认为 0.0 (可能用于某种采样策略，但这里未使用)。
        self.max_length = max_length
        # 存储最大序列长度。
        self.is_check_greedy = is_check_greedy
        # 存储是否检查贪婪解码的标志。

        self.cfg = cfg
        # 存储 CFG 缩放因子。
        self.steps = steps
        # 存储生成步数。
        self.gen_length = gen_length
        # 存储生成长度。
        self.block_length = block_length
        # 存储块长度。
        self.remasking = remasking
        # 存储重掩码策略。
    @property
    # 定义一个属性 (property)，使得 rank 方法可以像访问属性一样被调用 (即 `self.rank`)。
    def rank(self):
    # 定义 rank 方法。
        return self._rank
        # 返回当前进程的本地排名。

    @property
    # 定义一个属性。
    def world_size(self):
    # 定义 world_size 方法。
        return self._world_size
        # 返回分布式环境中的总进程数。

    def _forward_process(self, batch, prompt_index):
    # 定义一个名为 _forward_process 的内部方法，接收一个批次的 token ID (batch) 和一个指示哪些 token 属于提示 (prompt_index) 的布尔张量。
    # 这个方法似乎是 ComplexFormer 模型特有的某种前向扰动或掩码过程。
        b, l = batch.shape
        # 获取批次大小 (b) 和序列长度 (l)。

        target_len = (l - prompt_index.sum()).item()
        # 计算目标部分的长度。prompt_index.sum() 是提示部分的 token 数量。
        # .item() 将单元素张量转换为 Python 数字。
        k = torch.randint(1, target_len + 1, (), device=batch.device)
        # 在 [1, target_len] 范围内随机生成一个整数 k。这可能代表了在目标序列中要保留（不立即掩码）的 token 数量的某种基准。
        # () 表示生成一个标量。device=batch.device 确保 k 在与 batch 相同的设备上。

        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        # 这行代码比较复杂，目的是为批次中的每个样本生成一个不同的掩码数量 x_i。
        # torch.linspace(start, end, steps): 生成一个从 start 到 end 的等差序列，包含 steps 个点。
        # start = float(k)
        # end = k + (b - 1) * (target_len / b)
        # steps = b (批次大小)
        # 这似乎是想在批次的不同样本间平滑地分配掩码数量，使其均值接近 target_len / 2 (如果k接近target_len/2)，或者至少是围绕k变化。
        # torch.round(...).long(): 四舍五入并转换为长整型。
        x = ((x - 1) % target_len) + 1
        # 将 x 调整到 [1, target_len] 范围内。确保 x_i 不会是0或超过 target_len。
        assert x.min() >= 1 and x.max() <= target_len
        # 断言：确保 x 中的所有值都在有效范围内。

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        # 创建一个形状为 (b, target_len) 的索引张量，每行都是 [0, 1, ..., target_len-1]。
        is_mask = indices < x.unsqueeze(1)
        # 创建一个布尔掩码张量 is_mask，形状为 (b, target_len)。
        # x.unsqueeze(1) 将 x (形状为 [b]) 变为 [b, 1]，使其可以与 indices (形状为 [b, target_len]) 进行广播比较。
        # 对于批次中的第 i 个样本，其目标部分的前 x[i] 个 token（按原始顺序）将被标记为 True (潜在的掩码候选)。

        for i in range(b):
        # 遍历批次中的每个样本。
            is_mask[i] = is_mask[i][torch.randperm(target_len)]
            # 对每个样本的 is_mask 行进行随机打乱。
            # torch.randperm(target_len) 生成一个 0 到 target_len-1 的随机排列。
            # 这意味着，对于每个样本，我们随机选择 x[i] 个位置进行掩码，而不是固定选择前 x[i] 个。

        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)
        # 将 is_mask 扩展到整个序列长度。
        # torch.zeros(b, prompt_index.sum(), ...): 为提示部分创建一个全为 False 的掩码 (即提示部分不被掩码)。
        # torch.cat(..., dim=1): 在序列长度维度 (dim=1) 上拼接提示部分的掩码和目标部分的掩码。

        noisy_batch = torch.where(is_mask, self.mask_id, batch)
        # 根据 is_mask 创建 "有噪声的" (被掩码的) 批次。
        # 如果 is_mask 中对应位置为 True，则该 token 被替换为 self.mask_id。
        # 否则，保留原始 batch 中的 token。

        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)
        # 返回：
        # 1. noisy_batch: 经过掩码处理的批次。
        # 2. (x / target_len).unsqueeze(1).repeat(1, l): 一个张量，表示每个样本中目标部分被掩码的 token 的比例 (p_mask)。
        #    (x / target_len) 是每个样本的掩码比例 (形状 [b])。
        #    .unsqueeze(1) 变为 [b, 1]。
        #    .repeat(1, l) 扩展为 [b, l]，即每个时间步都用相同的掩码比例值。

    @torch.no_grad()
    # 装饰器，表示在此函数内的所有 PyTorch 操作都不会计算梯度。这在推理时非常重要，可以节省内存和计算。
    def get_logits(self, batch, prompt_index):
    # 定义 get_logits 方法，用于获取模型对输入批次的预测 logits。
    # batch: 输入的 token ID 批次。
    # prompt_index: 指示提示部分的布尔张量 (与 _forward_process 中的类似，但这里可能是一维的，然后广播)。
        if self.cfg > 0.:
        # 如果启用了 Classifier-Free Guidance (CFG)，即 self.cfg 大于 0。
            assert len(prompt_index) == batch.shape[1]
            # 断言：确保 prompt_index 的长度与输入序列的长度一致。
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            # 将一维的 prompt_index (形状 [l]) 扩展为 [1, l]，然后复制 batch_size 次，得到形状 [b, l]。
            un_batch = batch.clone()
            # 复制原始输入 batch，用于创建无条件 (unconditional) 输入。
            un_batch[prompt_index] = self.mask_id
            # 将 un_batch 中提示部分的所有 token 替换为 MASK ID。这是 CFG 中构造无条件输入的常见做法。
            batch = torch.cat([batch, un_batch])
            # 将原始 (有条件的) batch 和修改后的 (无条件的) un_batch 在批次维度上拼接起来。
            # 模型将一次性处理这两部分。

        logits = self.model(batch).logits
        # 将 (可能拼接后的) batch 输入模型，并获取输出的 logits。
        # .logits 是 Hugging Face Transformers 模型输出对象中通常包含预测得分的属性。

        if self.cfg > 0.:
        # 如果启用了 CFG。
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            # 将拼接后的 logits 分割回两部分：有条件的 logits 和无条件的 un_logits。
            # torch.chunk(tensor, chunks, dim): 沿指定维度将张量分割成指定数量的块。
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
            # 应用 CFG 公式: new_logits = unconditional_logits + guidance_scale * (conditional_logits - unconditional_logits)
            # 这里的 self.cfg 对应 (guidance_scale - 1)，所以 (self.cfg + 1) 对应 guidance_scale。
        return logits[:, :batch.shape[1]]
        # 返回 logits。如果使用了 CFG，这里 batch.shape[1] 应该是原始序列长度（在CFG处理前）。
        # [:, :batch.shape[1]] 确保返回的 logits 序列长度与原始输入一致 (尽管通常模型输出序列长度与输入相同)。

    @torch.no_grad()
    # 不计算梯度。
    def get_loglikelihood(self, prefix, target):
    # 定义 get_loglikelihood 方法，用于计算给定前缀 (prefix) 条件下，目标序列 (target) 的对数似然。
    # 这是 lm_eval 框架可能会调用的方法之一，用于评估困惑度等指标。
        seq = torch.concatenate([prefix, target])[None, :]
        # 将前缀和目标序列拼接起来，形成完整的序列。
        # [None, :] 或 .unsqueeze(0) 将其形状从 [seq_len] 变为 [1, seq_len]，即批大小为1。
        seq = seq.repeat((self.batch_size, 1)).to(self.device)
        # 将该序列复制 self.batch_size 次，形成一个批次。
        # .to(self.device) 将数据移动到指定的计算设备。

        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        # 创建一个布尔张量，标记出序列中属于前缀 (prefix) 的部分。长度为 seq.shape[1]。

        loss_acc = []
        # 初始化一个列表来累积损失。
        for _ in range(self.mc_num // self.batch_size):
        # 进行蒙特卡洛估计。循环次数为 mc_num (总迭代次数) 除以 batch_size (每次迭代处理的样本数)。
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)
            # 调用 _forward_process 方法，对当前批次的 seq 进行掩码扰动，并获取掩码比例 p_mask。
            # 注意：这里的 seq 是相同的序列复制了 batch_size 次。_forward_process 会对这个批次中的每个副本进行独立的随机掩码。

            mask_indices = perturbed_seq == self.mask_id
            # 找到被掩码的 token 的位置 (值为 self.mask_id 的位置)。

            logits = self.get_logits(perturbed_seq, prompt_index)
            # 获取模型对扰动序列 perturbed_seq 的预测 logits。

            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            # 计算交叉熵损失，但只在被掩码的位置计算。
            # logits[mask_indices]: 提取被掩码位置的预测 logits。其形状变为 [num_masked_tokens, vocab_size]。
            # seq[mask_indices]: 提取被掩码位置的真实 token ID (标签)。其形状变为 [num_masked_tokens]。
            # reduction='none': 使交叉熵损失为每个被掩码 token 返回一个损失值，而不是求和或平均。
            # / p_mask[mask_indices]: 用 p_mask (掩码比例) 对损失进行加权或归一化。这可能是 ComplexFormer 特有的损失计算方式。
            loss = loss.sum() / self.batch_size
            # 将所有被掩码位置的损失相加，然后除以批次大小，得到当前批次的平均损失。
            loss_acc.append(loss.item())
            # 将当前批次的平均损失 (转换为 Python float) 添加到 loss_acc 列表中。

        return - sum(loss_acc) / len(loss_acc)
        # 返回平均负对数似然。
        # sum(loss_acc) / len(loss_acc): 计算所有蒙特卡洛迭代的平均损失。
        # 取负号是因为对数似然通常越大越好，而损失是越小越好。

    @torch.no_grad()
    # 不计算梯度。
    def suffix_greedy_prediction(self, prefix, target):
    # 定义 suffix_greedy_prediction 方法。
    # 该方法用于检查模型是否能通过贪婪解码，从给定的前缀 (prefix) 生成出完全一致的目标 (target) 后缀。
    # 这在某些评估指标 (如 LAMBADA) 中需要。
        if not self.is_check_greedy:
        # 如果 self.is_check_greedy 为 False (如文档所建议用于 ComplexFormer)。
            return False
            # 直接返回 False，跳过这个检查，以加速评估。

        seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self.device)
        # 创建一个序列，长度为前缀和目标之和，初始时所有位置都填充为 MASK ID。批大小为1。
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        # 创建布尔张量，标记前缀部分。
        prefix, target = prefix.to(self.device), target.to(self.device)
        # 将前缀和目标张量移动到设备。
        seq[0, :len(prefix)] = prefix
        # 将序列的前缀部分替换为真实的 prefix token。

        for i in range(len(target)):
        # 迭代地生成目标序列的每个 token。
            mask_index = (seq == self.mask_id)
            # 找到当前序列中所有 MASK ID 的位置。
            logits = self.get_logits(seq, prompt_index)[mask_index]
            # 获取模型对当前序列的预测 logits，并只提取 MASK 位置的 logits。
            # seq 只有一个样本，所以 logits 形状是 [num_masks, vocab_size]。
            x0 = torch.argmax(logits, dim=-1)
            # 对每个 MASK 位置的 logits 进行 argmax，贪婪地选择最可能的 token。

            p = torch.softmax(logits.to(torch.float32), dim=-1)
            # 计算 MASK 位置 logits 的 softmax概率 (转换为 float32 以提高精度)。
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            # 获取贪婪选择的 token (x0) 对应的概率 (置信度)。
            _, index = torch.sort(confidence, descending=True)
            # 按置信度降序排序，获取排序后的索引。
            x0[index[1:]] = self.mask_id
            # 这一步很关键：只保留置信度最高的那个预测 token，将其他 MASK 位置的预测变回 MASK ID。
            # 这意味着每一步只确定一个 token，而不是一次性填充所有 MASK。这是迭代式解码的一种方式。
            seq[mask_index] = x0.clone()
            # 用新预测的 (可能部分填充的) x0 更新序列中的 MASK 位置。
        correct = target == seq[0, len(prefix):]
        # 比较生成的目标后缀 (seq[0, len(prefix):]) 与真实的 target 是否完全一致。
        correct = torch.all(correct)
        # 检查所有 token 是否都匹配。
        return correct.item() # .item() 将 bool 张量转为 Python bool
        # 返回 True 或 False。

    def _encode_pair(self, context, continuation):
    # 定义一个内部辅助方法 _encode_pair，用于将文本对 (context, continuation) 编码为 token ID。
    # 这个方法处理 lm_eval 中常见的 "loglikelihood" 任务格式。
        n_spaces = len(context) - len(context.rstrip())
        # 计算 context 字符串末尾的空格数量。
        if n_spaces > 0:
        # 如果 context 末尾有空格。
            continuation = context[-n_spaces:] + continuation
            # 将 context 末尾的空格移到 continuation 的开头。
            context = context[:-n_spaces]
            # 从 context 中移除末尾的空格。
            # 这样做是为了确保分词器能正确处理 context 和 continuation 之间的连接，特别是对于像 GPT-2 这样的分词器，它对空格敏感。

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        # 将拼接后的完整字符串 (context + continuation) 分词，并获取 input_ids。
        context_enc = self.tokenizer(context)["input_ids"]
        # 单独将 context 分词，并获取 input_ids。

        context_enc_len = len(context_enc)
        # 获取 context 部分编码后的 token 数量。
        continuation_enc = whole_enc[context_enc_len:]
        # 从完整编码序列中，切片出 continuation 部分的 token ID。
        # 这种做法比单独编码 continuation 更能保证与整体编码的一致性。

        return context_enc, continuation_enc
        # 返回编码后的 context token 列表和 continuation token 列表。

    def loglikelihood(self, requests):
    # lm_eval 框架要求实现的 loglikelihood 方法。
    # requests: 一个列表，每个元素是 lm_eval.api.instance.Instance 对象，包含 (context, continuation) 对。
        def _tokenize(e):
        # 定义一个内部的 _tokenize 函数，用于处理单个请求。
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            # 调用 _encode_pair 方法将 "prefix" (即 context) 和 "target" (即 continuation) 编码。
            return {
                "prefix_text": e["prefix"], # 原始文本
                "target_text": e["target"], # 原始文本
                "prefix": prefix,         # 编码后的 prefix token ID
                "target": target,         # 编码后的 target token ID
            }

        ds = [] # 初始化列表 ds
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        # 从 lm_eval 的 requests 对象中提取 prefix (context) 和 target (continuation)。
        # req.args 是一个元组，通常 (context, continuation)。
        ds = Dataset.from_list(ds)
        # 使用 Hugging Face datasets 库将列表转换为 Dataset 对象，方便后续处理。
        ds = ds.map(_tokenize)
        # 对 Dataset 中的每个元素应用 _tokenize 函数。
        ds = ds.with_format("torch")
        # 设置 Dataset 的格式为 "torch"，这样在访问元素时会自动转换为 PyTorch 张量。
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]
        # 计算每个样本编码后的总长度。

        assert max(prompt_len) <= 4096 # 应该用 self.max_length
        # 断言：确保所有样本的总长度不超过模型支持的最大长度 (这里硬编码为 4096，最好用 self.max_length)。

        out = []
        # 初始化一个列表来存储结果。
        with torch.no_grad():
        # 在不计算梯度的上下文中执行。
            for elem in tqdm(ds, desc="Computing likelihood..."):
            # 遍历 Dataset 中的每个元素，并使用 tqdm 显示进度条。
                prefix = elem["prefix"]
                # 获取编码后的前缀 (PyTorch 张量)。
                target = elem["target"]
                # 获取编码后的目标 (PyTorch 张量)。

                ll = self.get_loglikelihood(prefix, target)
                # 调用前面定义的 get_loglikelihood 方法计算对数似然。

                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)
                # 调用 suffix_greedy_prediction 方法检查贪婪解码一致性。

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
                # 将计算得到的对数似然 (ll) 和贪婪检查结果 (转换为 1.0 或 0.0) 作为元组添加到 out 列表中。
        torch.cuda.empty_cache()
        # 清空 PyTorch 的 CUDA 缓存，释放一些 GPU 显存。
        return out
        # 返回结果列表。

    def loglikelihood_rolling(self, requests):
    # lm_eval 框架可能调用的 loglikelihood_rolling 方法。用于计算滑动窗口对数似然，通常用于评估长文本的困惑度。
        raise NotImplementedError
        # 抛出 NotImplementedError 异常，表示这个方法尚未实现。

    def generate_until(self, requests: list[Instance]):
    # lm_eval 框架要求实现的 generate_until 方法。用于文本生成任务，直到遇到指定的停止条件。
    # requests: 一个列表，每个元素是 lm_eval.api.instance.Instance 对象。
    # Instance.args[0] 是输入文本 (prompt/question)。
    # Instance.args[1] 是一个字典，其中 Instance.args[1]['until'] 是停止标记列表。
        def _tokenize(e):
        # 内部 _tokenize 函数。
            return {
                "question": self.tokenizer(e["question"])["input_ids"],
                # 将问题文本分词并获取 input_ids。
                "question_text": e["question"],
                # 原始问题文本。
                "until": e["until"],
                # 停止标记列表。
            }

        ds = [{"question": req.args[0], "until": req.args[1]['until']} for req in requests]
        # 从 lm_eval 的 requests 对象中提取 "question" 和 "until" 条件。
        ds = Dataset.from_list(ds)
        # 转换为 Dataset 对象。
        ds = ds.map(_tokenize)
        # 应用 _tokenize 函数。
        ds = ds.with_format("torch")
        # 设置格式为 PyTorch 张量。

        out = []
        # 初始化结果列表。
        for elem in tqdm(ds, desc="Generating..."):
        # 遍历 Dataset，显示进度条。
            prompt = elem["question"].unsqueeze(0).to(self.device)
            # 获取编码后的问题 (prompt)，增加一个批次维度 (unsqueeze(0))，并移动到设备。
            stop_tokens = elem["until"]
            # 获取停止标记列表。

            # generate_autodiffusion 是一个外部函数，这里假设它来自 'pretrain.py'
            # 它负责实际的文本生成过程，可能使用了 ComplexFormer 特有的自回归扩散或类似机制。
            #print("="*30,type(self.model))
            generated_answer = self.model[0].generate(prompt)
            # 调用导入的 generate 函数 (这里代码中是 generate，注释中是 generate_autodiffusion，以代码为准)。
            # 传递模型、提示、以及初始化时设置的各种生成参数 (steps, gen_length, cfg 等)。
            # temperature=0 表示贪婪解码或接近贪婪解码。

            generated_answer = self.tokenizer.decode(generated_answer[0][prompt.shape[1]:], skip_special_tokens=False)
            # 解码生成的 token 序列。
            # generated_answer[0]: 取批次中的第一个 (也是唯一一个) 样本。
            # [prompt.shape[1]:]: 从提示之后的部分开始解码，即只解码生成的内容。
            # skip_special_tokens=False: 保留特殊 token (如 [EOS], [PAD])，因为停止条件可能包含它们。
            for stop_seq in stop_tokens:
            # 遍历所有停止标记。
                    if stop_seq in generated_answer:
                    # 如果生成的答案中包含某个停止标记。
                        generated_answer = generated_answer.split(stop_seq)[0]
                        # 以该停止标记为界分割字符串，并取第一部分，即停止标记之前的内容。
                        # 注意: 这只会处理第一个出现的停止标记。

            # remove special tokens
            generated_answer_ids = self.tokenizer(generated_answer)["input_ids"]
            # 再次对处理过的 (可能被截断的) generated_answer进行分词。
            generated_answer = self.tokenizer.decode(generated_answer_ids, skip_special_tokens=True)
            # 再次解码，但这次 skip_special_tokens=True，以移除所有特殊 token，得到干净的文本。
            out.append(generated_answer)
            # 将最终生成的文本添加到结果列表。

            if self.accelerator is not None: # 检查 accelerator 是否被初始化
                self.accelerator.wait_for_everyone()
                # 如果在分布式环境中，调用 accelerator.wait_for_everyone() 来同步所有进程。
                # 这确保在继续下一个样本或聚合结果之前，所有进程都已完成当前样本的生成。

        return out
        # 返回包含所有生成文本的列表。

if __name__ == "__main__":
# Python 的标准入口点，当脚本被直接执行时，这部分代码会运行。
    set_seed(1234)
    # 调用 set_seed 函数，设置随机种子为 1234，以确保实验的可复现性。
    cli_evaluate()
    # 调用从 lm_eval 导入的 cli_evaluate() 函数。
    # 这个函数会启动 lm_eval 的命令行评估流程。它会解析命令行参数 (lm_eval 通常通过命令行参数指定任务、模型、批大小等)，
    # 然后根据注册的模型 (如我们这里定义的 "complex_former_dist") 和指定的任务来执行评估。
    # Hydra 装饰器 @hydra.main 也会在这里发挥作用，它会处理配置文件，并将配置传递给 ComplexFormerEvalHarness 的构造函数。