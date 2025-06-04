#!/bin/bash



export TORCH_DISTRIBUTED_DEBUG=INFO
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5_0,mlx5_1
export NCCL_DEBUG_FILE=./logs/nccl.log
export NCCL_PXN_DISABLE=1
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY=a61267657f3dbe2e3e359f2ece24f0812b53328e
#accelerate launch --config_file ./pretrain/acc_config.yaml --deepspeed_config_file ./pretrain/zero3.json pretrain/train_eval.py 
accelerate launch --config_file ./pretrain/acc_config_acp.yaml pretrain/train_eval.py 

#ps aux | grep "NightMare6B/venv/bin/python" | grep -v grep | awk '{print $2}' | xargs kill -9