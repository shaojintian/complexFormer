#!/bin/bash
#SBATCH -J train_owt_ar                # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -e watch_folder/%x_%j.err     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=32G                  # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu          # Request partition
#SBATCH --constraint="[a5000|a6000|3090|a100]"
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption


export TORCH_DISTRIBUTED_DEBUG=INFO
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=INFO
# export NCCL_IB_HCA=mlx5_0,mlx5_1
# export NCCL_DEBUG_FILE=./logs/nccl.log
# export NCCL_PXN_DISABLE=1
#accelerate launch --config_file ./pretrain/acc_config.yaml --deepspeed_config_file ./pretrain/zero3.json pretrain/train_eval.py 
accelerate launch --config_file ./pretrain/acc_config.yaml pretrain/train_eval.py 

#ps aux | grep "NightMare6B/venv/bin/python" | grep -v grep | awk '{print $2}' | xargs kill -9