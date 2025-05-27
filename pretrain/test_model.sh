export HYDRA_FULL_ERROR=1
python -m pretrain.my_decoder_only_model

#torchrun --nproc_per_node=NUM_GPUS your_script_name.py