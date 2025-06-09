# pip install transformers==4.49.0 lm_eval==0.4.8 accelerate==0.34.2
#pip install antlr4-python3-runtime==4.11 math_verify sympy hf_xet


# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_ENDPOINT=https://hf-mirror.com
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY=a61267657f3dbe2e3e359f2ece24f0812b53328e
# conditional likelihood estimation benchmarks
#accelerate launch --config_file postrain/acc_config.yaml eval/eval.py --tasks gpqa_main_n_shot --num_fewshot 5 --model complex_former_dist --batch_size 8 --model_args model_path='./model',cfg=0.5,is_check_greedy=False,mc_num=128  

#accelerate launch --config_file postrain/acc_config.yaml eval/eval.py --tasks truthfulqa_mc2 --num_fewshot 0 --model complex_former_dist --batch_size 8 --model_args model_path='./model',cfg=2.0,is_check_greedy=False,mc_num=128  

#accelerate launch --config_file postrain/acc_config.yaml eval/eval.py --tasks arc_challenge --num_fewshot 0 --model complex_former_dist --batch_size 8 --model_args model_path='./model',cfg=0.5,is_check_greedy=False,mc_num=128  

#accelerate launch --config_file postrain/acc_config.yaml eval/eval.py --tasks hellaswag --num_fewshot 0 --model complex_former_dist --batch_size 8 --model_args model_path='./model',cfg=0.5,is_check_greedy=False,mc_num=128  

#accelerate launch --config_file postrain/acc_config.yaml eval/eval.py --tasks winogrande --num_fewshot 5 --model complex_former_dist --batch_size 8 --model_args model_path='./model',cfg=0.0,is_check_greedy=False,mc_num=128  

#accelerate launch --config_file postrain/acc_config.yaml eval/eval.py --tasks piqa --num_fewshot 0 --model complex_former_dist --batch_size 8 --model_args model_path='./model',cfg=0.5,is_check_greedy=False,mc_num=128  

#accelerate launch --config_file postrain/acc_config.yaml eval/eval.py --tasks mmlu --num_fewshot 5 --model complex_former_dist --batch_size 1 --model_args model_path='./model',cfg=0.0,is_check_greedy=False,mc_num=1  


#accelerate launch --config_file postrain/acc_config.yaml eval/eval.py --tasks mmlu_pro --num_fewshot 5 --model complex_former_dist --batch_size 1 --model_args model_path='./model',cfg=0.0,is_check_greedy=False,mc_num=1  


#accelerate launch --config_file postrain/acc_config.yaml eval/eval.py --tasks cmmlu --num_fewshot 5 --model complex_former_dist --batch_size 1 --model_args model_path='./model',cfg=0.0,is_check_greedy=False,mc_num=1  

#accelerate launch --config_file postrain/acc_config.yaml eval/eval.py --tasks ceval-valid --num_fewshot 5 --model complex_former_dist --batch_size 1 --model_args model_path='./model',cfg=0.0,is_check_greedy=False,mc_num=1  


# conditional generation benchmarks
#accelerate launch --config_file postrain/acc_config.yaml eval/eval.py --tasks bbh --model complex_former_dist --model_args model_path='./model',gen_length=512,steps=512,block_length=512  

#accelerate launch --config_file postrain/acc_config.yaml eval/eval.py --tasks gsm8k --model complex_former_dist --model_args model_path='./model',gen_length=512,steps=512,block_length=512  

#accelerate launch --config_file postrain/acc_config.yaml eval/eval.py --tasks minerva_math --model complex_former_dist --model_args model_path='./model',gen_length=512,steps=512,block_length=512  

#accelerate launch --config_file postrain/acc_config.yaml eval/eval.py --tasks humaneval --model complex_former_dist --confirm_run_unsafe_code --model_args model_path='./model',gen_length=512,steps=512,block_length=512  

#accelerate launch --config_file postrain/acc_config.yaml eval/eval.py --tasks mbpp --model complex_former_dist --confirm_run_unsafe_code --model_args model_path='./model',gen_length=512,steps=512,block_length=512  

accelerate launch --config_file postrain/acc_config.yaml \
    -m lm_eval --model hf \
    --tasks humaneval \
    --model_args pretrained=shaojintian/complex_attention_0.5B,parallelize=True,trust_remote_code=True \
    --batch_size 16