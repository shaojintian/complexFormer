import os
import sys
# 获取当前文件 (train.py) 的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取当前目录的父目录 (NightMare6B/)
parent_dir = os.path.dirname(current_dir)
# 将父目录添加到 Python 的搜索路径中
sys.path.append(parent_dir)
import torch
from itertools import chain
from datasets import load_dataset
import hydra
from typing import Optional, List, Dict
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    AutoModel,
    PreTrainedModel,
    AutoModelForCausalLM
)
import torch.nn.functional as F
from transformers import Trainer, DataCollatorForLanguageModeling, TrainerCallback
import math
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer import TRAINER_STATE_NAME
from rich import traceback
from typing import Dict, Tuple, Union
from pretrain import ComplexFormerModel, CustomConfig
import logging
import wandb
from omegaconf import OmegaConf
from transformers.integrations import WandbCallback
from pretrain import test_model
from torch import Tensor
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import DistributedType
import numpy as np
from datasets import DatasetDict,load_from_disk
from dataset import count_token

logger: logging.Logger = logging.getLogger(__name__)
logger.propagate = False
logger.addHandler(logging.FileHandler("./logs/train_eval.log"))
traceback.install()

os.environ["WANDB_LOG_MODEL"] = "false"

@hydra.main(
    config_path=".",
    config_name="config",
    version_base=None,
)
def main(config):
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, trust_remote_code=True, cache_dir=config.tokenizer_cache, local_files_only=True)
    AutoConfig.register("autodiffusion", CustomConfig)
    AutoModel.register(CustomConfig,ComplexFormerModel )

    # Load the custom model
    customConfig = AutoConfig.from_pretrained(config.model.save_dir, trust_remote_code=True)
    
# Initialize DeepSpeed plugin for pipeline and tensor parallelism
    ds_plugin = DeepSpeedPlugin(
        zero_stage=config.deepspeed.zero_stage,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
    )

    # Initialize Accelerate with DeepSpeed plugin
    accelerator = Accelerator(
        mixed_precision=config.training.mixed_precision,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        deepspeed_plugin=ds_plugin,
    )

    if config.mode == "train":
        
        model = AutoModel.from_pretrained(
            config.model.custom_model_load_path,
            config=customConfig,
            trust_remote_code=True,
            local_files_only=True,
        )
        # Initialize W&B
        if accelerator.is_main_process:
            wandb.init(
                project=config.wandb.project,
                name=config.wandb.name,
                config=OmegaConf.to_container(config, resolve=True),
            )

        model = model.to(accelerator.device)

        

        class SelfTrainer(Trainer):
            def __init__(self, *args, debug: bool = False, **kwargs):
                super().__init__(*args, **kwargs)
                self.debug = debug
                self.avg_loss = 0.0
                self._togger_loger()

            def compute_loss(self, model: PreTrainedModel, inputs: Dict[str, Tensor], num_items_in_batch: Optional[int] = None, return_outputs: bool = False):
                outputs = model(**inputs)
                logits = outputs
                labels = inputs.get("labels")
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)).float(), shift_labels.view(-1))
                self.avg_loss = loss.item()
                return (loss, outputs) if return_outputs else loss

            def compute_metrics(self, eval_pred):
                logits, labels = eval_pred
                preds = np.argmax(logits, axis=-1)
                # Assuming you have clf_metrics defined somewhere
                # clf_results = clf_metrics.compute(predictions=preds, references=labels)
                results = {"perplexity": np.exp(self.avg_loss)}
                # results.update(clf_results)
                return results

            def _togger_loger(self):
                if not self.debug:
                    logger.setLevel(logging.WARNING)

            # def _save_checkpoint(self, model, trial, metrics=None):
            #     output_dir = os.path.join(self.args.output_dir, f"checkpoint-{self.state.global_step}")
            #     accelerator.save_state(output_dir)
            #     if accelerator.is_main_process:
            #         self.tokenizer.save_pretrained(output_dir)
            #         self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))
            #         logger.warning(f"ckpt  saved to {output_dir}")

        # Load Dataset
        logger.info(f"Loading preprocessed dataset from {config.data.output_dir}...")
        preprocessed_dataset: DatasetDict = load_from_disk(config.data.output_dir)
        logger.info(f"Loaded dataset with {len(preprocessed_dataset)} {preprocessed_dataset.keys()} examples.")
        split_dataset = preprocessed_dataset['train'].train_test_split(test_size=0.1, seed=42)
        train_dataset = split_dataset["train"]
        #count_token(config,train_dataset)
        eval_dataset = split_dataset["test"]
        logger.info(f"Training dataset size: {len(train_dataset)}")
        logger.info(f"Validation dataset size: {len(eval_dataset)}")

        class TruncatingDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
            def __init__(self, tokenizer, max_length, **kwargs):
                super().__init__(tokenizer, **kwargs)
                self.max_length = max_length

            def __call__(self, examples):
                for example in examples:
                    example["input_ids"] = example["input_ids"][:self.max_length]
                    if "attention_mask" in example:
                        example["attention_mask"] = example["attention_mask"][:self.max_length]
                return super().__call__(examples)

        collator = TruncatingDataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            max_length=tokenizer.model_max_length,
            mlm=False
        )

        class EarlyStoppingCallback(TrainerCallback):
            def __init__(self, early_stopping_patience=3, min_delta=0.0):
                self.patience = early_stopping_patience
                self.min_delta = min_delta
                self.best_metric = None
                self.no_improvement_counter = 0

            def on_evaluate(self, args, state, control, **kwargs):
                current_metric = state.log_history[-1]['eval_loss']
                if self.best_metric is None or current_metric < (self.best_metric - self.min_delta):
                    self.best_metric = current_metric
                    self.no_improvement_counter = 0
                else:
                    self.no_improvement_counter += 1
                    if self.no_improvement_counter >= self.patience:
                        control.should_training_stop = True

        training_args = TrainingArguments(
            output_dir=config.training.output_path,
            overwrite_output_dir=True,
            learning_rate=config.training.learning_rate,
            warmup_ratio=config.training.warmup_ratio,
            lr_scheduler_type=config.training.lr_scheduler_type,
            num_train_epochs=config.training.epochs,
            per_device_train_batch_size=config.training.batch_size,
            per_device_eval_batch_size=config.training.batch_size,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            gradient_checkpointing=config.training.gradient_checkpointing,
            save_strategy="steps",
            save_steps=config.training.save_steps,
            save_total_limit=config.training.save_total_limit,
            bf16=config.training.bf16,
            fp16=config.training.fp16,
            logging_strategy="steps",
            logging_steps=config.training.log_step,
            eval_strategy="steps",
            eval_steps=config.training.log_step,
            report_to="wandb" if accelerator.is_main_process else None,
            remove_unused_columns=False,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            run_name=config.wandb.name,
            max_grad_norm=config.training.max_grad_norm,
        )

        callbacks_list = [
            EarlyStoppingCallback(early_stopping_patience=config.training.early_stopping_patience),
        ]
        if accelerator.is_main_process:
            callbacks_list.append(WandbCallback())
        trainer = SelfTrainer(
            model=model,
            args=training_args,
            data_collator=collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            debug=config.training.debug,
            callbacks=callbacks_list
        )

        last_checkpoint = None
        if os.path.isdir(training_args.output_dir) and accelerator.is_main_process:
            last_checkpoint = get_last_checkpoint(config.checkpointing.resume_ckpt_path)
            if last_checkpoint:
                logger.warning(f"Checkpoint detected, resuming training from: {last_checkpoint}")
            elif any(f.startswith("checkpoint-") for f in os.listdir(training_args.output_dir)):
                logger.warning(f"Checkpoint folders found in {training_args.output_dir}, but get_last_checkpoint failed. Specify path manually if needed.")
            else:
                logger.warning (f"No checkpoint found in {training_args.output_dir}. Starting training from scratch.")

        logger.info("Starting training with Accelerate and DeepSpeed...")
        try:
            train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
            logger.info("Training finished.")
            logger.info(f"Train Results: {train_result.metrics}")

            trainer.log_metrics("train", train_result.metrics)
            trainer.save_metrics("train", train_result.metrics)

            eval_metrics = trainer.evaluate()
            logger.info(f"Final evaluation metrics: {eval_metrics}")
            trainer.save_metrics("eval", eval_metrics)

            if accelerator.is_main_process:
                logger.info("Saving final model...")
                final_save_path = os.path.join(training_args.output_dir, "final_model")
                accelerator.save_state(final_save_path)
                tokenizer.save_pretrained(final_save_path)
                logger.info(f"Final model and tokenizer saved to {final_save_path}")

        except Exception as e:
            logger.error(f"An error occurred during training: {e}")
            import traceback
            traceback.print_exc()
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                logger.error("Saving accelerator state due to error...==========")
                trainer._save_checkpoint(model, trial=None)
                logger.error("Accelerator state saved due to error...==========")
        finally:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                logger.error("Attempting to save accelerator state due to error...")
                logger.error("Saving accelerator state due to error...==========")
                trainer._save_checkpoint(model, trial=None)
                logger.error("Accelerator state saved due to error...==========")


        if accelerator.is_main_process:
            wandb.finish()

    elif config.mode == "sample":
        device = ('cuda' if torch.cuda.is_available() else 'cpu')
        @torch.no_grad()
        #TODO:
        def _inference(config):
            model = AutoModel.from_pretrained(
                config.training.final_path,
                config=customConfig,
                trust_remote_code=True,
                local_files_only=True,
            ).eval().to(device)
            tokenizer.padding_side = "left" # Important for generation

            model.to(torch.bfloat16)

            input_text = "Please tell me some information about China \n"
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=config.max_seq_len).to(device)
            #tokenizer.apply_chat_template(inputs)
            input_ids: Tensor = inputs.input_ids
            attention_mask: Tensor = inputs.attention_mask
            logger.warning(f"attention IDs: {attention_mask.shape}")
            print(input_text, end="", flush=True)
            #genereate
            generated_inputids = model.generate(input_ids,attention_mask,max_length = config.max_seq_len)
            print("Answer is", tokenizer.batch_decode(generated_inputids,skip_special_tokens=True))
            # output = model(**inputs)
            # logits = output
            
            # for _ in range(config.max_seq_len - input_ids.shape[1]):
            #     outputs = model(input_ids, attention_mask=attention_mask)
            #     logits = outputs.logits if hasattr(outputs, 'logits') else outputs # Handle different model outputs
            #     next_token_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)

            #     # Gather next token IDs from all processes
            #     #gathered_next_token_ids = accelerator.gather(next_token_id)

            #     # Only the main process appends the token and prints
                
            #     input_ids = torch.cat([input_ids, next_token_id], dim=1)
            #     logger.warning(input_ids)
            #     next_token = tokenizer.decode(next_token_id[0], skip_special_tokens=True)
            #     print(next_token, end=" ", flush=True)
                
            #     attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)], dim=-1)
                #print(inputs_ids, end=" ", flush=True)
                #output = model(input_ids,attention_mask=attention_mask)
        _inference(config)
    elif config.mode == "ppl_eval":
        pass

if __name__ == "__main__":
    main()