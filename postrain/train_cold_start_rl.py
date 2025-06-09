import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch
import torch.nn.functional as F
from itertools import chain
from datasets import load_dataset, DatasetDict, load_from_disk
import hydra
from typing import Optional, List, Dict, Union, Tuple
from transformers import (
    AutoConfig,
    AutoTokenizer,
    TrainingArguments,
    PreTrainedModel,
    Trainer,
    TrainerCallback
)
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase
from dataclasses import dataclass # For DataCollator
import math
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer import TRAINER_STATE_NAME
from rich import traceback
import logging
import wandb
from omegaconf import OmegaConf
from transformers.integrations import WandbCallback
# from pretrain import ComplexFormerModel, CustomConfig # Assuming these are your base classes
# For Reward Model, we'll define/import it
# from your_model_file import ComplexFormerForRewardModeling, CustomConfig # Placeholder
from accelerate import Accelerator, DeepSpeedPlugin, DataLoaderConfiguration
from accelerate.utils import DistributedType
import numpy as np

# --- Mocking ComplexFormerForRewardModeling and CustomConfig for standalone run ---
# Replace these with your actual imports
if 'ComplexFormerForRewardModeling' not in globals():
    class CustomConfig(AutoConfig): # Make sure it inherits from AutoConfig
        model_type = "complex_former_rm" # Needs a unique model_type
        def __init__(self, hidden_size=768, initializer_range=0.02, **kwargs):
            super().__init__(**kwargs)
            self.hidden_size = hidden_size
            self.initializer_range = initializer_range
            # Add other necessary config parameters for ComplexFormerModel
            self.num_attention_heads = kwargs.get("num_attention_heads", 12)
            self.num_hidden_layers = kwargs.get("num_hidden_layers", 12)
            self.intermediate_size = kwargs.get("intermediate_size", 3072)


    class MockComplexFormerBackbone(PreTrainedModel): # Mock for the base
        config_class = CustomConfig
        def __init__(self, config):
            super().__init__(config)
            self.embedding = torch.nn.Embedding(config.vocab_size, config.hidden_size)
            self.dummy_layer = torch.nn.Linear(config.hidden_size, config.hidden_size)
        def forward(self, input_ids, attention_mask=None, **kwargs):
            # Returns (batch_size, seq_len, hidden_size)
            return self.dummy_layer(self.embedding(input_ids))

    class ComplexFormerForRewardModeling(PreTrainedModel):
        config_class = CustomConfig
        def __init__(self, config: CustomConfig):
            super().__init__(config)
            self.config = config
            # self.model = ComplexFormerModel(config) # Your base ComplexFormer
            self.model = MockComplexFormerBackbone(config) # Using Mock for now
            self.score_head = torch.nn.Linear(config.hidden_size, 1)
            self.score_head.apply(self._init_weights)

        def _init_weights(self, module):
            if isinstance(module, torch.nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()

        def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            last_hidden_state = outputs # if backbone directly returns hidden states
            if hasattr(outputs, 'last_hidden_state'):
                last_hidden_state = outputs.last_hidden_state
            
            if attention_mask is None:
                # This simple case assumes no padding or all sequences are full
                sequence_lengths = (torch.ones(input_ids.shape[0], device=input_ids.device) * (input_ids.shape[1] -1)).long()
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = input_ids.shape[0]
            
            # Handle cases where sequence_lengths might be -1 (e.g. empty attention_mask)
            sequence_lengths = torch.max(sequence_lengths, torch.zeros_like(sequence_lengths))
            
            last_token_hidden_state = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
            reward_score = self.score_head(last_token_hidden_state)
            return reward_score.squeeze(-1)
# --- End Mocking ---


logger: logging.Logger = logging.getLogger(__name__)
logger.propagate = False # To prevent duplicate logs if root logger is configured
# Clear existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
logger.addHandler(logging.FileHandler("./logs/train_reward_model.log", mode='w')) # Use mode 'w' to overwrite
logger.setLevel(logging.INFO) # Set level for the logger itself
traceback.install()

os.environ["WANDB_LOG_MODEL"] = "false" # Already good

# Data Collator for Reward Modeling
@dataclass
class RewardDataCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Features is a list of dicts, each dict is {"chosen_input_ids": ..., "rejected_input_ids": ...}
        # Or if you tokenized chosen and rejected separately in the dataset map function:
        # features is a list of dicts, like:
        # {
        #   'chosen_input_ids': [...], 'chosen_attention_mask': [...],
        #   'rejected_input_ids': [...], 'rejected_attention_mask': [...]
        # }

        batch = {}
        # Collate chosen and rejected inputs separately
        chosen_features = [{"input_ids": f["chosen_input_ids"], "attention_mask": f["chosen_attention_mask"]} for f in features]
        rejected_features = [{"input_ids": f["rejected_input_ids"], "attention_mask": f["rejected_attention_mask"]} for f in features]

        collated_chosen = self.tokenizer.pad(
            chosen_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        collated_rejected = self.tokenizer.pad(
            rejected_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch["chosen_input_ids"] = collated_chosen["input_ids"]
        batch["chosen_attention_mask"] = collated_chosen["attention_mask"]
        batch["rejected_input_ids"] = collated_rejected["input_ids"]
        batch["rejected_attention_mask"] = collated_rejected["attention_mask"]
        return batch


# Custom Trainer for Reward Modeling
class RewardTrainer(Trainer):
    def __init__(self, *args, debug: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug = debug
        self.avg_loss = 0.0 # For perplexity, not directly used for RM accuracy
        self.avg_accuracy = 0.0
        self.avg_reward_margin = 0.0
        # self._toggle_logger() # Renamed for clarity

    # def _toggle_logger(self): # Renamed
    #     if not self.debug:
    #         # This will affect the global logger, be careful if other parts need INFO
    #         # It's often better to manage logger levels at the handler level
    #         logging.getLogger("__main__").setLevel(logging.WARNING) # Or specific logger name

    def compute_loss(self, model: PreTrainedModel, inputs: Dict[str, torch.Tensor], return_outputs: bool = False):
        # inputs: {'chosen_input_ids': ..., 'chosen_attention_mask': ..., 'rejected_input_ids': ..., 'rejected_attention_mask': ...}
        rewards_chosen = model(
            input_ids=inputs["chosen_input_ids"],
            attention_mask=inputs["chosen_attention_mask"]
        )
        rewards_rejected = model(
            input_ids=inputs["rejected_input_ids"],
            attention_mask=inputs["rejected_attention_mask"]
        )

        # Pairwise ranking loss
        # loss = -torch.log(torch.sigmoid(rewards_chosen - rewards_rejected)).mean()
        loss = -F.logsigmoid(rewards_chosen - rewards_rejected).mean()

        # For logging metrics during training steps (if `logging_steps` is small)
        # This is more for quick checks than formal evaluation
        with torch.no_grad():
            self.current_batch_accuracy = (rewards_chosen > rewards_rejected).float().mean().item()
            self.current_batch_reward_margin = (rewards_chosen - rewards_rejected).mean().item()
            self.avg_loss = loss.item() # Keep track of raw loss

        return (loss, {"rewards_chosen": rewards_chosen, "rewards_rejected": rewards_rejected}) if return_outputs else loss

    def evaluation_loop(
        self,
        dataloader: torch.utils.data.DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Overriding evaluation_loop to compute custom metrics for reward model.
        """
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()

        all_rewards_chosen = []
        all_rewards_rejected = []
        total_loss = 0.0
        num_batches = 0

        for step, inputs in enumerate(dataloader):
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                rewards_chosen = model(
                    input_ids=inputs["chosen_input_ids"],
                    attention_mask=inputs["chosen_attention_mask"]
                )
                rewards_rejected = model(
                    input_ids=inputs["rejected_input_ids"],
                    attention_mask=inputs["rejected_attention_mask"]
                )
                loss = -F.logsigmoid(rewards_chosen - rewards_rejected).mean()

            # Gather across devices if using DDP/FSDP
            if self.accelerator.use_distributed:
                rewards_chosen = self.accelerator.gather(rewards_chosen)
                rewards_rejected = self.accelerator.gather(rewards_rejected)
                loss = self.accelerator.gather(loss).mean()


            all_rewards_chosen.append(rewards_chosen.cpu())
            all_rewards_rejected.append(rewards_rejected.cpu())
            total_loss += loss.item()
            num_batches += 1

        all_rewards_chosen_cat = torch.cat(all_rewards_chosen)
        all_rewards_rejected_cat = torch.cat(all_rewards_rejected)

        accuracy = (all_rewards_chosen_cat > all_rewards_rejected_cat).float().mean().item()
        avg_reward_margin = (all_rewards_chosen_cat - all_rewards_rejected_cat).mean().item()
        avg_loss = total_loss / num_batches

        metrics = {
            f"{metric_key_prefix}_loss": avg_loss,
            f"{metric_key_prefix}_accuracy": accuracy,
            f"{metric_key_prefix}_reward_margin": avg_reward_margin,
        }
        self.avg_accuracy = accuracy # Store for potential use by callbacks

        self.log(metrics)
        return metrics


# Hydra main function
@hydra.main(
    config_path=".", # Assuming your config.yaml is in the same directory
    config_name="config_rm", # Create a config_rm.yaml for reward model
    version_base=None,
)
def main(config):
    logger.info(f"Starting Reward Model training with config: \n{OmegaConf.to_yaml(config)}")

    # --- Tokenizer and Model Config Registration ---
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name,
        trust_remote_code=True,
        cache_dir=config.tokenizer_cache,
        local_files_only=config.tokenizer_local_files_only, # Added from your original
        use_fast=config.tokenizer_use_fast
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Tokenizer pad_token was None, set to eos_token: {tokenizer.eos_token}")


    # Register your custom config and model with AutoClass
    # This allows from_pretrained to find your custom classes
    AutoConfig.register(CustomConfig.model_type, CustomConfig) # model_type must match in config
    ComplexFormerForRewardModeling.register_for_auto_class("AutoModelForSequenceClassification") # Or a more fitting AutoModel class if available, or just AutoModel
    # For reward modeling, there isn't a perfect AutoModel class.
    # We can also just load it directly:
    # model = ComplexFormerForRewardModeling.from_pretrained(...)

    # --- Accelerator Setup ---
    ds_plugin = DeepSpeedPlugin(
        zero_stage=config.deepspeed.zero_stage,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
    ) if config.deepspeed.enabled else None

    dataloader_config = DataLoaderConfiguration(dispatch_batches=None, split_batches=False)
    accelerator = Accelerator(
        dataloader_config=dataloader_config,
        mixed_precision=config.training.mixed_precision,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        deepspeed_plugin=ds_plugin,
        log_with="wandb" if config.wandb.enabled and config.wandb.project else None, # Integrate W&B here
    )
    logger.info(f"Accelerator device: {accelerator.device}, Distributed type: {accelerator.distributed_type}")


    if config.mode == "train_rm":
        # --- Load Model ---
        model_config = CustomConfig.from_pretrained(
            config.model.config_load_path, # Path to a directory containing config.json for your model
            trust_remote_code=True,
            # Add other config overrides from your hydra config if needed
            vocab_size=len(tokenizer), # Ensure vocab size matches tokenizer
            pad_token_id=tokenizer.pad_token_id,
            # hidden_size=config.model.hidden_size, # Example override
        )
        logger.info(f"Loaded model config: {model_config}")

        if config.model.load_from_scratch:
            model = ComplexFormerForRewardModeling(model_config)
            logger.info("Initialized Reward Model from scratch.")
        else:
            # Load a pretrained backbone and add/replace the reward head
            # This assumes ComplexFormerForRewardModeling can handle loading a base ComplexFormerModel
            # and potentially ignoring/replacing the score_head if weights don't match.
            # Or, load the backbone ComplexFormerModel first, then instantiate ComplexFormerForRewardModeling(backbone, config)
            try:
                model = ComplexFormerForRewardModeling.from_pretrained(
                    config.model.custom_model_load_path, # Path to pretrained LM or RM checkpoint
                    config=model_config, # Pass the freshly loaded config
                    trust_remote_code=True,
                    # ignore_mismatched_sizes=True # Useful if only loading backbone
                )
                logger.info(f"Loaded Reward Model from {config.model.custom_model_load_path}")
            except Exception as e:
                logger.error(f"Error loading pretrained model: {e}. Initializing from scratch.")
                model = ComplexFormerForRewardModeling(model_config)


        # Initialize W&B if main process and enabled
        if accelerator.is_main_process and config.wandb.enabled:
            accelerator.init_trackers(
                project_name=config.wandb.project,
                config=OmegaConf.to_container(config, resolve=True),
                init_kwargs={"wandb": {"name": config.wandb.name}}
            )
            logger.info(f"WandB initialized for project {config.wandb.project} with run name {config.wandb.name}")


        # --- Dataset Loading and Preprocessing ---
        # Example: using a dummy dataset for now. Replace with your actual dataset loading.
        # Your dataset should have 'prompt', 'chosen', 'rejected' columns.
        def preprocess_reward_data(examples):
            # examples is a dict with keys 'prompt', 'chosen', 'rejected' (lists of strings)
            tokenized_chosen_list = []
            tokenized_rejected_list = []

            for prompt, chosen, rejected in zip(examples['prompt'], examples['chosen'], examples['rejected']):
                text_chosen = prompt + tokenizer.eos_token + chosen + tokenizer.eos_token # Add EOS for RM
                text_rejected = prompt + tokenizer.eos_token + rejected + tokenizer.eos_token

                tokenized_chosen = tokenizer(
                    text_chosen,
                    truncation=True,
                    max_length=config.data.max_seq_length, # Define in your config_rm.yaml
                    # padding="max_length" # Padding will be handled by collator
                )
                tokenized_rejected = tokenizer(
                    text_rejected,
                    truncation=True,
                    max_length=config.data.max_seq_length,
                    # padding="max_length"
                )
                tokenized_chosen_list.append(tokenized_chosen)
                tokenized_rejected_list.append(tokenized_rejected)
            
            # Restructure for batch output
            batch = {
                'chosen_input_ids': [tc['input_ids'] for tc in tokenized_chosen_list],
                'chosen_attention_mask': [tc['attention_mask'] for tc in tokenized_chosen_list],
                'rejected_input_ids': [tr['input_ids'] for tr in tokenized_rejected_list],
                'rejected_attention_mask': [tr['attention_mask'] for tr in tokenized_rejected_list],
            }
            return batch

        # Load your actual dataset here
        # For example, if using a dataset from Hugging Face Hub:
        # raw_datasets = load_dataset("your_preference_dataset_name", split="train")
        # Or load from disk:
        if os.path.exists(config.data.dataset_path):
            logger.info(f"Loading dataset from disk: {config.data.dataset_path}")
            raw_datasets = load_from_disk(config.data.dataset_path)
            if not isinstance(raw_datasets, DatasetDict): # If it's a single Dataset
                 raw_datasets = DatasetDict({"train": raw_datasets})
        else:
            logger.info(f"Dataset path {config.data.dataset_path} not found. Using dummy data.")
            dummy_data = {
                "prompt": ["Hello, how are you?", "What is the capital of France?"],
                "chosen": ["I am fine, thank you!", "The capital of France is Paris."],
                "rejected": ["I'm doing good.", "Paris is the capital."]
            }
            # Ensure lengths match for dummy data
            min_len = min(len(dummy_data["prompt"]), len(dummy_data["chosen"]), len(dummy_data["rejected"]))
            for k in dummy_data: dummy_data[k] = dummy_data[k][:min_len]

            raw_datasets = DatasetDict({"train": load_dataset("json", data_files={"train": [hydra.utils.to_absolute_path("./dummy_reward_data.json")]}, field="data")["train"] if os.path.exists(hydra.utils.to_absolute_path("./dummy_reward_data.json")) else Dataset.from_dict(dummy_data)})
            # Create a dummy json if it doesn't exist for testing
            if not os.path.exists(hydra.utils.to_absolute_path("./dummy_reward_data.json")):
                import json
                with open(hydra.utils.to_absolute_path("./dummy_reward_data.json"), "w") as f:
                    json.dump({"data": [{"prompt": p, "chosen": c, "rejected": r} for p,c,r in zip(dummy_data["prompt"], dummy_data["chosen"], dummy_data["rejected"])]}, f)


        logger.info(f"Raw datasets: {raw_datasets}")
        
        # Tokenize dataset
        # It's crucial that 'prompt', 'chosen', 'rejected' are columns in your dataset
        required_columns = ['prompt', 'chosen', 'rejected'] # Adjust if your column names are different
        if not all(col in raw_datasets["train"].column_names for col in required_columns):
            logger.error(f"Dataset must contain columns: {required_columns}. Found: {raw_datasets['train'].column_names}")
            # Fallback to dummy if columns are missing
            dummy_data = { "prompt": ["P1"], "chosen": ["C1"], "rejected": ["R1"] }
            raw_datasets = DatasetDict({"train": Dataset.from_dict(dummy_data)})
            logger.warning("Falling back to minimal dummy data due to missing columns.")


        tokenized_datasets = raw_datasets.map(
            preprocess_reward_data,
            batched=True,
            num_proc=config.data.preprocessing_num_workers,
            remove_columns=raw_datasets["train"].column_names, # Remove original text columns
            load_from_cache_file=not config.data.overwrite_cache,
            desc="Tokenizing preference data",
        )
        logger.info(f"Tokenized datasets: {tokenized_datasets}")

        if "train" not in tokenized_datasets:
            logger.error("No 'train' split found in tokenized_datasets. Exiting.")
            return

        if config.data.do_eval and "validation" not in tokenized_datasets:
             if len(tokenized_datasets["train"]) > 10: # Ensure there's enough data to split
                logger.info("No 'validation' split found. Splitting 'train' set.")
                split_dataset = tokenized_datasets["train"].train_test_split(test_size=config.data.eval_split_percentage, seed=config.training.seed)
                train_dataset = split_dataset["train"]
                eval_dataset = split_dataset["test"]
             else:
                logger.warning("Not enough data in 'train' set to create a validation split. Disabling evaluation.")
                train_dataset = tokenized_datasets["train"]
                eval_dataset = None
                config.data.do_eval = False
        else:
            train_dataset = tokenized_datasets["train"]
            eval_dataset = tokenized_datasets.get("validation", None) if config.data.do_eval else None


        logger.info(f"Training dataset size: {len(train_dataset)}")
        if eval_dataset:
            logger.info(f"Validation dataset size: {len(eval_dataset)}")
        else:
            logger.info("No validation dataset.")


        # --- Data Collator ---
        data_collator = RewardDataCollator(
            tokenizer=tokenizer,
            max_length=config.data.max_seq_length # Ensure this is consistent
        )

        # --- Training Arguments ---
        class EarlyStoppingCallbackRM(TrainerCallback):
            def __init__(self, early_stopping_patience=3, min_delta=0.001, metric_name="eval_accuracy"):
                self.patience = early_stopping_patience
                self.min_delta = min_delta
                self.metric_name = metric_name # For RM, accuracy is common
                self.best_metric = None
                self.no_improvement_counter = 0

            def on_evaluate(self, args, state, control, metrics, **kwargs):
                current_metric = metrics.get(self.metric_name)
                if current_metric is None:
                    logger.warning(f"Early stopping metric '{self.metric_name}' not found in metrics: {metrics.keys()}")
                    return

                if self.best_metric is None or current_metric > (self.best_metric + self.min_delta): # Higher is better for accuracy
                    self.best_metric = current_metric
                    self.no_improvement_counter = 0
                    logger.info(f"New best {self.metric_name}: {self.best_metric:.4f}")
                else:
                    self.no_improvement_counter += 1
                    logger.info(f"No improvement in {self.metric_name} for {self.no_improvement_counter} evaluations. Best: {self.best_metric:.4f}, Current: {current_metric:.4f}")
                    if self.no_improvement_counter >= self.patience:
                        logger.info(f"Stopping training early after {self.no_improvement_counter} evaluations without improvement.")
                        control.should_training_stop = True
        
        callbacks_list = []
        if config.training.early_stopping_patience > 0:
             callbacks_list.append(EarlyStoppingCallbackRM(
                 early_stopping_patience=config.training.early_stopping_patience,
                 metric_name=f"eval_{config.training.metric_for_best_model}" # Use the same metric
             ))


        training_args = TrainingArguments(
            output_dir=config.training.output_path,
            overwrite_output_dir=True,
            learning_rate=config.training.learning_rate,
            warmup_ratio=config.training.warmup_ratio,
            lr_scheduler_type=config.training.lr_scheduler_type,
            num_train_epochs=config.training.epochs,
            per_device_train_batch_size=config.training.batch_size,
            per_device_eval_batch_size=config.training.eval_batch_size, # Added eval_batch_size
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            gradient_checkpointing=config.training.gradient_checkpointing,
            save_strategy="steps" if config.data.do_eval else "epoch", # Save based on eval or epoch
            save_steps=config.training.save_steps if config.data.do_eval else len(train_dataset) // (config.training.batch_size * accelerator.num_processes * config.training.gradient_accumulation_steps), # Save per epoch if no eval
            save_total_limit=config.training.save_total_limit,
            # bf16=config.training.bf16, # Controlled by accelerator
            # fp16=config.training.fp16, # Controlled by accelerator
            logging_strategy="steps",
            logging_steps=config.training.log_steps, # Renamed from log_step
            evaluation_strategy="steps" if config.data.do_eval else "no",
            eval_steps=config.training.eval_steps if config.data.do_eval else None, # Renamed from log_step
            report_to="wandb" if config.wandb.enabled and accelerator.is_main_process else "none",
            remove_unused_columns=False, # Important for custom collator
            load_best_model_at_end=config.data.do_eval and config.training.early_stopping_patience > 0,
            metric_for_best_model=f"eval_{config.training.metric_for_best_model}" if config.data.do_eval else None, # e.g. eval_accuracy
            greater_is_better=config.training.metric_greater_is_better if config.data.do_eval else None, # True for accuracy
            run_name=config.wandb.name if config.wandb.enabled else None,
            max_grad_norm=config.training.max_grad_norm,
            deepspeed=config.deepspeed.config_path if config.deepspeed.enabled else None,
            seed=config.training.seed,
            # ddp_find_unused_parameters=False # May be needed depending on model structure and DDP
        )

        # --- Trainer ---
        trainer = RewardTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            debug=config.training.debug,
            callbacks=callbacks_list
        )

        # --- Resume from Checkpoint ---
        last_checkpoint = None
        if os.path.isdir(training_args.output_dir): # Check if output_dir exists
            # Use config.checkpointing.resume_ckpt_path if provided and valid
            if config.checkpointing.resume_ckpt_path and os.path.isdir(config.checkpointing.resume_ckpt_path):
                 last_checkpoint = config.checkpointing.resume_ckpt_path
                 logger.info(f"Attempting to resume from specified checkpoint: {last_checkpoint}")
            elif config.checkpointing.auto_resume: # auto_resume from output_dir
                potential_last_checkpoint = get_last_checkpoint(training_args.output_dir)
                if potential_last_checkpoint:
                    last_checkpoint = potential_last_checkpoint
                    logger.info(f"Automatically resuming from last checkpoint in output_dir: {last_checkpoint}")
            
            if last_checkpoint:
                logger.warning(f"Checkpoint detected, resuming training from: {last_checkpoint}")
            elif any(f.startswith("checkpoint-") for f in os.listdir(training_args.output_dir)) and config.checkpointing.auto_resume:
                 logger.warning(f"Checkpoint folders found in {training_args.output_dir}, but get_last_checkpoint failed. Ensure HF trainer state is saved.")
            else:
                 logger.info(f"No checkpoint found to resume from in {training_args.output_dir} or specified path. Starting training from scratch.")


        logger.info("Starting Reward Model training...")
        try:
            train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
            logger.info("Reward Model training finished.")
            
            if accelerator.is_main_process:
                metrics = train_result.metrics
                trainer.log_metrics("train", metrics)
                trainer.save_metrics("train", metrics)
                trainer.save_state() # Save trainer state
                logger.info(f"Saving final model to {training_args.output_dir}...")
                trainer.save_model(training_args.output_dir) # Saves the model using accelerator if applicable
                tokenizer.save_pretrained(training_args.output_dir)
                logger.info(f"Final model and tokenizer saved to {training_args.output_dir}")


            if config.data.do_eval:
                logger.info("Running final evaluation...")
                eval_metrics = trainer.evaluate()
                logger.info(f"Final evaluation metrics: {eval_metrics}")
                if accelerator.is_main_process:
                    trainer.log_metrics("eval", eval_metrics)
                    trainer.save_metrics("eval", eval_metrics)

        except Exception as e:
            logger.error(f"An error occurred during RM training: {e}", exc_info=True)
            # traceback.print_exc() # Already handled by logger exc_info=True
        finally:
            if accelerator.is_main_process and config.wandb.enabled:
                wandb.finish()
            accelerator.wait_for_everyone() # Ensure all processes sync before exiting


    elif config.mode == "eval_rm":
        if not config.data.do_eval:
            logger.info("Evaluation is disabled (config.data.do_eval=False). Exiting eval_rm mode.")
            return

        logger.info(f"Starting Reward Model evaluation on model: {config.model.custom_model_load_path}")
        # Load model (similar to training, but directly from the path to evaluate)
        model_config = CustomConfig.from_pretrained(
            config.model.custom_model_load_path, # Path to the RM checkpoint's config
            trust_remote_code=True,
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
        )
        model = ComplexFormerForRewardModeling.from_pretrained(
            config.model.custom_model_load_path, # Path to the RM checkpoint
            config=model_config,
            trust_remote_code=True,
        )

        # Prepare dataset (similar to training)
        # ... (Dataset loading and tokenization as in train_rm mode)
        # For brevity, assuming tokenized_datasets is loaded and eval_dataset exists
        if os.path.exists(config.data.eval_dataset_path): # Allow separate eval dataset
            logger.info(f"Loading eval dataset from disk: {config.data.eval_dataset_path}")
            raw_eval_datasets = load_from_disk(config.data.eval_dataset_path)
            if not isinstance(raw_eval_datasets, DatasetDict):
                 raw_eval_datasets = DatasetDict({"validation": raw_eval_datasets})
        elif os.path.exists(config.data.dataset_path): # Fallback to main dataset path
            logger.info(f"Loading main dataset for eval from disk: {config.data.dataset_path}")
            raw_eval_datasets = load_from_disk(config.data.dataset_path)
            if not isinstance(raw_eval_datasets, DatasetDict):
                 raw_eval_datasets = DatasetDict({"train": raw_eval_datasets}) # Assume 'train' to split later
        else: # Fallback to dummy
            logger.warning("No eval dataset path provided or found. Using dummy data for evaluation.")
            dummy_data = { "prompt": ["Eval P1"], "chosen": ["Eval C1"], "rejected": ["Eval R1"] }
            raw_eval_datasets = DatasetDict({"validation": Dataset.from_dict(dummy_data)})

        required_columns = ['prompt', 'chosen', 'rejected']
        eval_split_name = "validation" if "validation" in raw_eval_datasets else "train" # Prefer 'validation'
        if not all(col in raw_eval_datasets[eval_split_name].column_names for col in required_columns):
             logger.error(f"Eval dataset ({eval_split_name}) must contain {required_columns}. Found: {raw_eval_datasets[eval_split_name].column_names}")
             return

        tokenized_eval_datasets = raw_eval_datasets.map(
            preprocess_reward_data, batched=True, num_proc=config.data.preprocessing_num_workers,
            remove_columns=raw_eval_datasets[eval_split_name].column_names, desc="Tokenizing eval data"
        )
        eval_dataset = tokenized_eval_datasets.get(eval_split_name)

        if not eval_dataset:
            logger.error("Could not load or prepare evaluation dataset. Exiting.")
            return

        data_collator = RewardDataCollator(tokenizer=tokenizer, max_length=config.data.max_seq_length)
        
        training_args = TrainingArguments( # Only need output_dir and batch size for eval
            output_dir=os.path.join(config.training.output_path, "eval_only"),
            per_device_eval_batch_size=config.training.eval_batch_size,
            # report_to="none", # No need to report during standalone eval unless desired
        )

        trainer = RewardTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
        )
        
        logger.info("Running evaluation on the loaded Reward Model...")
        eval_metrics = trainer.evaluate()
        logger.info(f"Evaluation metrics for {config.model.custom_model_load_path}: {eval_metrics}")
        if accelerator.is_main_process:
            # Optionally save metrics to a file
            output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                logger.info(f"***** Eval results *****")
                for key, value in sorted(eval_metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    else:
        logger.error(f"Unknown mode: {config.mode}. Choose 'train_rm' or 'eval_rm'.")


if __name__ == "__main__":
    # For Hydra, configuration is typically managed via YAML files.
    # Create a `config_rm.yaml` in the same directory as this script.
    # Example `config_rm.yaml`:
    """
    # config_rm.yaml
    defaults:
      - _self_ # Inherit from this file

    tokenizer_name: "gpt2" # Or your tokenizer
    tokenizer_cache: "./tokenizer_cache"
    tokenizer_local_files_only: false
    tokenizer_use_fast: true

    mode: "train_rm" # "train_rm" or "eval_rm"

    model:
      config_load_path: "gpt2" # Path to a config.json for the base model
      custom_model_load_path: "gpt2" # Path to pretrained LM or RM checkpoint, or base model name
      load_from_scratch: false # If true, initializes RM from scratch using config_load_path
      # hidden_size: 768 # Example, if you want to override from base config

    data:
      dataset_path: "./prepared_preference_dataset_hf" # Path to HF dataset saved with save_to_disk
      eval_dataset_path: null # Optional: path to a separate eval dataset
      max_seq_length: 512
      preprocessing_num_workers: 4
      overwrite_cache: false
      do_eval: true
      eval_split_percentage: 0.05 # If no 'validation' split, split 'train'

    training:
      output_path: "./reward_model_output"
      learning_rate: 5.0e-6 # RM usually uses smaller LR
      warmup_ratio: 0.1
      lr_scheduler_type: "cosine"
      epochs: 3
      batch_size: 4 # Per device
      eval_batch_size: 8 # Per device
      gradient_accumulation_steps: 4
      gradient_checkpointing: false # Can enable if memory is an issue
      save_steps: 200
      eval_steps: 200
      log_steps: 50
      save_total_limit: 2
      # bf16 / fp16 settings from accelerator
      mixed_precision: "bf16" # "no", "fp16", "bf16"
      early_stopping_patience: 3 # Num evals without improvement, 0 to disable
      metric_for_best_model: "accuracy" # "accuracy" or "reward_margin"
      metric_greater_is_better: true # For accuracy=true, for loss=false
      max_grad_norm: 1.0
      seed: 42
      debug: false

    checkpointing:
      resume_ckpt_path: null # Specify a path like "./reward_model_output/checkpoint-1000"
      auto_resume: true # Try to resume from last checkpoint in output_path

    deepspeed:
      enabled: false # Set to true to use DeepSpeed
      config_path: null # Path to deepspeed_config.json
      zero_stage: 2 # Example

    wandb:
      enabled: true
      project: "complexformer_reward_model"
      name: "rm_run_1"
      # entity: "your_wandb_entity" # Optional
    """
    main()