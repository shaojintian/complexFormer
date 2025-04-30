import os
import torch
from itertools import chain
from datasets import load_dataset
import hydra
from transformers import (
    AutoConfig,
    # AutoModelForCausalLM, # Not needed if using AutoModel with custom
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer, # We use our custom trainer now
    TrainingArguments,
    AutoModel
)
from utils.utils import load_config, get_gpu_memory
# from torch.utils.tensorboard import SummaryWriter # Trainer handles TensorBoard
# Make sure LLaDATrainer is imported or defined before use
# from your_trainer_file import LLaDATrainer # Example import
# Or define the class directly in this script as shown above
import torch.nn.functional as F
from transformers import Trainer
import math # Import math for isnan check
from transformers.trainer_utils import get_last_checkpoint
from rich import traceback

# 启用彩色回溯
traceback.install()

class DiffusionTrainer(Trainer):
    def __init__(self, *args, mask_token_id=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Crucially requires the specific ID used for masking during LLaDA's training
        self.mask_token_id = mask_token_id
        if self.mask_token_id is None:
            raise ValueError("mask_token_id must be provided for LLaDATrainer")
        print(f"LLaDATrainer initialized with mask_token_id: {self.mask_token_id}")
        # You might also need tokenizer's pad_token_id if applicable
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer else -100 # Default ignore_index


    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Overrides the default loss computation for LLaDA-style training.

        Args:
            model: The model to train.
            inputs: A dictionary potentially containing 'input_ids', 'attention_mask'.
                    'input_ids' are expected to be the *original* token sequences.
                    'labels' from the default collator will be ignored.
            return_outputs (bool): Whether to return model outputs alongside the loss.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, Dict]]: The loss tensor, optionally
            with model outputs.
        """
        # 1. Get Original IDs (Targets)
        # We assume the 'input_ids' from the collator ARE the original sequences.
        original_ids = inputs.get("input_ids")
        labels = original_ids.clone() # The target is always the original sequence

        # 2. Apply Forward Masking Process (Training Version)
        # Sample a mask ratio 't' uniformly for each sequence in the batch
        # t determines the *probability* for each token to be masked.
        t = torch.rand(original_ids.shape[0], 1, device=original_ids.device) # Shape: (B, 1)

        # Generate random numbers for each token position
        mask_probs = torch.rand_like(original_ids, dtype=torch.float32) # Shape: (B, L)

        # Create the mask: True where mask_probs < t
        # Use broadcasting: Compare (B, L) with (B, 1)
        is_mask = mask_probs < t

        # Prevent masking padding tokens (if padding exists and pad_token_id is known)
        if self.pad_token_id != -100:
            padding_mask = original_ids.eq(self.pad_token_id)
            is_mask = is_mask & (~padding_mask) # Don't mask padding positions

        # Create the noisy input by replacing masked positions
        noisy_input_ids = torch.where(is_mask, self.mask_token_id, original_ids)

        # 3. Prepare Inputs for the Model
        # The model receives the 'noisy' sequence
        model_inputs = {"input_ids": noisy_input_ids}
        if "attention_mask" in inputs:
            # Use the original attention mask, as masked tokens should still be attended to
            model_inputs["attention_mask"] = inputs.get("attention_mask")

        # 4. Forward Pass
        # Get model predictions (logits) based on the noisy input
        outputs = model(**model_inputs)
        logits = outputs.logits # Shape: (B, L, VocabSize)

        # 5. Calculate Loss ONLY on Masked Positions
        loss_fct = torch.nn.CrossEntropyLoss() # Default reduction='mean'

        # Select the logits and corresponding labels ONLY where is_mask is True
        masked_logits = logits[is_mask] # Shape: (NumMaskedTokens, VocabSize)
        masked_labels = labels[is_mask] # Shape: (NumMaskedTokens)

        # Handle the edge case where no tokens were masked (e.g., t=0 or all padding)
        if masked_labels.numel() == 0:
            # Return a zero loss tensor, ensuring it's on the correct device
            # and requires grad if in training mode.
            loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            if self.args.past_index >= 0: # Check if manipulating past_key_values
                 self._past = outputs[self.args.past_index]

            if model.training:
                 loss.requires_grad_() # Important for backward pass if loss is 0
        else:
            loss = loss_fct(masked_logits.view(-1, logits.size(-1)), masked_labels.view(-1))

        # Check for NaN loss, which can happen with bf16/fp16 instability
        if math.isnan(loss.item()):
             print("Warning: NaN loss detected!")
             print(f"t values: {t.squeeze().tolist()}")
             print(f"Number of masked tokens: {masked_labels.numel()}")
             # Optionally, you could return a large finite loss or skip the step,
             # but raising an error or logging is usually better for debugging.
             # Here, we'll let it propagate for now.

        return (loss, outputs) if return_outputs else loss

def compute_metrics_simple(eval_pred):
        # This won't reflect the masked loss, just a standard eval loss if needed
        # Or you could implement masked eval loss here too
        logits, labels = eval_pred
        # Shift logits and labels for standard Causal LM eval loss
        # shift_logits = logits[..., :-1, :].contiguous()
        # shift_labels = labels[..., 1:].contiguous()
        # loss_fct = torch.nn.CrossEntropyLoss()
        # loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        # return {"eval_loss": loss.item()}
        # For now, let's just return GPU memory as a placeholder metric during eval
        try:
            gpu_allocation_str = get_gpu_memory()
            gpu_allocation = float(gpu_allocation_str.split()[0]) # Extract number
        except:
            gpu_allocation = 0.0
        return {"gpu_allocation_gb": gpu_allocation}

# 设置环境变量以优化CUDA内存分配 (Optional, keep if needed)
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
@hydra.main(
    config_path=".",
    config_name="config",
    version_base=None,
)
def main(config):
    # --- Configuration ---
    output_path = "results/nightmare_1.5b" # Changed output path
    model_path = "./tokenizer" # Base for tokenizer
    custom_model_load_path = "./pretrain/" # Path to your custom model config/weights
    config_yaml_path = "./pretrain/config.yaml" # Path to your YAML config

    # --- Load Model and Tokenizer ---
    self_config = load_config(config_yaml_path) # Load your custom config dict/object
    # If self_config needs conversion to a HuggingFace Config object:
    # from transformers import PretrainedConfig
    # if isinstance(self_config, dict):
    #     # Assuming your CustomConfig can be initialized from a dict
    #     # Or map fields manually if needed
    #     hf_config = CustomConfig(**self_config)
    # else:
    #     hf_config = self_config # Assume it's already a HF compatible config object

    # Ensure CustomConfig and MyDecoderOnlyModel are registered *before* from_pretrained
    # This registration might need to be adapted based on how your CustomConfig/Model work
    # Assuming CustomConfig is a class inheriting from PretrainedConfig
    # Assuming MyDecoderOnlyModel inherits from PreTrainedModel
    # AutoModel.register(CustomConfig, MyDecoderOnlyModel) # You might need specific class names

    # Load Tokenizer - Use the original base model's tokenizer usually
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # *** Important: Define or add the [MASK] token if it doesn't exist ***


    # Load Model - Use AutoModel for flexibility with custom architectures
    # Ensure the config used here matches the one saved in custom_model_load_path
    # Or pass your loaded hf_config explicitly
    model = AutoModel.from_pretrained(
        custom_model_load_path,
        # config=hf_config, # Pass your loaded config if needed
        trust_remote_code=True # Be cautious with this
    )

    # Resize token embeddings if new tokens were added


    # --- Calculate Parameters ---
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad) # Count only trainable
    print(f"Trainable model parameters: {num_params:,}")


    # --- Data Loading Function ---
    def find_files(dirs):
        files = []
        base_data_dir = "data/pt" # Define base data directory
        for dir_name in dirs:
            dir_path = os.path.join(base_data_dir, dir_name)
            if not os.path.isdir(dir_path):
                print(f"Warning: Directory not found {dir_path}")
                continue
            for dirpath, _, filenames in os.walk(dir_path):
                for filename in filenames:
                    if filename.endswith(".parquet"):
                        full_path = os.path.join(dirpath, filename)
                        files.append(full_path)
        print(f"Found {len(files)} data files.")
        return files


    # --- Load and Preprocess Dataset ---
    directories = [
        "accommodation_catering_hotel", "artificial_intelligence_machine_learning",
        "computer_communication", "computer_programming_code", "film_entertainment",
        "literature_emotion", "news_media", "tourism_geography",
        "current_affairs_government_administration", "mathematics_statistics",
    ]
    data_files = find_files(directories)
    if not data_files:
        raise FileNotFoundError("No parquet files found in the specified directories.")

    # Consider using streaming=True for very large datasets
    dataset = load_dataset("parquet", data_files=data_files, split="train", columns=["text"])

    # Select a smaller subset for debugging if needed
    # dataset = dataset.shuffle(seed=42).select(range(10000)) # Example subset

    def preprocess_dataset(examples):
        """Preprocesses text: adds EOS, tokenizes, concatenates, and chunks."""
        # Use the correct EOS token for the base model/tokenizer
        eos_token = tokenizer.eos_token if tokenizer.eos_token else "<|im_end|>" # Or specific token if needed
        # Ensure text field exists and handle potential None values
        texts = [str(text) + eos_token for text in examples["text"] if text is not None]
        if not texts:
            return {"input_ids": [], "attention_mask": []} # Return empty if no valid text

        # Tokenize (without adding special tokens here, EOS added manually)
        tokenized_examples = tokenizer(texts, add_special_tokens=False, truncation=False)

        # Concatenate all tokenized sequences
        concatenated_examples = {
            k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()
        }
        total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])

        # Chunking (adjust block_size as needed)
        block_size = 1024
        if total_length < block_size:
            print(f"Warning: Total length {total_length} is less than block_size {block_size}. Skipping batch.")
            return {"input_ids": [], "attention_mask": []} # Skip if not enough tokens for one block

        # We drop the small remainder chunk at the end to ensure fixed size blocks
        total_length = (total_length // block_size) * block_size

        # Split into chunks
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        # Add labels (which are just copies of input_ids for our loss)
        # result["labels"] = result["input_ids"].copy() # Not needed, our trainer uses input_ids
        return result

    # Apply preprocessing
    print("Preprocessing dataset...")
    preprocessed_dataset = dataset.map(
        preprocess_dataset,
        batched=True,
        batch_size=1000, # Adjust based on memory
        remove_columns=dataset.column_names,
        num_proc=max(1, os.cpu_count() // 2), # Use multiple cores safely
    )
    print(f"Preprocessing finished. Number of examples: {len(preprocessed_dataset)}")

    # Split into train/test
    split_dataset = preprocessed_dataset.train_test_split(test_size=0.01, shuffle=True, seed=42) # Smaller test set common for PT
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")


    # --- Data Collator ---
    # Use the standard LM collator. It will handle padding.
    # Our custom trainer ignores the 'labels' it creates and uses 'input_ids'.
    collator = dataloader.load_dataset(tokenizer=tokenizer, mlm=False)

    # --- Compute Metrics (Optional but good for monitoring) ---
    

    # --- Training Arguments ---
    training_args = TrainingArguments(
        output_dir=config.train.output_path,
        overwrite_output_dir=True,
        learning_rate=1e-4, # Adjust as needed
        warmup_ratio=0.05, # Common ratio for large PT
        lr_scheduler_type="cosine",
        num_train_epochs=1, # Usually many more for pre-training
        per_device_train_batch_size=4, # Adjust based on VRAM (12 might be too high for 14B)
        per_device_eval_batch_size=4,  # Adjust based on VRAM
        gradient_accumulation_steps=16, # Increase to simulate larger batch size (e.g., 4*16*num_gpus = effective batch)
        gradient_checkpointing=True, # Saves VRAM at cost of ~20-30% slower training
        save_strategy="steps",
        save_steps=10000, # Save more frequently during long PT
        save_total_limit=3, # Keep last 3 checkpoints
        bf16=True, # Use BFloat16 if supported (Ampere GPUs+)
        # fp16=True, # Use FP16 if BF16 not supported
        logging_strategy="steps",
        logging_steps=100, # Log more frequently
        evaluation_strategy="steps", # Evaluate periodically
        eval_steps=5000, # Evaluate every 5k steps
        report_to="tensorboard", # Log to TensorBoard
        remove_unused_columns=False, # Important: Keep input_ids for our custom loss
        # load_best_model_at_end=True, # Optional: Reload best checkpoint at the end
    )

    # --- Initialize Custom Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_simple, # Add back if you want eval metrics
        mask_token_id=mask_token_id, # Pass the determined mask token ID
    )

    # --- Detect Last Checkpoint for Resuming ---
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint:
            print(f"Checkpoint detected, resuming training from: {last_checkpoint}")
        elif any(f.startswith("checkpoint-") for f in os.listdir(training_args.output_dir)):
            print(f"Warning: Checkpoint folders found in {training_args.output_dir}, but get_last_checkpoint failed. Specify path manually if needed.")
        else:
            print(f"No checkpoint found in {training_args.output_dir}. Starting training from scratch.")

    # --- Start Training ---
    print("Starting training...")
    try:
        # Pass resume_from_checkpoint=True to automatically use last_checkpoint if found
        # Or pass the specific path: resume_from_checkpoint=last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

        # train_result contains metrics like train_runtime, train_samples_per_second etc.
        print("Training finished.")
        print(f"Train Results: {train_result.metrics}")

        # Save final training metrics
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)

    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
        # Optionally save state even if training failed mid-way
        print("Attempting to save trainer state due to error...")
        trainer.save_state() # Saves optimizer, scheduler, rng states etc. in output_dir/trainer_state.json


    # --- Save Final Model ---
    # This saves the model state at the end of training (or after the last successful save step if interrupted)
    print("Saving final model...")
    # Ensure the final save path exists
    final_save_path = os.path.join(training_args.output_dir, "final_model")
    os.makedirs(final_save_path, exist_ok=True)

    trainer.save_model(final_save_path) # Saves model weights, config
    tokenizer.save_pretrained(final_save_path) # Saves tokenizer files
    print(f"Final model and tokenizer saved to {final_save_path}")

    # --- Save Final Model ---
    print("Saving final model...")
    save_path = os.path.join(output_path, "final_model")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model and tokenizer saved to {save_path}")




if __name__ == "__main__":
    # This is just a placeholder. The script is designed to be run directly.
    # You can add command-line argument parsing if needed.
    main()