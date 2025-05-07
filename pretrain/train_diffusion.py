from transformers import Trainer
from torch.nn import functional as F
import os
import logging
from typing import Optional, Tuple
import numpy as np
logger = logging.getLogger(__name__)


class DiffusionTrainer(Trainer):
    """
    Custom Trainer class for LLaDA-style training.
    """
    def __init__(self, *args,debug=False,**kwargs):
        super().__init__(*args, **kwargs)
        self.debug = debug
        self._togger_loger()
    def compute_loss(self, model, inputs, num_items_in_batch:Optional[int]=None,return_outputs=False):
        """
        diffusion loss function
        Args:
            model: The model to train.
            inputs: The inputs to the model.
            num_items_in_batch: Number of items in the batch.
            return_outputs: Whether to return the outputs of the model.
        """
        logger.info(f"Input IDs shape: {inputs['input_ids'].shape}")
        logger.info(f"Attention mask shape: {inputs['attention_mask'].shape}")
        # --- Forward Pass ---
        # logger.info(f"Model input shapes: {{k: v.shape for k, v in model_inputs.items()}}")
        outputs = model(**inputs)
        logsits = outputs.logits
        # --- Compute Loss ---
        labels = inputs.get("labels")
        logger.info(f"Logits shape: {logits.shape}")
        #logger.info(f"Labels shape: {labels.shape}")
        labels = inputs.get("labels")
        logits_flatten :Tensor = logits.view(-1, self.model.config.vocab_size)
        labels_flatten :Tensor = labels.view(-1)
        # grouped
        # Compute the loss using kl divergence
        loss = F.kl_div(logits_flatten, labels_flatten, reduction="batchmean")
        # Shift logits and labels for autoregressive loss
        
        return (loss, outputs) if return_outputs else loss

    def compute_metrics(self, eval_pred):
        # 计算分类指标
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        clf_results = clf_metrics.compute(predictions=preds, references=labels)
        
        # 添加 PPL
        clf_results["perplexity"] = np.exp(self.avg_loss)  
        return clf_results
    
    def _togger_loger(self ):
        if self.debug == False:
            logger.setLevel(logging.WARNING)

    def _save_checkpoint(self, model, trial, metrics=None):
        # Save the model and tokenizer
        output_dir = os.path.join(self.args.output_dir, f"checkpoint-{self.state.global_step}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.save_model(output_dir, _internal_call=True)
        self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))
        # Save the tokenizer
        logger.info(f"Tokenizer saved to {self.args.output_dir}")

# Load Model - Use AutoModel for flexibility with custom architectures
# Ensure the config used here matches the one saved in custom_model_load_path
# Or pass your loaded hf_config explicitly
# 注册自定义配置和模型

#t
# Resize token embeddings if new tokens were added
