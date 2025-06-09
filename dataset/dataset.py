from datasets import load_dataset
import os
import logging
from typing import List, Dict, Any
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer
import hydra
from rich import traceback
import polars as pl
from itertools import chain
traceback.install()

# 配置日志
logging.basicConfig(level=logging.INFO, filename="./logs/preprocess.log", format="%(asctime)s - %(levelname)s - %(message)s")
logger: logging.Logger = logging.getLogger(__name__)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # Disable parallelism for tokenizers


@hydra.main(config_path="../qwen", config_name="config")
def preprocess_and_save(config: Any) -> None:
    """
    加载数据集，进行预处理，并保存预处理后的数据。
    """
    # download数据集
    # 0--- Data Loading Function ---
    

    # 1.--- Load and Preprocess Dataset ---

    # Consider using streaming=True for very large datasets to uniformed format
    logger.info("Loading dataset...")
    #dataset: DatasetDict = load_dataset("parquet", data_files=data_files, columns=["text"], cache_dir="./data/huggingface_cache/datasets")
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en",cache_dir="./data/huggingface_cache/datasets")

    # 2.加载分词器
    logger.info(f"Loading tokenizer: {config.tokenizer_name}...")
    """
    huggingface-cli download FacebookAI/xlm-roberta-base --local-dir ./tokenizer/cache/FacebookAI/xlm-roberta-base --local-dir-use-symlinks False
    huggingface-cli download FacebookAI/xlm-roberta-base --local-dir ./tokenizer/cache/FacebookAI/xlm-roberta-base --local-dir-use-symlinks False
    """
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name,
        cache_dir=config.tokenizer_cache,
        trust_remote_code=True,
        local_files_only=True,
    )

    def preprocess_dataset(examples: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        """Preprocesses text: adds EOS, tokenizes, concatenates, and chunks."""
        from typing import List, Dict
        from itertools import chain
        
        block_size = 1024
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_name,
            cache_dir=config.tokenizer_cache,
            trust_remote_code=True,
            local_files_only=True,
        )
        eos_token_id = tokenizer.eos_token_id or tokenizer.convert_tokens_to_ids("<|im_end|>")
        input_ids_list = []

        # 1. Tokenize each text without special tokens
        for text in examples["text"]:
            if text is None:
                continue
            input_ids = tokenizer.encode(
                str(text), 
                add_special_tokens=True, 
                truncation=True, 
                max_length=tokenizer.model_max_length - 1  # 留一个位置给 eos
            )
            input_ids.append(eos_token_id)  # 2. 加 eos_token_id
            input_ids_list.append(input_ids)

        if not input_ids_list:
            return {"input_ids": [], "attention_mask": []}

        # 3. 拼接所有 token 成为长序列
        concatenated_ids = list(chain(*input_ids_list))
        total_length = (len(concatenated_ids) // block_size) * block_size
        if total_length == 0:
            return {"input_ids": [], "attention_mask": []}

        # 4. 分块
        input_blocks = [
            concatenated_ids[i:i+block_size] 
            for i in range(0, total_length, block_size)
        ]
        attention_masks = [
            [1] * block_size for _ in input_blocks
        ]

        return {"input_ids": input_blocks, "attention_mask": attention_masks}
    # 3.Apply preprocessing
    logger.info("Preprocessing dataset...")
    preprocessed_dataset: DatasetDict = dataset.map(
        preprocess_dataset,
        batched=True,
        batch_size=1000,  # Adjust based on memory
        remove_columns=['id', 'url', 'title', 'text'],  # Remove original text column
        num_proc=max(1, os.cpu_count() // 2),  # Use multiple cores safely
        #num_proc=1,  # For simplicity, use single process; adjust as needed
    )
    logger.info(f"Preprocessing finished. Number of examples: {preprocessed_dataset.shape}")
    logger.info(f"Preprocessing finished. Number of examples: {len(preprocessed_dataset)}")

    # 4.创建输出目录
    output_dir: str = config.data.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 保存预处理后的数据
    logger.info(f"Saving preprocessed dataset to {output_dir}...")
    preprocessed_dataset.save_to_disk(output_dir)
    logger.info("Preprocessing and saving completed!")

if __name__ == "__main__":
    # 调用预处理函数
    preprocess_and_save()