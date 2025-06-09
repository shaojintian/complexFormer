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
logger.addHandler(logging.FileHandler("./logs/preprocess.log"))
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # Disable parallelism for tokenizers

def find_files(dirs: List[str]) -> List[str]:
        files: List[str] = []
        base_data_dir: str = "data/pt"  # Define base data directory
        for dir_name in dirs:
            dir_path: str = os.path.join(base_data_dir, dir_name)
            if not os.path.isdir(dir_path):
                logger.warning(f"Directory not found: {dir_path}")
                continue
            for dirpath, _, filenames in os.walk(dir_path):
                for filename in filenames:
                    if filename.endswith(".parquet"):
                        full_path: str = os.path.join(dirpath, filename)
                        files.append(full_path)
                        # Uncomment to debug columns
                        # logger.debug(pl.read_parquet(full_path).columns)
        logger.info(f"Found {len(files)} data files.")
        return files

directories: List[str] = [
        "accommodation_catering_hotel", "artificial_intelligence_machine_learning",
        "computer_communication", "computer_programming_code", "film_entertainment",
        "literature_emotion", "news_media", "tourism_geography",
        "current_affairs_government_administration", "mathematics_statistics",]

@hydra.main(config_path="../pretrain", config_name="config")
def preprocess_and_save(config: Any) -> None:
    """
    加载数据集，进行预处理，并保存预处理后的数据。
    """
    # download数据集
    # 0--- Data Loading Function ---
    

    # 1.--- Load and Preprocess Dataset ---
    data_files: List[str] = find_files(directories)
    if not data_files:
        raise FileNotFoundError("No parquet files found in the specified directories.")

    # Consider using streaming=True for very large datasets to uniformed format
    logger.info("Loading dataset...")
    #dataset: DatasetDict = load_dataset("parquet", data_files=data_files, columns=["text"], cache_dir="./data/huggingface_cache/datasets")
    dataset = load_dataset("openai/gsm8k", "main",cache_dir="./data/huggingface_cache/datasets")

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
        # Use the correct EOS token for the base model/tokenizer
        eos_token: str = tokenizer.eos_token if tokenizer.eos_token else "<|im_end|>"  # Or specific token if needed
        # Ensure text field exists and handle potential None values
        texts: List[str] = [str(text) + eos_token for text in examples["text"] if text is not None]
        if not texts:
            return {"input_ids": [], "attention_mask": []}  # Return empty if no valid text

        # Tokenize (without adding special tokens here, EOS added manually)
        tokenized_examples: Dict[str, List[List[int]]] = tokenizer(texts, add_special_tokens=False, truncation=True, max_length=tokenizer.model_max_length)

        # Concatenate all tokenized sequences
        concatenated_examples: Dict[str, List[int]] = {
            k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()
        }
        total_length: int = len(concatenated_examples[list(concatenated_examples.keys())[0]])

        # Chunking (adjust block_size as needed)
        block_size: int = 1024
        if total_length < block_size:
            logger.warning(f"Total length {total_length} is less than block_size {block_size}. Skipping batch.")
            return {"input_ids": [], "attention_mask": []}  # Skip if not enough tokens for one block

        # We drop the small remainder chunk at the end to ensure fixed size blocks
        total_length = (total_length // block_size) * block_size

        # Split into chunks
        result: Dict[str, List[List[int]]] = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    # 3.Apply preprocessing
    logger.info("Preprocessing dataset...")
    preprocessed_dataset: DatasetDict = dataset.map(
        preprocess_dataset,
        batched=True,
        batch_size=1000,  # Adjust based on memory
        remove_columns=["text"],  # Remove original text column
        num_proc=max(1, os.cpu_count() // 2),  # Use multiple cores safely
    )
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