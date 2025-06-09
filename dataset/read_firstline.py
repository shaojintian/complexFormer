from datasets import load_from_disk
from transformers import AutoTokenizer
import logging
import hydra

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(
    config_path="../qwen",
    config_name="config",
    version_base=None,
)
def main(config):
    # 1. 加载预处理数据集
    data_dir = "./data/pt/huggingface/preprocessed_datasets"  # 修改为你数据集路径
    preprocessed_dataset = load_from_disk(data_dir)
    logger.info(f"Loaded dataset splits: {list(preprocessed_dataset.keys())}")

    # 2. 查看某个 split 条数和字段
    split = "train"  # 你也可以改成 validation 或 test
    dataset = preprocessed_dataset[split]
    logger.info(f"Dataset '{split}' size: {len(dataset)}")
    logger.info(f"Dataset columns: {dataset.column_names}")

    # 3. 加载 tokenizer
    print(f"Loading tokenizer from {config.tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name,trust_remote_code=True,local_files_only=True,cache_dir=config.tokenizer_cache)

    # 4. 解码并打印几个样本
    for i in range(min(2, len(dataset))):
        input_ids = dataset[i]["input_ids"]  # 假设你的字段是 input_ids
        logger.info(f"Sample {i} input_ids: {input_ids}")
        decoded_text = tokenizer.decode(input_ids, skip_special_tokens=False)
        logger.info(f"Sample {i} decoded text: {decoded_text}")

if __name__ == "__main__":
    main()
