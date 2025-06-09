import logging # 假设你已经配置了 logger
from datasets import DatasetDict, load_from_disk # 确保 datasets 已安装

# 假设 logger 和 config 已经定义好了
# import logging
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)
#
# class Config:
#     class Data:
#         output_dir = "path/to/your/preprocessed_dataset_directory" # <--- 修改这里
#     data = Data()
# config = Config()

logger = logging.getLogger(__name__)

def count_token(config,preprocessed_dataset: DatasetDict):
    logger.info(f"Loading preprocessed dataset from {config.data.output_dir}...")
    try:
        preprocessed_dataset: DatasetDict = load_from_disk(config.data.output_dir)
        # The original log line is a bit misleading. len(preprocessed_dataset) is the number of splits.
        logger.info(f"Loaded dataset with {len(preprocessed_dataset)} split(s): {list(preprocessed_dataset.keys())}.")

        total_tokens_all_splits = 0
        total_examples_all_splits = 0

        logger.info("Calculating token counts for each split...")
        for split_name, dataset_split in preprocessed_dataset.items():
            num_examples_in_split = len(dataset_split)
            total_examples_all_splits += num_examples_in_split
            logger.info(f"Processing split: '{split_name}' with {num_examples_in_split} examples.")

            if 'input_ids' in dataset_split.column_names:
                # Calculate tokens for the current split
                # Each item in dataset_split['input_ids'] is a list of token IDs for one example.
                # We sum the length of these lists.
                try:
                    # This is generally efficient for Hugging Face Datasets as it leverages Arrow
                    tokens_in_split = sum(len(ids) for ids in dataset_split['input_ids'])
                    logger.info(f"Split '{split_name}': {num_examples_in_split} examples, {tokens_in_split:,} tokens.")
                    total_tokens_all_splits += tokens_in_split

                    if num_examples_in_split > 0:
                        avg_tokens_per_example_split = tokens_in_split / num_examples_in_split
                        logger.info(f"Split '{split_name}': Average tokens per example: {avg_tokens_per_example_split:.2f}")

                except Exception as e:
                    logger.error(f"Error calculating tokens for split '{split_name}' using 'input_ids': {e}")
                    logger.warning(f"Skipping token count for split '{split_name}' due to error.")
            else:
                logger.warning(f"Column 'input_ids' not found in split '{split_name}'. Cannot count tokens for this split.")
            logger.info("-" * 30)

        logger.info("=" * 30)
        logger.info(f"Total examples across all splits: {total_examples_all_splits:,}")
        logger.info(f"Total tokens across all splits: {total_tokens_all_splits:,}")

        if total_examples_all_splits > 0 and total_tokens_all_splits > 0:
            overall_avg_tokens = total_tokens_all_splits / total_examples_all_splits
            logger.info(f"Overall average tokens per example: {overall_avg_tokens:.2f}")
        logger.info("=" * 30)

    except FileNotFoundError:
        logger.error(f"Error: Preprocessed dataset not found at {config.data.output_dir}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")