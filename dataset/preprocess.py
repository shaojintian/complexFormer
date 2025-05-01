import os
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

@hydra.main(config_path="pretrain", config_name="config")
def preprocess_and_save(config):
    """
    加载数据集，进行预处理，并保存预处理后的数据。
    """
    # 加载数据集
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
    print("Loading dataset...")
    dataset = load_dataset("parquet", data_files=data_files, split="train", columns=["text"])
    

    # 加载分词器
    print(f"Loading tokenizer: {config.tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

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

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存预处理后的数据
    print(f"Saving preprocessed dataset to {output_dir}...")
    preprocessed_dataset.save_to_disk(output_dir)
    print("Preprocessing and saving completed!")

if __name__ == "__main__":

    # 调用预处理函数
    preprocess_and_save()