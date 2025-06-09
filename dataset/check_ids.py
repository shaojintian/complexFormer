import os
from datasets import load_from_disk, DatasetDict, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

# --- Configuration (Assume these are available, e.g., from your 'config' object) ---

# Path to the saved preprocessed dataset directory
# Example: output_dir = "./preprocessed_data_xlm_roberta"
output_dir = config.data.output_dir 

# Name of the tokenizer used during preprocessing
# Example: tokenizer_name = "FacebookAI/xlm-roberta-base"
tokenizer_name = config.tokenizer_name 

# Optional: Path to the cache directory used for the tokenizer
# Example: tokenizer_cache = "./model_cache"
tokenizer_cache = getattr(config, 'tokenizer_cache', None) # Use None if not specified in config

# Number of examples to display from the dataset
num_examples_to_show = 3 
# ---------------------------------------------------------------------

print(f"Attempting to load preprocessed dataset from: {output_dir}")

try:
    # 1. Load the preprocessed dataset from disk
    preprocessed_dataset: DatasetDict = load_from_disk(output_dir)
    print(f"Dataset loaded successfully. Splits available: {list(preprocessed_dataset.keys())}")

    # 2. Load the corresponding tokenizer
    print(f"\nLoading tokenizer: {tokenizer_name}...")
    try:
        # Ensure you load the exact same tokenizer used for preprocessing
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            cache_dir=tokenizer_cache,
            use_fast=False # Usually good practice, use False if the original used slow
        )
        print("Tokenizer loaded successfully.")
    except OSError as e:
        print(f"Error loading tokenizer: {e}")
        print("Cannot proceed without the tokenizer to decode results.")
        exit()
    except Exception as e:
         print(f"An unexpected error occurred loading the tokenizer: {e}")
         exit()


    # 3. Inspect the results
    if not preprocessed_dataset:
        print("Loaded dataset is empty.")
    else:
        # Get the name of the first split (e.g., 'train')
        first_split_name = list(preprocessed_dataset.keys())[0]
        print(f"\n--- Inspecting first {num_examples_to_show} examples from split: '{first_split_name}' ---")
        
        split_dataset: Dataset = preprocessed_dataset[first_split_name]

        if len(split_dataset) == 0:
            print(f"The split '{first_split_name}' contains no examples.")
        else:
            for i in range(min(num_examples_to_show, len(split_dataset))):
                print(f"\n--- Example {i} ---")
                example = split_dataset[i]

                # --- Print common tokenization outputs ---
                
                # Print input_ids (the core token IDs)
                if 'input_ids' in example:
                    input_ids = example['input_ids']
                    print(f"Input IDs ({len(input_ids)}): {input_ids}")

                    # Convert IDs back to tokens
                    tokens = tokenizer.convert_ids_to_tokens(input_ids)
                    print(f"Tokens      ({len(tokens)}): {tokens}")

                    # Decode IDs back to string (including special tokens)
                    decoded_string_with_special = tokenizer.decode(input_ids, skip_special_tokens=False)
                    print(f"Decoded (w/ special): '{decoded_string_with_special}'")

                    # Decode IDs back to string (excluding special tokens)
                    decoded_string_no_special = tokenizer.decode(input_ids, skip_special_tokens=True)
                    print(f"Decoded (no special): '{decoded_string_no_special}'")

                else:
                    print("'input_ids' column not found in this example.")

                # Print attention_mask (shows padding)
                if 'attention_mask' in example:
                    attention_mask = example['attention_mask']
                    print(f"Attention Mask ({len(attention_mask)}): {attention_mask}")
                else:
                    print("'attention_mask' column not found.")
                    
                # Print labels (if present, e.g., for language modeling)
                if 'labels' in example:
                    labels = example['labels']
                    print(f"Labels ({len(labels)}): {labels}")
                else:
                     print("'labels' column not found (this is normal if not LM task).")

                # --- Optional: Print original text if it was kept ---
                # Adjust 'text' if your original column name was different
                if 'text' in example: 
                    original_text = example['text']
                    print(f"Original Text (if kept): '{original_text}'")
                
                print("-" * (len(f"--- Example {i} ---"))) # Separator

except FileNotFoundError:
    print(f"Error: Preprocessed dataset directory not found at '{output_dir}'")
except Exception as e:
    print(f"An unexpected error occurred loading or processing the dataset: {e}")
    import traceback
    traceback.print_exc()