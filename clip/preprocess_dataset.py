from datasets import load_dataset
from transformers import AutoImageProcessor
import torch
from torch.utils.data import DataLoader
import os

# --- 配置 ---
DATASET_NAME = 'BLIP3o/BLIP3o-Pretrain-Long-Caption'
DATASET_SPLIT = 'train'
CACHE_DIR_DATASET = "./dataset/cache_tiny_imagenet_dataloader"
CACHE_DIR_PROCESSOR = "./cache_processor" # 处理器也可以缓存
CACHE_DIR_PREPROCESSED_DATASET = "./dataset/preprocessed"

# 选择一个模型名称来获取其 ImageProcessor
# 这个处理器会决定目标图像大小和归一化参数
PROCESSOR_MODEL_NAME = "google/vit-base-patch16-224"
# PROCESSOR_MODEL_NAME = "microsoft/resnet-50"

BATCH_SIZE = 32 # 你想要的批处理大小

os.makedirs(CACHE_DIR_DATASET, exist_ok=True)
os.makedirs(CACHE_DIR_PROCESSOR, exist_ok=True)

def preprocess_and_create_dataloader(batch_size=BATCH_SIZE):
    """
    加载 Tiny ImageNet 数据集，使用 ImageProcessor 进行预处理，
    并创建一个 DataLoader 以生成指定形状的批次数据。
    """
    # --- 1. 加载原始数据集 ---
    print(f"Loading dataset: {DATASET_NAME}...")
    try:
        raw_dataset = load_dataset(
            DATASET_NAME,
            split=DATASET_SPLIT,
            cache_dir=CACHE_DIR_DATASET,
            trust_remote_code=True
        )
        print("Raw dataset loaded successfully.")
        print("Raw dataset sample [0]:")
        print(raw_dataset[0]) # 打印第一个原始样本信息
    except Exception as e:
        print(f"Error loading raw dataset: {e}")
        return None

    # --- 2. 加载 ImageProcessor ---
    print(f"\nLoading ImageProcessor for: {PROCESSOR_MODEL_NAME}...")
    try:
        image_processor = AutoImageProcessor.from_pretrained(
            PROCESSOR_MODEL_NAME,
            cache_dir=CACHE_DIR_PROCESSOR
        )
        print("ImageProcessor loaded successfully.")
    except Exception as e:
        print(f"Error loading ImageProcessor: {e}")
        return None

    # --- 3. 定义预处理函数 (应用到每个样本或批次) ---
    def transform_function(examples):
        """
        将 ImageProcessor 应用到一批图像上。
        """
        images_to_process = []
        for img in examples['image']:
            if img.mode != 'RGB': # 确保图像是 RGB 格式
                img = img.convert('RGB')
            images_to_process.append(img)

        # ImageProcessor 返回一个字典，包含 'pixel_values' (PyTorch Tensors)
        processed_inputs = image_processor(images=images_to_process, return_tensors="pt")
        # 我们只关心 'pixel_values'，并将其作为新列或覆盖旧列
        examples['pixel_values'] = processed_inputs['pixel_values']
        return examples

    # --- 4. 应用预处理到数据集 (生成 processed_dataset) ---
    print("\nApplying preprocessing to the dataset...")
    try:
        # 使用 .map() 来应用转换。这会添加 'pixel_values' 列。
        # 此时，每个 'pixel_values' 样本的形状是 (num_channels, height, width)
        processed_dataset = raw_dataset.map(
            transform_function,
            batched=True,
            batch_size=100, # map 操作的内部批处理大小，可以调整
            remove_columns=['image'] # 移除原始 PIL Image 列以节省内存
        )
        print("Dataset preprocessing complete.")
        print("\nProcessed dataset sample [0] (before DataLoader):")
        # 此时 processed_dataset[0]['pixel_values'] 是一个 (C, H, W) 的 Tensor
        print(f"  Label: {processed_dataset[0]['label']}")
        #print(f"  pixel_values shape: {processed_dataset[0]['pixel_values'].shape}")
        #print(f"  pixel_values dtype: {processed_dataset[0]['pixel_values'].dtype}")
    except Exception as e:
        print(f"Error during dataset .map() operation: {e}")
        return None

    # --- 5. 创建 DataLoader 以进行批处理 ---
    # DataLoader 会将 processed_dataset 中的样本组合成批次。
    # 它还需要知道如何从每个字典样本中提取出用于模型输入的张量。
    # Hugging Face datasets 与 PyTorch DataLoader 可以很好地集成。
    # 我们需要告诉 DataLoader 我们感兴趣的列是 'pixel_values' 和 'label'。
    # DataLoader 会自动将这些列中的张量堆叠起来形成批次。

    # 为了让 DataLoader 正确工作，通常需要将数据集的格式设置为 PyTorch 张量。
    # 如果 'pixel_values' 已经是 PyTorch 张量，并且 'label' 是标量或可转换为张量，
    # 那么 set_format 通常不是必需的，但明确设置可以避免一些问题。
    try:
        processed_dataset.set_format(type='torch', columns=['pixel_values', 'label'])
        print("\nDataset format set to 'torch'.")
    except Exception as e:
        print(f"Error setting dataset format: {e}")
        # 如果设置格式失败，DataLoader 可能仍能工作，但最好检查原因

    try:
        # 检查数据集的格式
        print(f"\nProcessed saved dataset format: {processed_dataset.format}")
        processed_dataset.save_to_disk(CACHE_DIR_PREPROCESSED_DATASET)
    except Exception as e:
        print(f"Error saved dataset format: {e}")

    print(f"\nCreating DataLoader with batch_size: {batch_size}...")
    # 对于训练，通常 shuffle=True。对于评估/测试，shuffle=False。
    # num_workers > 0 可以用于多进程数据加载。
    dataloader = DataLoader(
        processed_dataset,
        batch_size=batch_size,
        shuffle=True, # 在训练时通常打乱数据
        # num_workers=4, # 根据你的 CPU 核心数调整
        # pin_memory=True # 如果使用 GPU，可以加速数据传输
    )
    print("DataLoader created successfully.")
    
    return dataloader


# --- 主程序块 ---
if __name__ == "__main__":
    # 调用函数来获取 DataLoader
    train_dataloader = preprocess_and_create_dataloader(batch_size=BATCH_SIZE)

    if train_dataloader:
        print(f"\n--- Iterating through the DataLoader (first few batches) ---")
        num_batches_to_show = 2
        for i, batch in enumerate(train_dataloader):
            if i >= num_batches_to_show:
                break

            print(f"\nBatch {i+1}:")
            # 'batch' 是一个字典，键是你在 set_format 中指定的列名
            pixel_values_batch = batch['pixel_values']
            labels_batch = batch['label']

            print(f"  pixel_values_batch shape: {pixel_values_batch.shape}")
            # 输出应该是: torch.Size([BATCH_SIZE, num_channels, height, width])
            # 例如: torch.Size([32, 3, 224, 224])
            print(f"  pixel_values_batch dtype: {pixel_values_batch.dtype}")

            print(f"  labels_batch shape: {labels_batch.shape}")
            # 输出应该是: torch.Size([BATCH_SIZE])
            print(f"  labels_batch dtype: {labels_batch.dtype}")
            print(f"  First 5 labels in batch: {labels_batch[:5]}")
    else:
        print("Failed to create DataLoader.")

    print("\n--- Script Finished ---")