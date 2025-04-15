import os
from modelscope.msdatasets import MsDataset
import pandas as pd

def load_dataset():
    """下载并预处理数据集"""
    # 下载 ModelScope 的中英文数据集（示例：CSL 论文标题摘要）
    dataset = MsDataset.load(
        'csl_摘要',  # 数据集名称，可替换为其他数据集如 'POI'/'wenzhong' 等
        subset_name='default',  # 子集名称
        split='train',          # 训练集
        namespace='damo',       # 数据提供方
        cache_dir='./data'      # 数据保存到项目根目录的 data 文件夹
    )

    # 转换为 DataFrame 处理
    df = dataset.to_pandas()
    
    # 清洗数据：合并中文字段（示例字段，需根据实际数据集调整）
    texts = df['title'] + ' ' + df['abstract']
    
    # 保存为分词器训练格式（每行一个句子）
    os.makedirs('./data/corpus', exist_ok=True)
    with open('./data/corpus/train.txt', 'w', encoding='utf-8') as f:
        for text in texts:
            # 简单清洗：去除非中英文字符 + 换行符标准化
            cleaned = ''.join([c if ord(c) < 128 or '\u4e00' <= c <= '\u9fff' else ' ' for c in text])
            f.write(cleaned.replace('\n', ' ') + '\n')
    
    print(f"数据集已保存至：{os.path.abspath('./data/corpus/train.txt')}")

if __name__ == '__main__':
    load_dataset()