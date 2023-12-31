from datasets import load_from_disk
from collections import Counter
import os

# 文件夹 A 的路径
folder_a_path = '/home/cyzhao/prompt2model_test/testdataset/NI/eval'

# 遍历文件夹 A 中的所有数据集文件夹
for dataset_folder in os.listdir(folder_a_path):
    dataset_folder_path = os.path.join(folder_a_path, dataset_folder)

    # 加载数据集
    dataset = load_from_disk(dataset_folder_path)

    # 获取输出列数据
    output_col_data = dataset['output_col']

    # 统计不同字符串的数量
    output_col_counter = Counter(output_col_data)

    # 打印统计结果
    
    if len(output_col_counter.items()) > 5:
        continue
    print(f"Dataset in folder {dataset_folder}:")
    for string, count in output_col_counter.items():
        print(f"{string}: {count} occurrences")

    print("\n")
