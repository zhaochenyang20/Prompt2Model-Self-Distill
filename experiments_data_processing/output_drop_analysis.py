from datasets import load_from_disk
from collections import Counter
import os
import matplotlib.pyplot as plt

task_name = 'task281'
root_path = f'/home/azureuser/p2mss/p2mss/generation_11/NI_{task_name}_exp_11'
res = {}

counts = {}
for folder_name in os.listdir(root_path):
    sub_dir_path = os.path.join(root_path, folder_name, 'output_recording')
    # task281_0.6_True_False_200_11
    if not(folder_name.startswith(task_name) and os.path.isdir(sub_dir_path)):
        continue
    dataset = load_from_disk(sub_dir_path)
    cd = Counter(dataset['drop_reason'])
    total = sum(cd.values())
    proportions = {key: value / total for key, value in cd.items()}
    _, generation_temperature, input_length_constraint, output_length_constraint, _, _ = folder_name.split('_')
    params = (generation_temperature, input_length_constraint, output_length_constraint)
    if params not in counts:
        counts[params] = 0
    counts[params] += 1
    if params not in res:
        res[params] = {}
    for key, value in proportions.items():
        if key not in res[params]:
            res[params][key] = 0
        res[params][key] += value


filters = ['ablation filter', 'consistent filter', 'random chosen']

for params in counts.keys():
    for key in filters:
        if key in res[params]:
            res[params][key] /= counts[params]


data_with_labels = []
for params, results in res.items():
    values = [results.get(filter_type, 0) for filter_type in filters]
    label = f"Params: {params}"
    data_with_labels.append((params, values, label))

data_with_labels.sort(key=lambda x: x[0])

plt.figure(figsize=(10, 6))  # 设置图表大小

# 遍历排序后的数据来绘制线条和添加图例
for params, values, label in data_with_labels:
    plt.plot(filters, values, label=label)

plt.title(f'{task_name} Output Filter Analysis')
plt.xlabel('Filter Type')
plt.ylabel('Value')

# 添加图例
plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(root_path, f"{task_name}_output_filter_analysis.png"))
plt.close()