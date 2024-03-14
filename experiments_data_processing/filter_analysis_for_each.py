from datasets import load_from_disk
import os
import pandas as pd
import json

root_paths = [
    "/home/azureuser/p2mss/p2mss/generation_11/NI_task036_exp_11/task036_0.9_False_True_20_11",
    "/home/azureuser/p2mss/p2mss/generation_11/NI_task039_exp_11/task039_0.8_True_False_20_11",
    "/home/azureuser/p2mss/p2mss/generation_11/NI_task121_exp_11/task121_1.0_True_False_20_11",
    "/home/azureuser/p2mss/p2mss/generation_11/NI_task281_exp_11/task281_1.0_True_False_20_11",
    "/home/azureuser/p2mss/p2mss/generation_11/NI_task1195_exp_11/task1195_0.9_False_False_20_11",
    "/home/azureuser/p2mss/p2mss/generation_11/NI_task1345_exp_11/task1345_1.0_False_True_20_11",
    "/home/azureuser/p2mss/p2mss/generation_11/NI_task1562_exp_11/task1562_0.9_False_True_20_11",
    "/home/azureuser/p2mss/p2mss/generation_11/NI_task1622_exp_11/task1622_0.9_True_True_20_11"
    ]

data_for_csv = []

for root_path in root_paths:
    task_name = root_path.split('/')[-1].split('_')[0]

    def counter(dataset):
        reasons_count = {}
        for item in dataset:
            reason = item['drop_reason']
            if reason not in reasons_count:
                reasons_count[reason] = 1
            else:
                reasons_count[reason] += 1
        return reasons_count

    dataset_path = os.path.join(root_path, 'all_generated_data')
    dataset = load_from_disk(dataset_path)
    reasons_count = counter(dataset)

    total_generated = sum(reasons_count.values())
    json_file_path = os.path.join(root_path, 'result.json')

    with open(json_file_path, 'r') as file:
        data = json.load(file)
    accuracy = max(data.values())
    
    data_for_csv.append({
        'task name': task_name,
        'duplicated input ratio': reasons_count.get('duplicated input',0) / total_generated * 100,
        'ablation filter ratio': reasons_count.get('ablation filter',0) / total_generated * 100,
        'input length constrain ratio': reasons_count.get('input length constrain',0) / total_generated * 100,
        'accuracy': accuracy
    })

df = pd.DataFrame(data_for_csv)
csv_file_path = 'drop_reasons_statistics.csv'
df.to_csv(csv_file_path, index=False)
    
    # # 计算皮尔逊相关系数
    # correlation_coefficient = np.corrcoef(x, y)[0, 1]
    # print(f"{reason} Pearson correlation coefficient: {correlation_coefficient}")