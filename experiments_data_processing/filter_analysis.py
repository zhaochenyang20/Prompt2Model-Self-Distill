from datasets import load_from_disk
import matplotlib.pyplot as plt
import os
import numpy as np
import statistics

root_paths = [
    '/home/azureuser/p2mss/p2mss/classification_14/NI_task190_exp_14',
    '/home/azureuser/p2mss/p2mss/classification_14/NI_task199_exp_14',
    '/home/azureuser/p2mss/p2mss/classification_14/NI_task200_exp_14',
    '/home/azureuser/p2mss/p2mss/classification_14/NI_task284_exp_14',
    '/home/azureuser/p2mss/p2mss/classification_14/NI_task329_exp_14',
    '/home/azureuser/p2mss/p2mss/classification_14/NI_task346_exp_14',
    '/home/azureuser/p2mss/p2mss/classification_14/NI_task738_exp_14',
    '/home/azureuser/p2mss/p2mss/classification_14/NI_task937_exp_14',
    '/home/azureuser/p2mss/p2mss/classification_14/NI_task1385_exp_14',
    '/home/azureuser/p2mss/p2mss/classification_14/NI_task1386_exp_14',
    '/home/azureuser/p2mss/p2mss/classification_14/NI_task1516_exp_14',
    '/home/azureuser/p2mss/p2mss/classification_14/NI_task1529_exp_14',
    '/home/azureuser/p2mss/p2mss/classification_14/NI_task1612_exp_14',
    '/home/azureuser/p2mss/p2mss/classification_14/NI_task1615_exp_14',
    '/home/azureuser/p2mss/p2mss/generation_11/NI_task036_exp_11',
    '/home/azureuser/p2mss/p2mss/generation_11/NI_task039_exp_11',
    '/home/azureuser/p2mss/p2mss/generation_11/NI_task121_exp_11',
    '/home/azureuser/p2mss/p2mss/generation_11/NI_task281_exp_11',
    '/home/azureuser/p2mss/p2mss/generation_11/NI_task1195_exp_11',
    '/home/azureuser/p2mss/p2mss/generation_11/NI_task1345_exp_11',
    '/home/azureuser/p2mss/p2mss/generation_11/NI_task1562_exp_11',
    '/home/azureuser/p2mss/p2mss/generation_11/NI_task1622_exp_11'
    ]

portions = {}


for root_path in root_paths:
    task_name = root_path.split('/')[-1].split('_')[1]

    # init a dic to record the total count for different drop reason
    total_reasons_count = {}

    def counter(dataset):
        reasons_count = {}
        for item in dataset:
            reason = item['drop_reason']
            if reason not in reasons_count:
                reasons_count[reason] = 1
            else:
                reasons_count[reason] += 1
        return reasons_count

    for folder in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder)
        if folder.split('_')[0] == task_name:
            dataset_path = os.path.join(folder_path, 'all_generated_data')
            dataset = load_from_disk(dataset_path)
            reasons_count = counter(dataset)
            
            # update total count
            for reason, count in reasons_count.items():
                if reason not in total_reasons_count:
                    total_reasons_count[reason] = count
                else:
                    total_reasons_count[reason] += count

    # prepare data for drawing
    total_generated = sum(total_reasons_count.values())
    reasons = list(total_reasons_count.keys())
    proportions = [(total_reasons_count[reason] / total_generated) * 100 for reason in reasons]  # 计算百分比

    for reason in reasons:  
        if reason not in portions:
            portions[reason]={}
        portions[reason][task_name]=(total_reasons_count[reason] / total_generated)

avg = {}
std_dev = {}
range_2std = {}
for reason in portions:
    avg[reason] = sum(portions[reason][task] for task in portions[reason])/ len(portions[reason]) 
    tasks_values = list(portions[reason].values())
    std_dev[reason] = statistics.stdev(tasks_values)
    lower_bound = avg[reason] - 2 * std_dev[reason]
    upper_bound = avg[reason] + 2 * std_dev[reason]
    range_2std[reason] = (lower_bound, upper_bound)
print("Average:", avg)  # 打印每个reason的平均值
print("Standard Deviation:", std_dev)  # 打印每个reason的标准差
print("Range (Mean ± 2*SD):", range_2std)  # 打印均值±2倍标准差的范围 

    # # draw the figure
    # plt.figure(figsize=(10, 6))
    # plt.bar(reasons, proportions, color='skyblue')
    # plt.xlabel('Drop Reason')
    # plt.ylabel('Percentage (%)')
    # plt.xticks(rotation=45, ha="right")  
    # plt.title(f'{task_name} Percentage of Each Drop Reason')
    # plt.tight_layout()  

    # # save figure
    # plt.savefig(os.path.join(root_path, 'drop_reasons_percentage.png'))

improvement = {
    # 'task190': 39.70606905,
    # 'task199': 35.11606838,
    # 'task200': -5.361573034,
    # 'task738': 10.49757681,
    # 'task937': 8.138888889,
    # 'task1385': 2.356143498,
    # 'task1386': 3.123519553,
    # 'task1516': 35.62375,
    # 'task1529': 53.10798024,
    # 'task1612': -8.708024691,
    # 'task1615': 40.4541013,
    # 'task284': 0.3750205198,  
    # 'task329': 17.20365951,  
    # 'task346': 23.1655814,
    'task036': 31.2,
    'task121': 7.45,
    'task039': 19.76,
    'task281': 5.58,
    'task1195': 33.8,
    'task1345': 12.28,
    'task1562': 33.17,
    'task1622': 31.59
    }


# # draw graph based on the keys of improvement
# keys = improvement.keys()

# # draw one graph for each reason 
# for reason in portions:
#     x = [improvement[key] for key in keys]
#     y = [portions[reason].get(key,0) for key in keys]

#     plt.figure(figsize=(10, 6))
#     plt.scatter(x, y, color='blue')
#     plt.title(f'{reason} Improvement vs. Proportions')
#     plt.xlabel('Improvement')
#     plt.ylabel('Portions')
#     plt.grid(True)
#     plt.savefig(os.path.join(f'/home/azureuser/p2mss/p2mss/main/experiments_data_processing/drop_reasons_{reason}_percentage_generation.png'))
#     plt.close()
    
#     # culculate pearson correlation coefficient
#     correlation_coefficient = np.corrcoef(x, y)[0, 1]
#     print(f"{reason} Pearson correlation coefficient: {correlation_coefficient}")