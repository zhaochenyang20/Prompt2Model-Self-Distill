from datasets import load_from_disk
import matplotlib.pyplot as plt
import os
import numpy as np

root_paths = [
    # '/home/azureuser/p2mss/p2mss/NI_task190_exp_5',
    # '/home/azureuser/p2mss/p2mss/NI_task199_exp_5',
    # '/home/azureuser/p2mss/p2mss/NI_task200_exp_5',
    # '/home/azureuser/p2mss/p2mss/NI_task738_exp_5',
    # '/home/azureuser/p2mss/p2mss/NI_task937_exp_5',
    # '/home/azureuser/p2mss/p2mss/NI_task1385_exp_5',
    # '/home/azureuser/p2mss/p2mss/NI_task1386_exp_5',
    # '/home/azureuser/p2mss/p2mss/NI_task1516_exp_5',
    # '/home/azureuser/p2mss/p2mss/NI_task1529_exp_5',
    # '/home/azureuser/p2mss/p2mss/NI_task1612_exp_5',
    # '/home/azureuser/p2mss/p2mss/NI_task1615_exp_5'
    '/home/azureuser/p2mss/p2mss/NI_task937_exp_6'
    ]

portions = {}

for root_path in root_paths:
    task_name = root_path.split('/')[-1].split('_')[1]

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
        folder_path = os.path.join(root_path, folder)  # Ensure the correct path is used to load the dataset
        if folder.split('_')[0] == task_name:
            dataset_path = os.path.join(folder_path, 'all_generated_data')
            dataset = load_from_disk(dataset_path)
            reasons_count = counter(dataset)



        # # 计算并打印每个drop_reason的比例
        # for reason, count in reasons_count.items():
        #     proportion = (count / total_generated) * 100  # 计算比例
        #     print(f"{reason}: {count} ({proportion:.2f}%)")
        

        # 准备数据
        total_generated = sum(reasons_count.values())
        reasons = list(reasons_count.keys())
        proportions = [(reasons_count[reason] / total_generated) * 100 for reason in reasons]  # 计算百分比

        # 绘制柱状图显示比例
        plt.figure(figsize=(10, 6))
        plt.bar(reasons, proportions, color='skyblue')
        plt.xlabel('Drop Reason')
        plt.ylabel('Percentage (%)')
        plt.xticks(rotation=45, ha="right")  # 旋转标签以便于阅读
        plt.title(f'{task_name} Percentage of Each Drop Reason')
        plt.tight_layout()  # 调整布局以防止标签被截断

        # 保存图表到文件
        plt.savefig(os.path.join(root_path, 'result/drop_reasons_percentage.png'))

    
    # # 计算皮尔逊相关系数
    # correlation_coefficient = np.corrcoef(x, y)[0, 1]
    # print(f"{reason} Pearson correlation coefficient: {correlation_coefficient}")