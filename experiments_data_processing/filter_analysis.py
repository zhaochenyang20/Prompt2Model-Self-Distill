from datasets import load_from_disk
import matplotlib.pyplot as plt
import os
import numpy as np

root_paths = [
    '/home/azureuser/p2mss/p2mss/NI_task190_exp_5',
    '/home/azureuser/p2mss/p2mss/NI_task199_exp_5',
    '/home/azureuser/p2mss/p2mss/NI_task200_exp_5',
    '/home/azureuser/p2mss/p2mss/NI_task738_exp_5',
    '/home/azureuser/p2mss/p2mss/NI_task937_exp_5',
    '/home/azureuser/p2mss/p2mss/NI_task1385_exp_5',
    '/home/azureuser/p2mss/p2mss/NI_task1386_exp_5',
    '/home/azureuser/p2mss/p2mss/NI_task1516_exp_5',
    '/home/azureuser/p2mss/p2mss/NI_task1529_exp_5',
    '/home/azureuser/p2mss/p2mss/NI_task1612_exp_5',
    '/home/azureuser/p2mss/p2mss/NI_task1615_exp_5'
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

    # draw the figure
    plt.figure(figsize=(10, 6))
    plt.bar(reasons, proportions, color='skyblue')
    plt.xlabel('Drop Reason')
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=45, ha="right")  
    plt.title(f'{task_name} Percentage of Each Drop Reason')
    plt.tight_layout()  

    # save figure
    plt.savefig(os.path.join(root_path, 'drop_reasons_percentage.png'))

improvement = {
        'task199': 0.655-0.328,
        'task200': 0.5-0.47,
        'task738': 0.841-0.574,
        'task937': 0.587-0.486,
        'task1385': 0.349-0.34,
        'task1386': 0.320-0.314,
        'task1516': 0.443-0.157,
        'task1529': 0.669-0.098,
        'task1612': 0.479-0.513,
        'task1615': 0.53-0.604,
        'task190': 0.112-0.272
    }


# draw graph based on the keys of improvement
keys = improvement.keys()

# draw one graph for each reason 
for reason in portions:
    x = [improvement[key] for key in keys]
    y = [portions[reason][key] for key in keys]

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue')
    plt.title(f'{reason} Improvement vs. Proportions')
    plt.xlabel('Improvement')
    plt.ylabel('Portions')
    plt.grid(True)
    plt.savefig(os.path.join(f'./drop_reasons_{reason}_percentage.png'))
    plt.close()
    
    # culculate pearson correlation coefficient
    correlation_coefficient = np.corrcoef(x, y)[0, 1]
    print(f"{reason} Pearson correlation coefficient: {correlation_coefficient}")