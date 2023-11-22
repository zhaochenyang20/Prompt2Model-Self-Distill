import datasets
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

num_experiments = 7
valid_experiments = {1,2,4}

dataset_size = {} # size of filtered dataset
for i in valid_experiments:
    dataset_size[i] = []
    main_folder = f'/home/cyzhao/SQuAD_experiments_{i}'
    if os.path.exists(main_folder):
        for sub_folder in os.listdir(main_folder):
            sub_folder_path = os.path.join(main_folder, sub_folder)
            if os.path.isdir(sub_folder_path):
                dataset_path = os.path.join(sub_folder_path, "dataset")
                if os.path.exists(dataset_path):
                    dataset = datasets.load_from_disk(dataset_path)
                    dataset_size[i].append(dataset.num_rows)
                else:
                    print(f"Dataset file not found in: {sub_folder_path}")
    else:
        print(f"Main folder not found: {main_folder}")

generated_dataset_size = {}
select_ratio = {}
epoch_1_results = {}
epoch_2_results = {}
epoch_3_results = {}
for i in valid_experiments:
    generated_dataset_size[i]=[]
    select_ratio[i]=[]
    epoch_1_results[i]=[]
    epoch_2_results[i]=[]
    epoch_3_results[i]=[]
    file_path = f'/home/cyzhao/SQuAD_experiments_{i}/experiment_results.csv'
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        j = 0
        for row in csv_reader:
            generated_dataset_size[i].append(int(row[1])*int(row[2]))
            select_ratio[i].append(dataset_size[i][j]/generated_dataset_size[i][j])
            epoch_1_results[i].append(float(row[8]))
            epoch_2_results[i].append(float(row[9]))
            epoch_3_results[i].append(float(row[10]))
            j+=1

# for i in valid_experiments:
#     plt.figure()
#     x = range(len(select_ratio[i]))
#     y1 = select_ratio[i]
#     y2 = epoch_1_results[i]
#     y3 = epoch_2_results[i]
#     y4 = epoch_3_results[i]
#     plt.plot(x, y1, label='select_ratio')
#     plt.plot(x, y2, label='epoch_1_results')
#     plt.plot(x, y3, label='epoch_2_results')
#     plt.plot(x, y4, label='epoch_3_results')
    
#     plt.xlabel('params setting')
    
#     for j in range(len(x)):
#         plt.text(x[j], y1[j], f'{y1[j]:.2f}', ha='center', va='bottom')
#         plt.text(x[j], y2[j], f'{y2[j]:.2f}', ha='center', va='bottom')
#         plt.text(x[j], y3[j], f'{y3[j]:.2f}', ha='center', va='bottom')
#         plt.text(x[j], y4[j], f'{y4[j]:.2f}', ha='center', va='bottom')
    
#     plt.legend()
#     plt.title(f'Experiment {i}')
#     plt.savefig(f'experiment_{i}.png')



for i in valid_experiments:
    ratio = np.array(select_ratio[i])
    epoch_1 = np.array(epoch_1_results[i])
    epoch_2 = np.array(epoch_2_results[i])
    epoch_3 = np.array(epoch_3_results[i])


    correlation_coefficient_select_epoch1, _ = pearsonr(ratio, epoch_1)
    correlation_coefficient_select_epoch2, _ = pearsonr(ratio, epoch_2)
    correlation_coefficient_select_epoch3, _ = pearsonr(ratio, epoch_3)

    print(f"Pearson Correlation Coefficient (select_ratio vs. epoch_1_results): {correlation_coefficient_select_epoch1:.2f}")
    print(f"Pearson Correlation Coefficient (select_ratio vs. epoch_2_results): {correlation_coefficient_select_epoch2:.2f}")
    print(f"Pearson Correlation Coefficient (select_ratio vs. epoch_3_results): {correlation_coefficient_select_epoch3:.2f}")

# experiment 1
# Pearson Correlation Coefficient (select_ratio vs. epoch_1_results): 0.25
# Pearson Correlation Coefficient (select_ratio vs. epoch_2_results): 0.92
# Pearson Correlation Coefficient (select_ratio vs. epoch_3_results): 0.91

# experiment 2
# Pearson Correlation Coefficient (select_ratio vs. epoch_1_results): 0.83
# Pearson Correlation Coefficient (select_ratio vs. epoch_2_results): 0.93
# Pearson Correlation Coefficient (select_ratio vs. epoch_3_results): 0.93

# experiment 3
# Pearson Correlation Coefficient (select_ratio vs. epoch_1_results): 0.75
# Pearson Correlation Coefficient (select_ratio vs. epoch_2_results): 0.87
# Pearson Correlation Coefficient (select_ratio vs. epoch_3_results): 0.85