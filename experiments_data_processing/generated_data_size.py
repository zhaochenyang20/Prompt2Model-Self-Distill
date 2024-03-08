import os
import csv
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from datasets import load_from_disk

root_path = '/home/azureuser/p2mss/p2mss/NI_task937_exp_6'

csv_data = []

for folder_name in os.listdir(root_path):
    if folder_name.startswith('task937'):
        _, generation_temperature, input_length_constraint, output_length_constraint, generation_epoch, _ = folder_name.split('_')
        dataset_path = os.path.join(root_path, folder_name, 'dataset')
        dataset = load_from_disk(dataset_path)
        num_rows = dataset.num_rows
        csv_data.append([generation_temperature, input_length_constraint, output_length_constraint, generation_epoch, num_rows])

csv_data.sort(key=lambda x: int(x[3]))
csv_file_path = 'dataset_summary.csv'

with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(['generation_temperature', 'input_length_constraint', 'output_length_constraint', 'generation_epoch', 'num_rows'])
    # Write the data
    for row in csv_data:
        writer.writerow(row)

epoch_data = defaultdict(lambda: {'sum': 0, 'count': 0, 'values': []})

# Populate epoch_data with sum and count
for row in csv_data:
    epoch = int(row[3])  # Convert generation_epoch to integer
    num_rows = int(row[4])
    epoch_data[epoch]['sum'] += num_rows
    epoch_data[epoch]['count'] += 1
    epoch_data[epoch]['values'].append(num_rows)

# Calculate average num_rows per epoch
averages = {epoch: data['sum'] / data['count'] for epoch, data in epoch_data.items()}
std_devs = {epoch: np.std(data['values'], ddof=1) for epoch, data in epoch_data.items()}  # ddof=1 for sample standard deviation

# Sort by epoch for plotting
sorted_epochs = sorted(averages.keys())
average_num_rows = [averages[epoch] for epoch in sorted_epochs]
std_num_rows = [std_devs[epoch] for epoch in sorted_epochs]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(sorted_epochs, average_num_rows, marker='o', linestyle='-', label='Average')
plt.fill_between(sorted_epochs, np.subtract(average_num_rows, std_num_rows), np.add(average_num_rows, std_num_rows), color='gray', alpha=0.2, label='Std Dev')

plt.xlabel('Generation Epoch')
plt.ylabel('Generated Dataset Size')
plt.title('Generated Dataset Size For each Generation Epoch')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(root_path, f"dataset_size_with_std.png"))
plt.close()