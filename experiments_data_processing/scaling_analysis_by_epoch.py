import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict
from datasets import load_from_disk

task_name = 'task281'
root_path = f'/home/azureuser/p2mss/p2mss/generation_11/NI_{task_name}_exp_11'


data_by_epoch = defaultdict(lambda: defaultdict(list))

for folder_name in os.listdir(root_path):
    if folder_name.startswith(task_name):
        _, generation_temperature, input_length_constraint, output_length_constraint, _, _ = folder_name.split('_')
        results_path = os.path.join(root_path, folder_name, 'result.json')
        dataset = load_from_disk(os.path.join(root_path, folder_name, 'dataset'))
        generated_data_size = len(dataset)

        
        with open(results_path, 'r') as file:
            results = json.load(file)
        
        for epoch, score in results.items():
            params_key = (float(generation_temperature), input_length_constraint == 'True', output_length_constraint == 'True')
            data_by_epoch[int(epoch)][params_key].append((int(generated_data_size), score))

for epoch, params_data in data_by_epoch.items():
    plt.figure(figsize=(10, 6))
    for params, gen_epochs_scores in params_data.items():
        gen_epochs_scores.sort(key=lambda x: x[0])
        generated_data_sizes = [item[0] for item in gen_epochs_scores]
        scores = [item[1] for item in gen_epochs_scores]

        label = f"Temp={params[0]}, InLen={params[1]}, OutLen={params[2]}"
        plt.plot(generated_data_sizes, scores, '-o', label=label)

    plt.title(f"{task_name} Score Trend for Epoch {epoch}")
    plt.xlabel("Generation Dataset Size")
    plt.ylabel("Score")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(os.path.join(root_path, f"{task_name}_epoch_{epoch}_score_trend.png"))
    plt.close()
