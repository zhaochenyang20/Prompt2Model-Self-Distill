import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict

root_path = '/home/azureuser/p2mss/p2mss/NI_task937_exp_6'

data_by_epoch = defaultdict(lambda: defaultdict(list))

for folder_name in os.listdir(root_path):
    if folder_name.startswith('task937'):
        _, generation_temperature, input_length_constraint, output_length_constraint, generation_epoch, _ = folder_name.split('_')
        results_path = os.path.join(root_path, folder_name, 'result.json')
        
        with open(results_path, 'r') as file:
            results = json.load(file)
        
        for epoch, score in results.items():
            params_key = (float(generation_temperature), input_length_constraint == 'True', output_length_constraint == 'True')
            data_by_epoch[int(epoch)][params_key].append((int(generation_epoch), score))

for epoch, params_data in data_by_epoch.items():
    plt.figure(figsize=(10, 6))
    for params, gen_epochs_scores in params_data.items():
        gen_epochs_scores.sort(key=lambda x: x[0])
        generation_epochs = [item[0] for item in gen_epochs_scores]
        scores = [item[1] for item in gen_epochs_scores]

        label = f"Temp={params[0]}, InLen={params[1]}, OutLen={params[2]}"
        plt.plot(generation_epochs, scores, '-o', label=label)

    plt.title(f"Score Trend for Epoch {epoch}")
    plt.xlabel("Generation Epoch")
    plt.ylabel("Score")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(os.path.join(root_path, f"epoch_{epoch}_score_trend.png"))
    plt.close()
