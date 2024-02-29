import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict

root_path = '/home/azureuser/p2mss/p2mss/NI_task937_exp_6'

data_by_params = defaultdict(lambda: defaultdict(list))

for folder_name in os.listdir(root_path):
    if folder_name.startswith('task937'):
        _, generation_temperature, input_length_constraint, output_length_constraint, generation_epoch, _ = folder_name.split('_')
        results_path = os.path.join(root_path, folder_name, 'result.json')
        
        # reading results.json
        with open(results_path, 'r') as file:
            results = json.load(file)
        
        # update epoch and corresponding score for each generation epoch
        for epoch, score in results.items():
            params_key = (generation_temperature, input_length_constraint, output_length_constraint)
            data_by_params[params_key][int(generation_epoch)].append((int(epoch), score))

# draw graph
for params, epochs_data in data_by_params.items():
    plt.figure(figsize=(10, 6))
    generation_epochs = sorted(epochs_data.keys())

    # calculate the average of 3 epoch score of a set of parameters for each generation epoch
    scores = [sum([score for _, score in epochs_data[epoch]]) / len(epochs_data[epoch]) for epoch in generation_epochs]  
    
    plt.plot(generation_epochs, scores, 'o-', label=f"Average Score")
    plt.title(f"Score Trend for Temp={params[0]}, InLen={params[1]}, OutLen={params[2]}")
    plt.xlabel("Generation Epoch")
    plt.ylabel("Average Score")
    plt.legend()
    
    # save graph
    filename = f"score_trend_temp_{params[0]}_inlen_{params[1]}_outlen_{params[2]}.png"
    plt.savefig(os.path.join(root_path, filename))
    plt.close()
