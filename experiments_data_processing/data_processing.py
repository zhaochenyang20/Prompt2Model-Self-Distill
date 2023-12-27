import os
from collections import defaultdict

import numpy as np
from datasets import load_from_disk
from sklearn.metrics import f1_score

# from IPython import embed


def count_values(example, counts, label):
    counts[example[label]] += 1
    return example


def find_best_directories(root_dir):
    best_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "best_ckpt_generated_content" in dirnames:
            best_dirs.append(os.path.join(dirpath, "best_ckpt_generated_content"))
    return best_dirs


root_dir = "/home/cyzhao/rerun_experiments"
# best_directories = find_best_directories(root_dir)
best_directories = ['/home/cyzhao/NI_task202_exp_3/best_ckpt_generated_content']
ground_truth_biased_count = 0
model_output_biased_count = 0
total = 0
both_biased = 0

ground_truth_labels = {}
all_labels = {}
correct_inputs = []

for directory in best_directories:
    data = load_from_disk(directory)
    parts = directory.split("/")
    task_name = parts[4]
    categories = 0
    biased = False
    y = []
    for label in ["groud_truth", "model_output"]:
        counts = defaultdict(int)
        data.map(
            count_values,
            fn_kwargs={"counts": counts, "label": label},
            keep_in_memory=True,
        )
        y.append(data[label])
        if label == "groud_truth":
            if len(counts.keys()) > 5:
                break
        print(task_name)
        for value, count in counts.items():
            print(f"'{value}': {count} times")
            if value not in all_labels:
                all_labels[value] = 0
        values = list(counts.values())
        mean = np.mean(values)
        std_dev = np.std(values)
        if label == "groud_truth":
            total += 1
            categories = len(counts.values())
            threshold = 10
            if std_dev < threshold:
                print(f"{label} data: not biased")
            else:
                print(f"{label} data: biased")
                biased = True
                ground_truth_biased_count += 1
            for value, count in counts.items():
                ground_truth_labels[value] = count
        elif label == "model_output":
            if len(counts.values()) < categories:
                print(f"{label} data: biased")
                model_output_biased_count += 1
                if biased:
                    both_biased += 1
            else:
                threshold = 20
                if std_dev < threshold:
                    print(f"{label} data: not biased")
                else:
                    print(f"{label} data: biased")
                    model_output_biased_count += 1
                    if biased:
                        both_biased += 1
        print(f"mean: {mean}, std: {std_dev}")
        if len(y) == 2:
            f1_weighted = f1_score(y[1], y[0], average="weighted")
            print(f"{task_name} f1 score: {f1_weighted}")
        if label == "model_output":
            print("=" * 50)
        else:
            print("-" * 50)
    for i in range(len(data['groud_truth'])):
        if data['groud_truth'][i] == data['model_output'][i]:
            all_labels[data['groud_truth'][i]]+=1
            correct_inputs.append(i)


for label in all_labels.keys():
    print(f"{label} = {all_labels[label]/ground_truth_labels[label]}")       

print(correct_inputs)
print(f"in all, we have {total} classification tasks")
print(f"ground truth biased ratio: {ground_truth_biased_count/total}")
print(f"model output biased ratio: {model_output_biased_count/total}")
print(f"both biased ratio: {both_biased/total}")


