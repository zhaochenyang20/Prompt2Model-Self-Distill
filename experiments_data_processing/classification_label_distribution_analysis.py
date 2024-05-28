import re
from datasets import load_from_disk
from collections import Counter
from functools import partial

def find_last_occurrence(model_output: str, labels: list[str]) -> str:
    pattern = '|'.join(re.escape(label) for label in labels)
    regex = re.compile(pattern)
    matches = list(regex.finditer(model_output))
    return matches[-1].group() if matches else None

def get_distribution(results, labels):
    filter = partial(find_last_occurrence, labels=labels)
    filtered_results = []
    irrelevant = 0
    for result in results:
        res = filter(result)
        if res:
            filtered_results.append(res)
        else:
            irrelevant += 1
    total = len(results)
    filtered_results_counter = Counter(filtered_results)
    output_proportions = {key: value / total for key, value in filtered_results_counter.items()}
    output_proportions['irrelevant'] = irrelevant / total

    return output_proportions

def L1_distance(target_proportion, current_proportion):
    return sum(abs(target_proportion[key] - current_proportion.get(key, 0)) for key in target_proportion)

for task_name in ["task1516", "task1529", "task1612", "task1615", "task284", "task329", "task346"]:

    finetune_test_dataset = load_from_disk(f'/home/azureuser/p2mss/p2mss/{task_name}_rerun_test/inference_result')
    baseline_test_dataset = load_from_disk(f'/home/azureuser/p2mss/p2mss/baseline_generated_data/20240310_test_{task_name}')

    ground_truth = finetune_test_dataset['groud_truth']
    finetune_output = finetune_test_dataset['model_output']
    baseline_output = baseline_test_dataset['model_output']
    labels = list(Counter(ground_truth).keys())

    # ground_truth distribution
    ground_truth_counter = Counter(ground_truth)
    total = sum(ground_truth_counter.values())
    ground_truth_proportions = {key: value / total for key, value in ground_truth_counter.items()}
    ground_truth_proportions['irrelevant'] = 0

    # finetune_output distribution
    finetune_output_counter = Counter(finetune_output)
    total = sum(finetune_output_counter.values())
    finetune_output_proportions = {key: value / total for key, value in finetune_output_counter.items()}


    baseline_output_proportions = get_distribution(baseline_output, labels)

    finetune_distance = L1_distance(ground_truth_proportions, finetune_output_proportions)
    baseline_distance = L1_distance(ground_truth_proportions, baseline_output_proportions)
    print(ground_truth_proportions)
    print(finetune_output_proportions)
    print(baseline_output_proportions)
    print(f'{task_name}: finetune = {finetune_distance}, baseline = {baseline_distance} diff={baseline_distance-finetune_distance}')
