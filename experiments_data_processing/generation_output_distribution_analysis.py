from datasets import load_from_disk
from collections import Counter
import scipy.stats as stats

for task_name in ["task1345", "task281", "task1562", "task1622"]:

    finetune_test_dataset = load_from_disk(f'/home/azureuser/p2mss/p2mss/{task_name}_rerun_test/inference_result')
    baseline_test_dataset = load_from_disk(f'/home/azureuser/p2mss/p2mss/baseline_generated_data/20240310_test_{task_name}')

    ground_truth = finetune_test_dataset['groud_truth']
    finetune_output = finetune_test_dataset['model_output']
    baseline_output = baseline_test_dataset['model_output']
    
    ground_truth_lengths = [len(item) for item in ground_truth]
    finetune_output_lengths = [len(item) for item in finetune_output]
    baseline_output_lengths = [len(item) for item in baseline_output]

    print(task_name)
    print("Shapiro-Wilk Test for ground_truth:", stats.shapiro(ground_truth_lengths))
    print("Shapiro-Wilk Test for finetune_output:", stats.shapiro(finetune_output_lengths))
    print("Shapiro-Wilk Test for baseline_output:", stats.shapiro(baseline_output_lengths))

    print("Two-Sample t-Test:", stats.ttest_ind(ground_truth_lengths, finetune_output_lengths))
    print("Two-Sample t-Test:", stats.ttest_ind(ground_truth_lengths, baseline_output_lengths))