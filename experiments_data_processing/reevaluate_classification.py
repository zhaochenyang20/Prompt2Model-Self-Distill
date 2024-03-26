import os
import re
from datasets import load_from_disk
from collections import Counter
import string
import json

def find_last_occurrence(model_output: str, labels: list[str]) -> str:
    pattern = '|'.join(re.escape(label) for label in labels)
    regex = re.compile(pattern)
    matches = list(regex.finditer(model_output))
    return matches[-1].group() if matches else None

# cited from https://github.com/allenai/natural-instructions/blob/55a365637381ce7f3748fa2eac7aef1a113bbb82/eval/automatic/evaluation.py#L24
def normalize_answer(s):
    """Lower text and remove punctuation, and extra whitespace."""

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))

def exact_match(prediction, ground_truth, xlingual=False):
    # small changed based on our current code
    if prediction is None:
        return 0
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


root = '/home/azureuser/p2mss/p2mss/classification_14'

results = {}

for task_folder in os.listdir(root):
    # /home/azureuser/p2mss/p2mss/classification_14/
    # NI_task190_exp_14
    task_name = task_folder.split('_')[1]

    results[task_name] = {}

    for param_set in os.listdir(os.path.join(root ,task_folder)):
        # /home/azureuser/p2mss/p2mss/classification_14/NI_task190_exp_14/
        # task190_0.6_False_False_40_14
        if not param_set.startswith(task_name):
            continue
        generated_contents_path = os.path.join(root, task_folder, param_set, 'generated_contents')
        
        results[task_name][param_set] = {}

        # go through 3 epochs' scores
        for i in range(1,4):
            # load evalution results
            evaluate_outputs_path = os.path.join(generated_contents_path, str(i))
            evaluate_outputs = load_from_disk(evaluate_outputs_path)
            # calculate exact match score
            labels = list(Counter(evaluate_outputs['groud_truth']).keys())
            score = 0
            for model_output, ground_truth in zip(evaluate_outputs['model_output'], evaluate_outputs['groud_truth']):
                score += exact_match(find_last_occurrence(model_output, labels), ground_truth)
            # results 
            # 'task190': {
            #    'task190_0.6_False_False_40_14': {
            #       '1': score,
            #       '2': score,
            #       '3': score
            #    }
            results[task_name][param_set][i] = score / len(evaluate_outputs)

best_ckpt = {}
diff = []

for task, params in results.items():
    # task: task190
    # params: task190_0.6_False_False_40_14
    max_score = 0           # max accuracy score for a task
    max_params = []         # corresponding max parameters set
    max_epochs = []         # corresponding max epoch
    # set max_params and max_epochs as list in case we have same evaluation score
    for param_set, score_recording in params.items():
        # get max score and correspinding epoch for each set of params
        # when we have more than one max score, using the smallest epoch, same as the original method
        local_max_epoch, local_max_score = max(score_recording.items(), key=lambda score_recording: score_recording[1])
        if local_max_score > max_score:
            max_score = local_max_score
            # overwrite past params and epochs
            max_params = [param_set]
            max_epochs = [local_max_epoch]
        elif local_max_score == max_score:
            # the order of params doesn't matter, if we have same params ckpt, we can use this
            # but epoch number matters
            max_params.append(param_set)
            max_epochs.append(local_max_epoch)
            
    # recording max params and epochs for each task
    best_ckpt[task] = {"max_score": max_score, "max_params": max_params, "max_epochs": max_epochs}
    
    # we need to check
    # 1. if original params in current params
    # load original best ckpt's params
    original_best_validation_result_path = os.path.join(root, f"NI_{task}_exp_14", 'best_validation_result.json')
    with open(original_best_validation_result_path, 'r') as file:
        data = json.load(file)
    original_best_params = data['evaluate_result_path'].split('/')[-2]
    if original_best_params in max_params: 
        # 2. if original max evaluation score == current max evaluation score
        # 3. if original epoch == current epoch (for same score, we use smallest epoch)
        # load original params' results
        original_results_path = os.path.join(root, f"NI_{task}_exp_14", original_best_params, 'result.json')
        with open(original_results_path, 'r') as file:
            data = json.load(file)
        original_max_score = max(data.values())
        original_max_epoch = min([int(key) for key, value in data.items() if value == original_max_score])
        current_index = max_params.index(original_best_params)   
        if not (original_max_score==max_score and max_epochs[current_index] == original_max_epoch):
            diff.append(task)
        else:
            print(f"{task}")
            print(f"same param: {original_best_params}={max_params[current_index]}")
            print(f"same epoch: {original_max_epoch}={max_epochs[current_index]}")
            print(f"same score: {original_max_score}={max_score}")
    else:
        diff.append(task)
