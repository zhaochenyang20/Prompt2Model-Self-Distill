from zeno_client import ZenoClient, ZenoMetric
import pandas as pd
from datasets import load_from_disk
from collections import Counter
import re
import string
from functools import partial

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

def exact_match_score(
    GROUND_TRUTH,
    tuned_model_generated_outputs,
):
    labels = list(Counter(GROUND_TRUTH).keys())
    index = 0
    n = len(GROUND_TRUTH)
    for i in range(n):
        index += exact_match(find_last_occurrence(tuned_model_generated_outputs[i], labels), GROUND_TRUTH[i])
    score = index / len(GROUND_TRUTH)
    return score

def check_mutual_inclusion(row, labels):
    model_output = row['model_output']
    groud_truth = row['groud_truth']
    return exact_match(find_last_occurrence(model_output, labels), groud_truth)

client = ZenoClient("zen_fcbJnbmEJglYTu8PSRnWCIWp99yKtt1V5YsAYRo_0Ls")

tasks = ['1516', '1529', '1612', '1615', '284', '329', '346']

for task in tasks:

    project = client.create_project(
        name=f"classification Self-ICL baseline {task}",
        view="text-classification",
        metrics=[
            ZenoMetric(name="accuracy", type="mean", columns=["correct"])
        ]
    )

    test_dataset = load_from_disk(f"/home/azureuser/p2mss/p2mss/self_icl_baseline_generated_data/20240327_task{task}")
    df_test = test_dataset.to_pandas()
    df_test['id'] = range(len(df_test))
    df_test['id'] = df_test['id'].astype(str)
    project.upload_dataset(df_test, id_column="id", data_column="model_input", label_column="groud_truth")

    baseline_results = load_from_disk(f"/home/azureuser/p2mss/p2mss/self_icl_baseline_generated_data/20240327_task{task}")
    df_baseline = baseline_results.to_pandas()
    df_baseline['id'] = range(len(df_baseline))
    df_baseline['id'] = df_baseline['id'].astype(str)
    labels = list(Counter(df_baseline['groud_truth']).keys())
    check_func = partial(check_mutual_inclusion, labels=labels)
    df_baseline['correct'] = df_baseline.apply(check_func, axis=1).astype(int)
    project.upload_system(df_baseline, name="baseline", id_column="id", output_column="model_output")
    
    # finetuned_results = load_from_disk(f'/home/azureuser/p2mss/p2mss/classification_14/NI_task{task}_exp_14/best_ckpt_generated_content')
    # df_finetune = finetuned_results.to_pandas()
    # df_finetune['id'] = range(len(df_finetune))
    # df_finetune['correct'] = df_finetune.apply(check_mutual_inclusion, axis=1)
    # project.upload_system(df_finetune, name="finetune", id_column="id", output_column="model_output")  

