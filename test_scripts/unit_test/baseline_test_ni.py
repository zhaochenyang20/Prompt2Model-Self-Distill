import os

# TODO
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gc
import json
import torch
from pathlib import Path
from prompt2model.quality_evaluator import self_consistency_filter
from functools import partial
import datasets
from datasets import load_from_disk
from vllm import LLM, SamplingParams
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
from prompt2model.output_annotator import construct_meta_prompt
from prompt2model.prompt_parser import MockPromptSpec, TaskType
from prompt2model.utils import count_tokens_from_string
from prompt2model.utils.path import ROOT, TEST_DATA_ROOT, MODEL_PATH
import string
import re
from collections import Counter

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

def lcs_length_dp(x, y):
    """Compute the length of the longest common subsequence between two strings using dynamic programming."""
    m, n = len(x), len(y)
    dp_table = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                dp_table[i][j] = 0
            elif x[i - 1] == y[j - 1]:
                dp_table[i][j] = dp_table[i - 1][j - 1] + 1
            else:
                dp_table[i][j] = max(dp_table[i - 1][j], dp_table[i][j - 1])

    return dp_table[m][n]

def rouge_l_score(GROUND_TRUTH, tuned_model_generated_outputs):
    scores = []
    for gt, gen in zip(GROUND_TRUTH, tuned_model_generated_outputs):
        lcs = lcs_length_dp(gt, gen)
        if lcs == 0:
            scores.append(0)
            continue
        precision = lcs / len(gen)
        recall = lcs / len(gt)
        f_measure = (2 * precision * recall) / (precision + recall)
        scores.append(f_measure)
    return sum(scores) / len(scores)

def evaluate_model(task_names, classification=False):  
    output_dir = Path(ROOT + "/baseline_without_filter_rerun_test_results_0526")
    
    model = LLM(
        model=MODEL_PATH,
        gpu_memory_utilization=0.9,
        swap_space = 16,
        tensor_parallel_size=1,
    )

    for task_name in task_names:


        suffix = '\n'

        # change notion
        # notions = [':', ' ']
        notions = ['=']
        for notion in notions:
            
            # double check test type: "test" or "eval"
            for test_type in ["test"]:
                test_dataset = datasets.load_from_disk(
                    f"{TEST_DATA_ROOT}/prompt2model_test/testdataset/NI/{test_type}/{task_name}"
                )
                file_path = ROOT+"/main/NI_tasks/tasks.json"

                with open(file_path, "r", encoding="utf-8") as json_file:
                    data = json.load(json_file)

                index = 0
                for task in data:
                    if task["task_name"] == task_name:
                        break
                    index += 1
                task_instruction = data[index]["task_instruction"]
                task_name = data[index]["task_name"]
                examples = data[index]["examples"]

                prompt_spec = MockPromptSpec(
                    task_type=TaskType.TEXT_GENERATION,
                    instruction=task_instruction,
                    examples=examples,
                )

                def map_func(example):
                    example["model_input"] = construct_meta_prompt(
                        instruction=prompt_spec.instruction,
                        examples=prompt_spec.examples,
                        new_input=example["input_col"],
                        is_generation=False,
                        few_shots_prompt="",
                        notion=notion,
                        suffix=suffix
                    )
                    example["model_output"] = example["output_col"]
                    return example

                test_dataset = test_dataset.map(map_func, load_from_cache_file=False)

                test_dataset = test_dataset.filter(
                    lambda x: (
                        count_tokens_from_string(x["model_input"], "vicuna") <= 4096
                        # and count_tokens_from_string(x["model_output"], "vicuna") <= 4096
                    )
                )

                prompts = test_dataset["model_input"]
                GROUND_TRUTH = test_dataset["model_output"]
                hyperparameter_choices = {}

                # change this to the same params as inference
                # sampling_params = SamplingParams(
                #     top_k=hyperparameter_choices.get("top_k", -1),
                #     top_p=hyperparameter_choices.get("top_p", 1),
                #     temperature=hyperparameter_choices.get("temperature", 0),
                #     max_tokens=hyperparameter_choices.get("max_tokens", 500),
                # )

                sampling_params = SamplingParams(
                    n=hyperparameter_choices.get("n", 10),
                    best_of=hyperparameter_choices.get("best_of", 20),
                    top_k=hyperparameter_choices.get("top_k", 40),
                    temperature=hyperparameter_choices.get("temperature", 1.0),
                    max_tokens=hyperparameter_choices.get("max_tokens", 500),
                )


                consistency_filter = partial(
                    self_consistency_filter,
                    min_frequency=hyperparameter_choices.get("min_frequency", 0.2),
                )
                model_outputs = model.generate(prompts, sampling_params)
                print(model_outputs[0:3])
                decoded_outputs = []


                for idx, _ in enumerate(model_outputs):
                    outputs = [
                        output.text.strip()
                        for output in model_outputs[idx].outputs
                        if (output.text is not None and output.text != "")
                    ]
                    consistent_output = consistency_filter(outputs)
                    if (
                        consistent_output is not None
                        and consistent_output != ""
                        and isinstance(consistent_output, str)
                    ):
                        decoded_outputs.append(consistent_output)
                    else:
                        decoded_outputs.append("No Output")

                evaluate_result = (
                    rouge_l_score(GROUND_TRUTH, decoded_outputs)
                    if not classification
                    else exact_match_score(GROUND_TRUTH, decoded_outputs)
                )
                
                print(f"{task_name} {test_type} => {evaluate_result}")
                #TODO change file name every day
                evaluate_generated_content_path = output_dir / f"20240526_{task_name}" # 0327用的是只sample一个的，0526用的是sample10个的参数
                datasets.Dataset.from_dict(
                    dict(
                        model_output=decoded_outputs,
                        model_input=prompts,
                        groud_truth=GROUND_TRUTH,
                    )
                ).save_to_disk(evaluate_generated_content_path)
            
            gc.collect()
        
        # del model
        torch.cuda.empty_cache()
    destroy_model_parallel()

# TODO change task
# TODO determine baseline or finetuned model
# TODO deterine generation or classification

# classification tasks
# classification_tasks = ["task1516", "task1529", "task1612", "task1615", "task284", "task329", "task346"]
# evaluate_model(classification_tasks, finetuned=False, classification=True)          
# generation tasks:

if __name__ == "__main__":
    generation_tasks = ["task281", "task1345", "task1562", "task1622"]
    evaluate_model(generation_tasks, classification=False)

