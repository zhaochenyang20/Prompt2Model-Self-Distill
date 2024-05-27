import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import gc
import json
import re
from pathlib import Path
from prompt2model.quality_evaluator import self_consistency_filter
from functools import partial
import datasets
from vllm import LLM, SamplingParams
from prompt2model.output_annotator import construct_meta_prompt
from prompt2model.prompt_parser import MockPromptSpec, TaskType
from prompt2model.utils import count_tokens_from_string
from prompt2model.utils.path import ROOT, TEST_DATA_ROOT, MODEL_PATH
import re
import numpy as np
from prompt2model.quality_evaluator import (
    ablation_list_filter,
    min_max_length_filter,
)


VICUNA = LLM(
    model=MODEL_PATH,
    # model=test_path,
    gpu_memory_utilization=0.9,
    swap_space = 16,
    tensor_parallel_size=1,  # 根据卡数改
)

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

def exact_match_score(
    GROUND_TRUTH,
    tuned_model_generated_outputs,
):
    index = 0
    for i in range(len(GROUND_TRUTH)):
        if (GROUND_TRUTH[i] == tuned_model_generated_outputs[i]):
            index += 1
    exact_match = index / len(GROUND_TRUTH)
    return exact_match

def evaluate_model(task_names, exact_match=False):
    for task_name in task_names:
        # 改了这里的名字
        ["test", "eval"]
        for test_type in ["test"]:
            test_dataset = datasets.load_from_disk(
                f"{TEST_DATA_ROOT}/prompt2model_test/testdataset/NI/{test_type}/{task_name}"
            )
            inputs_dir = Path(ROOT+"/pass_filter")
            inputs_dir.mkdir(parents=True, exist_ok=True)

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

            matches = re.findall(
                r'\[input\]="(.*?)"\s*\[output\]="(.*?)"',
                examples,
                re.DOTALL,
            )
            
            example_outputs = [match[1] for match in matches]

            def calculate_string_metrics(string_list):
                lengths = np.array([len(s) for s in string_list])
                mean_length = np.mean(lengths)
                std_dev = np.std(lengths)
                mean_plus_2std = mean_length + 2 * std_dev
                mean_minus_2std = mean_length - 2 * std_dev
                return mean_length, mean_plus_2std, mean_minus_2std

            _, mean_plus_2std, mean_minus_2std = calculate_string_metrics(
                [match[0] for match in matches]
            )
            
            length_filter = partial(
            min_max_length_filter,
            min_length=int(mean_minus_2std),
            max_length=int(mean_plus_2std),
        )

            ablation_filter = partial(ablation_list_filter, optional_list=["input", "output", "\n\n", "\\_\\_"])
        
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
                )
                example["model_output"] = example["output_col"]
                return example

            test_dataset = test_dataset.map(map_func, load_from_cache_file=False)

            test_dataset = test_dataset.filter(
                lambda x: (
                    count_tokens_from_string(x["model_input"], "vicuna") <= 3000
                    and count_tokens_from_string(x["model_output"], "vicuna") <= 500
                )
            )

            prompts = test_dataset["model_input"]
            GROUND_TRUTH = test_dataset["model_output"]
            hyperparameter_choices = {}

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
            #! 这里测试轮次比较多，是为了看结果是否稳定
            # vicuna base model MODEL_PATH
            VICUNA_outputs = VICUNA.generate(prompts, sampling_params)
            decoded_outputs = []

            for idx, _ in enumerate(VICUNA_outputs):
                outputs = [
                    output.text.strip()
                    for output in VICUNA_outputs[idx].outputs
                    if (output.text is not None and output.text != "")
                ]
                # TODO: change this line back
                passed_outputs = ablation_filter(length_filter(outputs))
                # passed_outputs = ablation_filter(outputs)
                if passed_outputs is None:
                    decoded_outputs.append("No Output")
                else:
                    decoded_outputs.append(passed_outputs[0])

            evaluate_result = (
                rouge_l_score(GROUND_TRUTH, decoded_outputs)
                if not exact_match
                else exact_match_score(GROUND_TRUTH, decoded_outputs)
            )
            print(f"{task_name} {test_type}: {evaluate_result}")
            #! 记得改名字
            evaluate_generated_content_path = inputs_dir / f"20240525_2_{test_type}_{task_name}"
            datasets.Dataset.from_dict(
                dict(
                    model_output=decoded_outputs,
                    model_input=prompts,
                    groud_truth=GROUND_TRUTH,
                )
            ).save_to_disk(evaluate_generated_content_path)
        gc.collect()

# TODO 改任务
print("generation tasks:")
task_names = ["task036","task039", "task121", "task281", "task1195", "task1345", "task1562", "task1622"]
task_names = ["task281"]

evaluate_model(task_names, exact_match=False)