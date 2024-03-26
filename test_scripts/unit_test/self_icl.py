import os

# TODO change card
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gc
import json
from pathlib import Path
from prompt2model.quality_evaluator import self_consistency_filter
from functools import partial
import datasets
from IPython import embed
from vllm import LLM, SamplingParams
from prompt2model.output_annotator import construct_meta_prompt
from prompt2model.prompt_parser import MockPromptSpec, TaskType
from prompt2model.utils import count_tokens_from_string
from prompt2model.utils.path import STORE_ROOT, ROOT, TEST_DATA_ROOT, MODEL_PATH
from datasets import load_from_disk
import string
import re
from collections import Counter

VICUNA = LLM(
    model=MODEL_PATH,
    gpu_memory_utilization=0.9,
    swap_space = 16,
    tensor_parallel_size=1,  # 根据卡数改
)

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

def evaluate_model(task_names, finetuned=False, exact_match=False):

    for task_name in task_names:
        few_shots_prompt = ""
        result_path = f"/home/azureuser/p2mss/p2mss/classification_14/NI_{task_name}_exp_14/best_validation_result.json"
        if not exact_match:
            result_path = f"/home/azureuser/p2mss/p2mss/generation_11/NI_{task_name}_exp_11/best_validation_result.json"
        with open(result_path, 'r') as file:
            data = json.load(file)
            evaluate_result_path = data.get("evaluate_result_path", "")
            dataset_path = '/'.join(evaluate_result_path.split('/')[:-1]) + '/dataset'
        dataset = load_from_disk(dataset_path)
        inputs = dataset['input_col']
        outputs = dataset['output_col']
        # TODO: change the number of examples in prompts
        n = len(inputs) // 3
        for i in range(n):
            few_shots_prompt += 'USER: [input] = ' + inputs[i] + '\n'
            few_shots_prompt += 'ASSISTANT: ' + outputs[i] + '\n'

        # TODO 改了这里的名字 ["test", "eval"]
        for test_type in ["test"]:
            test_dataset = datasets.load_from_disk(
                f"{TEST_DATA_ROOT}/prompt2model_test/testdataset/NI/{test_type}/{task_name}"
            )
            inputs_dir = Path(ROOT+"/baseline_generated_data")

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
                print(example)
                example["model_input"] = construct_meta_prompt(
                    instruction=prompt_spec.instruction,
                    examples=prompt_spec.examples,
                    new_input=example["input_col"],
                    is_generation=False,
                    few_shots_prompt = few_shots_prompt
                )
                example["model_output"] = example["output_col"]
                return example

            test_dataset = test_dataset.map(map_func, load_from_cache_file=False)

            # test_dataset = test_dataset.filter(
            #     lambda x: (
            #         count_tokens_from_string(x["model_input"], "vicuna") <= 3000
            #         and count_tokens_from_string(x["model_output"], "vicuna") <= 500
            #     )
            # )

            prompts = test_dataset["model_input"]
            num_prompts = len(prompts)
            prompt_len = max(len(s) for s in prompts)
            GROUND_TRUTH = test_dataset["model_output"]
            hyperparameter_choices = {}

            # change this to the same params as inference
            sampling_params = SamplingParams(
                top_k=hyperparameter_choices.get("top_k", -1),
                top_p=hyperparameter_choices.get("top_p", 1),
                temperature=hyperparameter_choices.get("temperature", 0),
                max_tokens=hyperparameter_choices.get("max_tokens", 500),
            )

            consistency_filter = partial(
                self_consistency_filter,
                min_frequency=hyperparameter_choices.get("min_frequency", 0.2),
            )
            #! 这里测试轮次比较多，是为了看结果是否稳定
            # vicuna base model MODEL_PATH
            print(f'!!!!!prompt length is: {prompt_len}')
            print(f'!!!!!num_prompts is: {num_prompts}')
            VICUNA_outputs = VICUNA.generate(
                prompts = prompts, 
                sampling_params = sampling_params,
            )
            decoded_outputs = []

            for idx, _ in enumerate(VICUNA_outputs):
                outputs = [
                    output.text.strip()
                    for output in VICUNA_outputs[idx].outputs
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
                if not exact_match
                else exact_match_score(GROUND_TRUTH, decoded_outputs)
            )
            print(f"{task_name} {test_type}: {evaluate_result}")
            # TODO change the name
            evaluate_generated_content_path = inputs_dir / f"20240325_self_icl_{test_type}_{task_name}"
            datasets.Dataset.from_dict(
                dict(
                    model_output=decoded_outputs,
                    model_input=prompts,
                    groud_truth=GROUND_TRUTH,
                )
            ).save_to_disk(evaluate_generated_content_path)
        gc.collect()

# TODO change tasks

# print("generation tasks:")
# task_names = ["task036","task039", "task121", "task281", "task1195", "task1345", "task1562", "task1622"]
# evaluate_model(task_names, finetuned=False, exact_match=False)

# print("classification tasks:")
# task_names = ["task346", "task190", "task199", "task1612", "task200", "task738", "task937", 
#               "task1385", "task1386", "task1516", "task1529", "task1615", "task284", "task329"][0::2]
evaluate_model(["task200"], finetuned=False, exact_match=True)
