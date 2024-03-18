import os

# TODO 改卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gc
import json
import re
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
        if (GROUND_TRUTH[i] in tuned_model_generated_outputs[i]) or (tuned_model_generated_outputs[i] in GROUND_TRUTH[i]):
            if tuned_model_generated_outputs[i] != "":
                index += 1
    exact_match = index / len(GROUND_TRUTH)
    return exact_match

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
        # TODO: 改长度
        n = len(inputs) // 22
        for i in range(n):
            few_shots_prompt += 'USER: [input] = ' + inputs[i] + '\n'
            few_shots_prompt += 'ASSISTANT: ' + outputs[i] + '\n'

        
        # 改了这里的名字 ["test", "eval"]
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

            sampling_params = SamplingParams(
                n=hyperparameter_choices.get("n", 10),
                best_of=hyperparameter_choices.get("best_of", 20),
                top_k=hyperparameter_choices.get("top_k", 10),
                temperature=hyperparameter_choices.get("temperature", 0.2),
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
                # prompt_token_ids=[[0] * prompt_len for _ in range(num_prompts)]
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
            # TODO 记得改名字
            evaluate_generated_content_path = inputs_dir / f"20240318_{test_type}_{task_name}"
            datasets.Dataset.from_dict(
                dict(
                    model_output=decoded_outputs,
                    model_input=prompts,
                    groud_truth=GROUND_TRUTH,
                )
            ).save_to_disk(evaluate_generated_content_path)
        gc.collect()

# TODO 改任务
# print("generation tasks:")
# task_names = ["task036","task039", "task121", "task281", "task1195", "task1345", "task1562", "task1622"]
# evaluate_model(task_names, finetuned=False, exact_match=False)
print("classification tasks:")
task_names = ["task346", "task190", "task199", "task1612", "task200", "task738", "task937", 
              "task1385", "task1386", "task1516", "task1529", "task1615", "task284", "task329"][0::2]
evaluate_model(["task281"], finetuned=False, exact_match=False)
