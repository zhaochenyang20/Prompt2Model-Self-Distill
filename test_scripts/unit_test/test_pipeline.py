import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

# TODO change card
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gc
import json
from pathlib import Path
from functools import partial
import datasets
from vllm import LLM, SamplingParams
from prompt2model.output_annotator import construct_meta_prompt
from prompt2model.prompt_parser import MockPromptSpec, TaskType
from prompt2model.utils.path import STORE_ROOT, ROOT, TEST_DATA_ROOT, MODEL_PATH
from prompt2model.utils import count_tokens_from_string
from datasets import load_from_disk
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

ckpt_dic = {
    "task281":"/home/azureuser/p2mss/p2mss/ckpt_data_p2ms/task281_1.0_False_False_20_33/checkpoint-594", 
    "task1345":"/home/azureuser/p2mss/p2mss/ckpt_data_p2ms/task1345_1.0_False_False_20_33/checkpoint-576", 
    "task1562":"/home/azureuser/p2mss/p2mss/ckpt_data_p2ms/task1562_1.0_False_False_20_33/checkpoint-573", 
    "task1622":"/home/azureuser/p2mss/p2mss/ckpt_data_p2ms/task1622_1.0_False_False_20_33/checkpoint-597",
    "task284":"/home/azureuser/p2mss/p2mss/ckpt_data_p2ms/task284_0.7_False_False_40_33/checkpoint-1170", 
    "task1516":"/home/azureuser/p2mss/p2mss/ckpt_data_p2ms/task1516_0.7_False_False_40_33/checkpoint-1068", 
    "task1529":"/home/azureuser/p2mss/p2mss/ckpt_data_p2ms/task1529_0.7_False_False_40_33/checkpoint-1080", 
    "task1612":"/home/azureuser/p2mss/p2mss/ckpt_data_p2ms/task1612_0.7_False_False_40_33/checkpoint-1149", 
    "task1615":"/home/azureuser/p2mss/p2mss/ckpt_data_p2ms/task1615_0.7_False_False_40_33/checkpoint-1110", 
    "task329":"/home/azureuser/p2mss/p2mss/ckpt_data_p2ms/task329_0.7_False_False_40_33/checkpoint-1095", 
    "task346":"/home/azureuser/p2mss/p2mss/ckpt_data_p2ms/task346_0.7_False_False_40_33/checkpoint-1053"

}


def evaluate_model(task_names, exact_match=False):

    test_type = 'test'
    for task_name in task_names:
        # TODO
        model_path = ckpt_dic[task_name]
        model = LLM(
            model=model_path,
            tokenizer='lmsys/vicuna-7b-v1.5',
            gpu_memory_utilization=0.9,
            swap_space = 16,
            tensor_parallel_size=1,
        )
        
        test_dataset = datasets.load_from_disk(
            f"{TEST_DATA_ROOT}/prompt2model_test/testdataset/NI/{test_type}/{task_name}"
        )
        print(type(test_dataset))
        test_dataset = test_dataset.select(range(20))
        print(type(test_dataset))


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
                few_shots_prompt = ""
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
        print(prompts[0])
        GROUND_TRUTH = test_dataset["model_output"]

        print(f'len prompts = {len(prompts)}, len ground truth = {len(GROUND_TRUTH)}')
        hyperparameter_choices = {}

        # change this to the same params as inference
        sampling_params = SamplingParams(
            n=hyperparameter_choices.get("n", 1),
            top_k=hyperparameter_choices.get("top_k", -1),
            top_p=hyperparameter_choices.get("top_p", 1),
            temperature=hyperparameter_choices.get("temperature", 0),
            max_tokens=hyperparameter_choices.get("max_tokens", 500),
        )

        model_outputs = model.generate(
            prompts = prompts, 
            sampling_params = sampling_params,
        )
        print(f"len(model_outputs): {len(model_outputs)}")
        print(model_outputs)
        # token
        # 20122
        # print
        decoded_outputs = []

        for idx, _ in enumerate(model_outputs):
            outputs = [
                output.text.strip()
                for output in model_outputs[idx].outputs
                if (output.text is not None and output.text != "")
            ]
            if len(outputs) >= 1:
                decoded_outputs.append(outputs[0])
            else:
                print('in else')
                print(outputs)
                print(model_outputs[idx].outputs)
                decoded_outputs.append("")


        print(len(decoded_outputs), len(prompts), len(GROUND_TRUTH))
        evaluate_result = (
            rouge_l_score(GROUND_TRUTH, decoded_outputs)
            if not exact_match
            else exact_match_score(GROUND_TRUTH, decoded_outputs)
        )
        print(f"{task_name} {test_type}: {evaluate_result}")
        # TODO change the name
        inputs_dir = Path(ROOT+"/generated_results_finetuning_without_filter")
        evaluate_generated_content_path = inputs_dir / f"20240501_{test_type}_{task_name}_debug"
        print(len(decoded_outputs), len(prompts), len(GROUND_TRUTH))
        print(decoded_outputs)
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

# task_names = ["task1562"] #"task1345", "task281", "task1562", 
# evaluate_model(task_names, exact_match=False)

# print("classification tasks:")
# task_names = ["task346", "task190", "task199", "task1612", "task200", "task738", "task937", 
#               "task1385", "task1386", "task1516", "task1529", "task1615", "task284", "task329"][0::2]
# generation_tasks = [ "task1622"]
# evaluate_model(generation_tasks, exact_match=False)
classification_tasks = [ "task1615"] #"task284", "task1516","task1529", "task1612", "task1615", "task329", 
# evaluate_model(classification_tasks, exact_match=True)

tokenizer = AutoTokenizer.from_pretrained(
    'lmsys/vicuna-7b-v1.5',
    trust_remote_code=True
)
s = '"sentence_A:"'.strip()
token_ids = tokenizer.encode(s, add_special_tokens=False)
print(token_ids)
print(tokenizer.decode(token_ids))
s = '"sentence\\_A:"'.strip()
token_ids = tokenizer.encode(s, add_special_tokens=False)
print(token_ids)
print(tokenizer.decode(token_ids))
# s = "B\\_entails\\_A ".strip()
# token_ids = tokenizer.encode(s, add_special_tokens=False)
# print(tokenizer.decode(token_ids))