import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import gc
import json
from functools import partial
from pathlib import Path

import datasets
import ray
import torch
from IPython import embed
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

from prompt2model.output_annotator.prompt_template import construct_meta_prompt
from prompt2model.prompt_parser import MockPromptSpec, TaskType
from prompt2model.utils import count_tokens_from_string

# import os


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


def evaluate_model(task_names):
    for task_name in task_names:
        experiment_name = "NI_" + task_name + "_exp_5"
        for test_type in ["test", "eval"]:
            test_dataset = datasets.load_from_disk(
                f"/home/cyzhao/prompt2model_test/testdataset/NI/{test_type}/{task_name}"
            )
            inputs_dir = Path("/home/cyzhao/")

            file_path = "/home/cyzhao/main/NI_tasks/tasks.json"

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

            construct_prompt = partial(
                construct_meta_prompt,
                instruction=prompt_spec.instruction,
                examples=prompt_spec.examples,
            )

            tokenizer = AutoTokenizer.from_pretrained(
                "/data/ckpts/huggingface/models/models--lmsys--vicuna-7b-v1.5/snapshots/de56c35b1763eaae20f4d60efd64af0a9091ebe5",
                local_files_only=True,
                padding_side="left",
                trust_remote_code=True,
            )

            def map_func(example):
                example["model_input"] = construct_prompt(
                    new_input=example["input_col"]
                )
                example["model_output"] = example["output_col"]
                example["text"] = (
                    example["model_input"]
                    + example["model_output"]
                    + tokenizer.eos_token
                )
                return example

            test_dataset = test_dataset.map(map_func, load_from_cache_file=False)

            test_dataset = test_dataset.filter(
                lambda x: (
                    count_tokens_from_string(x["model_input"]) <= 3200
                    and count_tokens_from_string(x["model_output"]) <= 500
                )
            )

            prompts = test_dataset["model_input"]
            GROUND_TRUTH = test_dataset["model_output"]
            hyperparameter_choices = {}
            sampling_params = SamplingParams(
                top_k=hyperparameter_choices.get("top_k", -1),
                top_p=hyperparameter_choices.get("top_p", 1),
                temperature=hyperparameter_choices.get("temperature", 0),
                max_tokens=hyperparameter_choices.get("max_tokens", 500),
            )
            MODEL_INPUTS = prompts
            VALIDATION_DATASET = datasets.Dataset.from_dict(
                {"model_ouput": GROUND_TRUTH, "model_input": MODEL_INPUTS}
            )

            #! 这里测试轮次比较多，是为了看结果是否稳定
            # vicuna base model "/data/ckpts/huggingface/models/models--lmsys--vicuna-7b-v1.5/snapshots/de56c35b1763eaae20f4d60efd64af0a9091ebe5"
            base_model = "/data/ckpts/huggingface/models/models--lmsys--vicuna-7b-v1.5/snapshots/de56c35b1763eaae20f4d60efd64af0a9091ebe5"
            # 改了这里的名字
            path = f"/data2/cyzhao/best_ckpt/{experiment_name}"
            ray.init(ignore_reinit_error=True)
            tuned_vicuna = LLM(
                model=base_model,
                gpu_memory_utilization=0.9,
                tensor_parallel_size=len(
                    os.environ["CUDA_VISIBLE_DEVICES"].split(",")
                ),  # 根据卡数改
            )
            tuned_vicuna_outputs = tuned_vicuna.generate(prompts, sampling_params)
            tuned_vicuna_generated_outputs = [
                each.outputs[0].text for each in tuned_vicuna_outputs
            ]
            rouge_socre = rouge_l_score(GROUND_TRUTH, tuned_vicuna_generated_outputs)
            print(f"{task_name} {test_type}: {rouge_socre}")
            with open(inputs_dir / f"evaluate_10_times.txt", "a+") as file:
                file.write(
                    f"\n\nresult of {path} th:\n\n------------------------------------------------{rouge_socre}------------------------------------------------\n\n"
                )
            del tuned_vicuna
            #! 记得改名字
            evaluate_generated_content_path = inputs_dir / f"base_vicuna_{task_name}"
            # print(f"Genrated contents are stored in {str(evaluate_generated_content_path)}")
            print(
                f"length of tuned_vicuna_generated_outputs: {len(tuned_vicuna_generated_outputs)}"
            )
            print(f"length of prompts: {len(prompts)}")
            print(f"length of groud_truth: {len(GROUND_TRUTH)}")
            datasets.Dataset.from_dict(
                dict(
                    model_output=tuned_vicuna_generated_outputs,
                    model_input=prompts,
                    groud_truth=GROUND_TRUTH,
                )
            ).save_to_disk(evaluate_generated_content_path)
            gc.collect()
            torch.cuda.empty_cache()
            ray.shutdown()
            destroy_model_parallel()


task_names = ["task121"]
evaluate_model(task_names)
