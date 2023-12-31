import os

# TODO 改卡
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import gc
import json
import re
from pathlib import Path

import datasets
import torch
from IPython import embed
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
from prompt2model.prompt_parser import MockPromptSpec, TaskType
from prompt2model.utils import count_tokens_from_string

# import os

PROMPT_TEMPLATE = """
A chat between a curious user and an artificial intelligence assistant.
The assistant gives concise answers to the user's questions.
USER: The artificial intelligence assistant only needs to help annotate label. The task is: {instruction} 
ASSISTANT: Okay. 
{examples}
USER: [input] = {new_input}
ASSISTANT: 
"""  # noqa E501

def construct_meta_prompt(
    instruction: str = None,
    examples: str = None,
    new_input: str = None,
) -> str:
    """Constructs a prompt template for the dataset generator.

    Args:
        instruction: The natural language instruction for the prompt.
        input: A new input to be annotated.
        high_quality_input_string: A string representing the high quality examples.
    """
    return PROMPT_TEMPLATE.format(
        instruction=instruction,
        new_input=new_input,
        examples=examples,
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


def exact_match_score(GROUND_TRUTH, tuned_model_generated_outputs):
    index = 0
    for i in range(len(GROUND_TRUTH)):
        if (
            GROUND_TRUTH[i] in tuned_model_generated_outputs[i]
            or tuned_model_generated_outputs[i] in GROUND_TRUTH[i]
        ):
            index += 1
    exact_match = index / len(GROUND_TRUTH)
    return exact_match


def evaluate_model(task_names, finetuned=False, exact_match=False):
    for task_name in task_names:
        experiment_name = "NI_" + task_name + "_exp_1"
        base_model = "/data/ckpts/huggingface/models/models--lmsys--vicuna-7b-v1.5/snapshots/de56c35b1763eaae20f4d60efd64af0a9091ebe5"
        # 改了这里的名字
        path = f"/data2/cyzhao/best_ckpt/{experiment_name}"
        tuned_vicuna = LLM(
            model=base_model if not finetuned else path,
            gpu_memory_utilization=0.9,
            tensor_parallel_size=1,  # 根据卡数改
        )
        for test_type in ["test", "eval"]:
            test_dataset = datasets.load_from_disk(
                f"/home/cyzhao/prompt2model_test/testdataset/NI/{test_type}/{task_name}"
            )
            inputs_dir = Path("/home/cyzhao/baseline_generated_data")

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

            tokenizer = AutoTokenizer.from_pretrained(
                "/data/ckpts/huggingface/models/models--lmsys--vicuna-7b-v1.5/snapshots/de56c35b1763eaae20f4d60efd64af0a9091ebe5",
                local_files_only=True,
                padding_side="left",
                trust_remote_code=True,
            )
            matches = re.findall(
                r'\[input\]="(.*?)"\s*\[output\]="(.*?)"',
                prompt_spec.examples,
                re.DOTALL,
            )
            assert matches != []
            annotation_prompt_string = ""
            for input, output in matches:
                annotation_prompt_string += f"USER: [input] = {input}\n"
                annotation_prompt_string += f"ASSISTANT: {output}\n"
            assert annotation_prompt_string != ""
            def map_func(example):
                example["model_input"] = construct_meta_prompt(
                    instruction=prompt_spec.instruction,
                    examples=annotation_prompt_string.strip(),
                    new_input=example["input_col"],
                ).strip()
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
                    count_tokens_from_string(x["model_input"], "vicuna") <= 3000
                    and count_tokens_from_string(x["model_output"], "vicuna") <= 500
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

            #! 这里测试轮次比较多，是为了看结果是否稳定
            # vicuna base model "/data/ckpts/huggingface/models/models--lmsys--vicuna-7b-v1.5/snapshots/de56c35b1763eaae20f4d60efd64af0a9091ebe5"
            tuned_vicuna_outputs = tuned_vicuna.generate(prompts, sampling_params)
            tuned_vicuna_generated_outputs = [
                each.outputs[0].text for each in tuned_vicuna_outputs
            ]
            evaluate_result = (
                rouge_l_score(GROUND_TRUTH, tuned_vicuna_generated_outputs)
                if not exact_match
                else exact_match_score(GROUND_TRUTH, tuned_vicuna_generated_outputs)
            )
            print(f"{task_name} {test_type}: {evaluate_result}")
            with open(inputs_dir / f"evaluate_10_times.txt", "a+") as file:
                file.write(
                    f"\n\nresult of {path} th:\n\n------------------------------------------------{evaluate_result}------------------------------------------------\n\n"
                )
            #! 记得改名字
            evaluate_generated_content_path = inputs_dir / f"base_{task_name}"
            datasets.Dataset.from_dict(
                dict(
                    model_output=tuned_vicuna_generated_outputs,
                    model_input=prompts,
                    groud_truth=GROUND_TRUTH,
                )
            ).save_to_disk(evaluate_generated_content_path)
        del tuned_vicuna
        gc.collect()
        torch.cuda.empty_cache()
        destroy_model_parallel()


# TODO 改任务
task_names = ["task200"]
evaluate_model(task_names, finetuned=False, exact_match=True)
