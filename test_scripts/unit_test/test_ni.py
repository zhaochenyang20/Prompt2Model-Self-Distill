import os
import torch
print(torch.cuda.is_available())
torch._C._cuda_init()

# TODO 改卡
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import gc
import json
import re
from pathlib import Path
from prompt2model.quality_evaluator import self_consistency_filter
from functools import partial
import datasets
import torch
from IPython import embed
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
from prompt2model.prompt_parser import MockPromptSpec, TaskType
from prompt2model.utils import count_tokens_from_string
from prompt2model.utils.prompt import PROMPT_TEMPLATE

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
        task_instruction=instruction,
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
        base_model = "/data/datasets/models/huggingface/lmsys/vicuna-7b-v1.5"
        # 改了这里的名字
        experiment_name = "NI_" + task_name + "_exp_-1"
        path = f"/data/tir/projects/tir5/users/xjia2/best_ckpt/{experiment_name}"
        tuned_vicuna = LLM(
            model=base_model if not finetuned else path,
            gpu_memory_utilization=0.70,
            swap_space = 16, 
            tensor_parallel_size=1,  # 根据卡数改
        )
        for test_type in ["test", "eval"]:
            test_dataset = datasets.load_from_disk(
                f"/home/xjia2/p2mss/prompt2model_test/testdataset/NI/{test_type}/{task_name}"
            )
            inputs_dir = Path("/home/xjia2/p2mss/baseline_generated_data")

            file_path = "/home/xjia2/p2mss/main/NI_tasks/tasks.json"

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
                top_k=hyperparameter_choices.get("top_k", 10),
                temperature=hyperparameter_choices.get("temperature", 0.2),
                max_tokens=hyperparameter_choices.get("max_tokens", 500),
            )
            consistency_filter = partial(
                self_consistency_filter,
                min_frequency=hyperparameter_choices.get("min_frequency", 0.2),
            )
            #! 这里测试轮次比较多，是为了看结果是否稳定
            # vicuna base model "/data/datasets/models/huggingface/lmsys/vicuna-7b-v1.5"
            tuned_vicuna_outputs = tuned_vicuna.generate(prompts, sampling_params)
            
            decoded_outputs = []
            
            for idx, _ in enumerate(tuned_vicuna_outputs):
                outputs = [
                    output.text.strip()
                    for output in tuned_vicuna_outputs[idx].outputs
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
            with open(inputs_dir / f"evaluate_10_times.txt", "a+") as file:
                file.write(
                    f"\n\nresult of {path} th:\n\n------------------------------------------------{evaluate_result}------------------------------------------------\n\n"
                )
            #! 记得改名字
            evaluate_generated_content_path = inputs_dir / f"base_{test_type}_{task_name}"
            datasets.Dataset.from_dict(
                dict(
                    model_output=decoded_outputs,
                    model_input=prompts,
                    groud_truth=GROUND_TRUTH,
                )
            ).save_to_disk(evaluate_generated_content_path)
        del tuned_vicuna
        gc.collect()
        torch.cuda.empty_cache()
        destroy_model_parallel()


# TODO 改任务
# print("generation tasks:")
# task_names = ["task039", "task281", "task121", "task1195", "task034", "task1622", "task1562", "task671", "task1345", "task035", "task1659", "task569", "task1631", "task1557", "task036"]
# task_names = ["task281", "task121", "task1195", "task034", "task1622", "task1562", "task671", "task1345", "task035", "task1659", "task569", "task1631", "task1557", "task036"]
# task_names = ["task034", "task1622", "task1562", "task671", "task1345", "task035", "task1659", "task569", "task1631", "task1557", "task036"]
# evaluate_model(task_names, finetuned=False, exact_match=False)
print("classification tasks:")
task_names = ["task202", "task199", "task1388", "task201", "task190", "task1386", "task1554", "task738", "task1385", "task1529", "task200", "task1612", "task937", "task1516", "task1615"]
task_names = ["task201", "task190", "task1386", "task1554", "task738", "task1385", "task1529", "task200", "task1612", "task937", "task1516", "task1615"]
evaluate_model(task_names, finetuned=False, exact_match=True)
