"""Test Input Generator."""
import os

# TODO 改卡
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import json
from pathlib import Path

from prompt2model.input_generator import VLLMPromptBasedInputGenerator
from prompt2model.prompt_parser import MockPromptSpec, TaskType

# 测试
input_generator = VLLMPromptBasedInputGenerator(gpu_memory_utilization=0.9)


# TODO change task name "201" "202" 有问题
# task_names = ["738", "1554", "1386", "020", "1516", "935", "1612", "937", "200", "1388", "1529", "199", "1344", "1385"]
task_names = ["738"]

for task_name in task_names:
    print(f"\n\nin task_{task_name}")
    file_path = "/home/cyzhao/main/NI_tasks/tasks.json"
    with open(file_path, "r", encoding="utf-8") as json_file:
        all_tasks = json.load(json_file)
    choosen_task = None
    for task in all_tasks:
        if task["task_name"] == "task" + task_name:
            task_tuple = (
                task["task_name"],
                task["task_instruction"],
                task["examples"],
                task["expected_content"],
                f"/home/cyzhao/prompt2model_test/testdataset/NI/eval/task{task_name}",
                f"/home/cyzhao/prompt2model_test/testdataset/NI/test/task{task_name}",
                task.get("optional_list", []),
                task.get("metric", "exact_match"),
                task.get("labels", [])
            )
            choosen_task = task
            break

    prompt_spec = MockPromptSpec(
        task_type=TaskType.TEXT_GENERATION,
        instruction=choosen_task["task_instruction"],  # # noqa E501
        examples=choosen_task["examples"],  # noqa E501
    )

    inputs = input_generator.batch_generation_inputs(
        prompt_spec,
        5,
        10,
        dict(
            top_k=40,
            temperature=0.6,
            min_input_length=50,
        ),
        expected_content=choosen_task["expected_content"],
        optional_list=choosen_task["optional_list"],
        conditional_labels=choosen_task["labels"],
    )
