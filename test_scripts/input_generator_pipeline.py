"""Test Input Generator."""

from pathlib import Path
import json

import datasets

from prompt2model.input_generator import VLLMPromptBasedInputGenerator
from prompt2model.prompt_parser import MockPromptSpec, TaskType

# 测试

inputs_dir = Path("/home/cyzhao/generated_datasets")
inputs_dir.mkdir(parents=True, exist_ok=True)

generated_inputs = []

# TODO change task name
task_name = '121'
file_path = '/home/cyzhao/main/NI_tasks/tasks.json'  
with open(file_path, 'r', encoding='utf-8') as json_file:
    all_tasks = json.load(json_file)
choosen_task = None
for task in all_tasks:
    if task['task_name'] == 'task'+task_name:
        task_tuple = (
            task['task_name'],
            task['task_instruction'],
            task['examples'],
            task['expected_content'],
            f"/home/cyzhao/prompt2model_test/testdataset/NI/eval/task{task_name}",
            f"/home/cyzhao/prompt2model_test/testdataset/NI/test/task{task_name}",
            task['optional_list']
        )
        choosen_task = task
        break

prompt_spec = MockPromptSpec(
    task_type=TaskType.TEXT_GENERATION,
    instruction=choosen_task['task_instruction'],  # # noqa E501
    examples=choosen_task['examples'],  # noqa E501
)

input_generator = VLLMPromptBasedInputGenerator(gpu_memory_utilization=0.9)

# prompt = input_generator.construct_generation_prompt(
#     instruction=prompt_spec.instruction,
#     few_shot_example_string=prompt_spec.examples,
#     generated_inputs=["How are you?", "I am fine."],
#     context_cutoff=3500,
# )

# filter_prompt = input_generator.construct_filter_prompt(
#     few_shot_example_string=prompt_spec.examples, new_input="how are you?"
# )
# generation_epochs,generation_batch_size,generation_top_k,generation_temperature,min_frequency,min_input_length,training_epochs
# 10                15                      50              0.3                     0.5         125                5
inputs = input_generator.batch_generation_inputs(
    prompt_spec,
    10,
    10,
    dict(
        top_k=40,
        temperature=0.6,
        min_input_length=100,
    ),
    expected_content=choosen_task['expected_content'],
    optional_list=choosen_task['optional_list']
)
