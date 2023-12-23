"""Test Input Generator."""

import os
from pathlib import Path
from datasets import Dataset, load_from_disk
from prompt2model.output_annotator import VLLMPromptBasedOutputAnnotator
from prompt2model.prompt_parser import MockPromptSpec, TaskType

config =         {
        "task_instruction": "In this task, you are given two statements. The task is to output whether a given textual premise, i.e. Statement 2, entails or implies a given scientific fact, i.e. Statement 1. The output should be 'entails' if Statement 2 supports Statement 1 and should be 'neutral' otherwise.",
        "task_name": "task1554",
        "examples": "[input]=\"Sentence 1: The sum of all chemical reactions that take place within an organism is known as metabolism. Sentence 2: Metabolism is the sum total of all chemical reactions performed by an organism.\"\n[output]=\"entails\"\n\n[input]=\"Sentence 1: The endocrine system produces most of the hormones that regulate body functions. Sentence 2: Your endocrine glands produce hormones that control all your body functions.\"\n[output]=\"entails\"\n\n[input]=\"Sentence 1: Warm and humid temperature and moisture conditions describe an air mass that originates over the Atlantic ocean near the equator. Sentence 2: Maritime tropical air Warm, humid air mass that forms over tropical and subtropical oceans.\"\n[output]=\"neutral\"\n\n",
        "expected_content": "\"Sentence 1\" and \"Sentence 2\"",
        "optional_list": [
            "input",
            "output",
            "\n\n",
            "\\_\\_"
        ],
        "metric": "exact_match",
        "labels": ["entails", "neutral"]
    }

prompt_spec = MockPromptSpec(
    task_type=TaskType.TEXT_GENERATION,
    instruction=config["task_instruction"],
    examples=config["examples"],  # noqa E501
)

dataset_path = "/home/cyzhao/NI_task1554_exp_1/task1554_0.2_True_False_1/inputs"

inputs = load_from_disk(dataset_path)['input_col']

output_annotator = VLLMPromptBasedOutputAnnotator()

output_dataset = output_annotator.annotate_outputs(
    input_strings=inputs,
    prompt_spec=prompt_spec,
    hyperparameter_choices={},
    optional_list=config["optional_list"],
)
