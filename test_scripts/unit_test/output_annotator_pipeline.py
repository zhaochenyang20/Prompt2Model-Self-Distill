"""Test Input Generator."""

import os
from pathlib import Path
from datasets import Dataset, load_from_disk
from prompt2model.output_annotator import VLLMPromptBasedOutputAnnotator
from prompt2model.prompt_parser import MockPromptSpec, TaskType
from utils.path import ROOT

config =      {
        "task_instruction": "In this task, you're given a statement, and three sentences as choices. Your job is to determine which sentence clearly disagrees with the statement. Indicate your answer as '1', '2', or '3' corresponding to the choice number of the selected sentence.",
        "task_name": "task202",
        "examples": "[input]=\"Statement: Next to the MGM Grand you will find M and M World, four stories of merchandise and memorabilia dedicated to the candy that doesn't melt in your hand. Choices: 1. The candy has many fans who love its attractions. 2. There's four stories of memorabilia dedicated to a candy. 3. That particular candy melts and becomes difficult to eat.\"\n[output]=\"3\"\n\n[input]=\"Statment: I've forgotten his name now, confessed Tuppence. Choices: 1. Tuppence forgot his name. 2.Tuppence remembered his name later. 3. Tuppence never could forget his name.\"\n[output]=\"3\"\n\n[input]=\"Statement: One of the first organizational realignments taking place is in the Office of the Taxpayer Advocate. Choices: 1. The office of the taxpayer advocate is the last to be realigned. 2. The realignment is taking place over a few weeks. 3. The office of the taxpayer advocate is having an organizational realignment.\"\n[output]=\"1\"\n\n[input]=\"Statement: yeah i tell you what though if you go price some of those tennis shoes i can see why now you know they're getting up in the hundred dollar range. Choices: 1. The tennis shoes have a range of prices. 2. The tennis shoes can be in the hundred dollar range. 3. The tennis shoes are not over hundred dollars.\"\n[output]=\"3\"\n\n",
        "expected_content": "a \"Statement\" and three \"Choices\"",
        "optional_list": [
            "input",
            "output",
            "\n\n",
            "\\_\\_"
        ],
        "metric": "exact_match",
        "labels": ["1", "2", "3"]
    }

prompt_spec = MockPromptSpec(
    task_type=TaskType.TEXT_GENERATION,
    instruction=config["task_instruction"],
    examples=config["examples"],  # noqa E501
)

dataset_path = ROOT+"/NI_task202_exp_1/task202_0.6_False_False_1/inputs"

inputs = load_from_disk(dataset_path)['input_col']

output_annotator = VLLMPromptBasedOutputAnnotator()

output_dataset = output_annotator.annotate_outputs(
    input_strings=inputs,
    prompt_spec=prompt_spec,
    hyperparameter_choices={},
    optional_list=config["optional_list"],
)
