import gc
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from functools import partial
from pathlib import Path

import datasets
import ray
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

from prompt2model.output_annotator.prompt_template import construct_meta_prompt
from prompt2model.prompt_parser import MockPromptSpec, TaskType
from utils.path import ROOT, STORE_ROOT

#! TODO 改为新任务的 test set path


original_dataset = load_dataset("squad", split="validation")


def join_function(example):
    new_example = {}
    new_example["input_col"] = (
        "Question: " + example["question"] + " Context: " + example["context"]
    )
    new_example["output_col"] = example["answers"]["text"][0]
    return new_example


joined_dataset = original_dataset.map(join_function)
joined_dataset.remove_columns(["question", "answers"])

validation_set = datasets.Dataset.from_dict(joined_dataset[0:1000])
test_set = datasets.Dataset.from_dict(joined_dataset[1000:2000])
test_dataset = test_set
inputs_dir = Path(ROOT+"/evaluation_outputs")

prompt_spec = MockPromptSpec(
    task_type=TaskType.TEXT_GENERATION,
    #! TODO 改 instruction 和 examples
    instruction="Your task is to generate an answer to a natural question. In this task, the input is a string that consists of both a question and a context passage. The context is a descriptive passage related to the question and contains the answer. And the question can range from Math, Cultural, Social, Geometry, Biology, History, Sports, Technology, Science, and so on.",  # # noqa E501
    examples="""
[input]="Question: What city did Super Bowl 50 take place in? Context: Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50."
[output]="Santa Clara"

[input]="Question: What river runs through Warsaw? Context: Warsaw (Polish: Warszawa [varˈʂava] ( listen); see also other names) is the capital and largest city of Poland. It stands on the Vistula River in east-central Poland, roughly 260 kilometres (160 mi) from the Baltic Sea and 300 kilometres (190 mi) from the Carpathian Mountains. Its population is estimated at 1.740 million residents within a greater metropolitan area of 2.666 million residents, which makes Warsaw the 9th most-populous capital city in the European Union. The city limits cover 516.9 square kilometres (199.6 sq mi), while the metropolitan area covers 6,100.43 square kilometres (2,355.39 sq mi)."
[output]="Vistula River"

[input]="Question: The Ottoman empire controlled territory on three continents, Africa, Asia and which other? Context: The Ottoman Empire was an imperial state that lasted from 1299 to 1923. During the 16th and 17th centuries, in particular at the height of its power under the reign of Suleiman the Magnificent, the Ottoman Empire was a powerful multinational, multilingual empire controlling much of Southeast Europe, Western Asia, the Caucasus, North Africa, and the Horn of Africa. At the beginning of the 17th century the empire contained 32 provinces and numerous vassal states. Some of these were later absorbed into the empire, while others were granted various types of autonomy during the course of centuries."
[output]="Europe"
"""
    # noqa E501
)

PROMPT_TEMPLATE = """
A chat between a curious user and an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers to the user's questions.
USER: 

{task_instruction}

ASSISTANT: Okay.

USER:

{new_input}

ASSISTANT: The output is

"""

def map_func(example):
    example["model_input"] = PROMPT_TEMPLATE.format(
        task_instruction=prompt_spec.instruction,
        new_input=example["input_col"],
    )
    example["model_output"] = example["output_col"]
    return example


test_dataset = datasets.Dataset.from_dict(test_dataset.map(map_func, load_from_cache_file=False)[:40])
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
base_model = "/data/datasets/models/huggingface/lmsys/vicuna-7b-v1.5"
path = STORE_ROOT+"/ckpt_1/checkpoint-31"
ray.init(ignore_reinit_error=True)
tuned_vicuna = LLM(model=path, gpu_memory_utilization=0.5, tensor_parallel_size=1)
tuned_vicuna_outputs = tuned_vicuna.generate(prompts, sampling_params)
tuned_vicuna_generated_outputs = [each.outputs[0].text for each in tuned_vicuna_outputs]


def evalute_squad(
    GROUND_TRUTH,
    tuned_model_generated_outputs,
):
    index = 0
    for i in range(len(GROUND_TRUTH)):
        if (
            GROUND_TRUTH[i] in tuned_model_generated_outputs[i]
            or tuned_model_generated_outputs[i] in GROUND_TRUTH[i]
        ):
            index += 1
    exact_match = index / len(GROUND_TRUTH)
    return exact_match


for idx, i in enumerate(list(range(0, len(test_dataset), 1000))):
    exact_match = evalute_squad(
        GROUND_TRUTH=GROUND_TRUTH[i : i + 1000],
        tuned_model_generated_outputs=tuned_vicuna_generated_outputs[i : i + 1000],
    )
    with open(inputs_dir / f"evaluate_base_model_on_the_whole.txt", "a+") as file:
        print(
            f"\n\nresult of {idx + 1} th:\n\n------------------------------------------------{exact_match}------------------------------------------------\n\n"
        )
        file.write(
            f"\n\nresult of {idx + 1} th:\n\n------------------------------------------------{exact_match}------------------------------------------------\n\n"
        )
del tuned_vicuna
# #! 记得改名字
gc.collect()
torch.cuda.empty_cache()
ray.shutdown()
destroy_model_parallel()

# ckpt 2: 0.402, 0.448, 0.46
# ckpt 1: 0.402, 0.46/0.448, 0.46
# with flash_attn: 0.568,
