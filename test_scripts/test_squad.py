import argparse
import gc
import os
from functools import partial
from multiprocessing import Process
from pathlib import Path

import datasets
import evaluate
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from prompt2model.model_evaluator import Seq2SeqEvaluator
from prompt2model.model_executor import ModelOutput
from prompt2model.output_annotator.prompt_template import construct_meta_prompt
from prompt2model.prompt_parser import MockPromptSpec, TaskType

test_dataset = datasets.load_from_disk(
    "/home/cyzhao/prompt2model_test/testdataset/SQuAD_transformed"
)

ckpt_path = Path("/home/cyzhao/ckpt")
inputs_dir = Path("/home/cyzhao/generated_datasets")

prompt_spec = MockPromptSpec(
    task_type=TaskType.TEXT_GENERATION,
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
    example["model_input"] = construct_prompt(new_input=example["input_col"])
    example["model_output"] = example["output_col"]
    example["text"] = (
        example["model_input"] + example["model_output"] + tokenizer.eos_token
    )
    return example


test_dataset = test_dataset.map(map_func, load_from_cache_file=False)
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


def main(model_name):
    base_model = "/data/ckpts/huggingface/models/models--lmsys--vicuna-7b-v1.5/snapshots/de56c35b1763eaae20f4d60efd64af0a9091ebe5"
    # tuned_vicuna = LLM(model=str(ckpt_path / model_name))
    tuned_vicuna = LLM(model=base_model)
    tuned_vicuna_outputs = tuned_vicuna.generate(prompts, sampling_params)
    tuned_vicuna_generated_outputs = [
        each.outputs[0].text for each in tuned_vicuna_outputs
    ]
    index = 0
    for i in range(len(GROUND_TRUTH)):
        if (
            GROUND_TRUTH[i] in tuned_vicuna_generated_outputs[i]
            or tuned_vicuna_generated_outputs[i] in GROUND_TRUTH[i]
        ):
            index += 1
    print(index / len(GROUND_TRUTH))
    file_name = f"result_{model_name}"
    with open(inputs_dir / f"{file_name}.txt", "w") as file:
        file.write(
            f"result of {model_name}\n\n------------------------------------------------{index / len(GROUND_TRUTH)}------------------------------------------------\n\n"
        )
    print(model_name)
    del tuned_vicuna
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="")
    args = parser.parse_args()
    main(args.model_name)
