"""The main pipeline of Prompt2Model-Self-Distill."""

import json
import logging
import os
import random
import re
import shutil
from pathlib import Path

import datasets
import numpy
import torch
import time

from prompt2model.utils import count_tokens_from_string
from prompt2model.utils.path import MODEL_PATH

# wandb sync wandb/offline-run-*
os.environ["WANDB_MODE"] = "offline"



from collections import OrderedDict

from prompt2model.prompt_parser import MockPromptSpec, TaskType
from utils.evaluate import exact_match_score, rouge_l_score
from utils.inference import vllm_inference
from utils.input_generation import generate_and_write_inputs
from utils.output_annotation import annotate_and_write_outputs
from utils.trainer import finetune_vicuna
from prompt2model.output_annotator import construct_meta_prompt


def set_seed(seed=42):
    # set seed for all possible avenues of stochasticity
    numpy.random.seed(seed=seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def check_and_remove_checkpoints(ckpt_path):
    required_files = [
        "config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
        "tokenizer.json",
        "tokenizer.model",
    ]

    sorted_list = sorted(
        [each for each in os.listdir(ckpt_path) if each.startswith("checkpoint-")],
        key=extract_number,
    )

    for each in sorted_list:
        checkpoint_path = os.path.join(ckpt_path, each)
        if all(
            os.path.exists(os.path.join(checkpoint_path, file))
            for file in required_files
        ):
            print(f"Checkpoint '{each}' is complete.")
        else:
            print(f"Checkpoint '{each}' is incomplete and will be removed.")
            shutil.rmtree(checkpoint_path)

    return len(
        sorted(
            [each for each in os.listdir(ckpt_path) if each.startswith("checkpoint-")],
            key=extract_number,
        )
    )

def extract_number(s):
    return int(re.search(r"\d+", s).group())

def store_evaluation_content(
    content_store_path, tuned_model_generated_outputs, prompts, GROUND_TRUTH
):
    print(f"Genrated contents are stored in {content_store_path}")
    datasets.Dataset.from_dict(
        dict(
            model_output=tuned_model_generated_outputs,
            model_input=prompts,
            groud_truth=GROUND_TRUTH,
        )
    ).save_to_disk(content_store_path)

def validate_or_test(
    evaluation_dataset_path,
    ckpt_path,
    instruction,
    examples,
    gpu_memory_utilization,
    tensor_parallel_size,
    evaluate_result_path,
    test_content_store_path=None,
    log_and_data_path=None,
    validation=True,
    metric="exact_match",
):


    def map_func(example):
        example["model_input"] = construct_meta_prompt(
            instruction=instruction,
            new_input=example["input_col"],
            examples=examples,
            is_generation=False,
        )
        example["model_output"] = example["output_col"].strip()
        return example

    loaded_dataset = datasets.load_from_disk(evaluation_dataset_path)
    # loaded_dataset = datasets.Dataset.from_dict(loaded_dataset[:20])
    loaded_dataset = loaded_dataset.map(map_func, load_from_cache_file=False)
    test_dataset = loaded_dataset.filter(
        lambda x: (
            count_tokens_from_string(x["model_input"], "vicuna") <= 3000
            and count_tokens_from_string(x["model_output"], "vicuna") <= 500
        )
    )
    prompts = test_dataset["model_input"]
    GROUND_TRUTH = test_dataset["model_output"]

    #! test
    tuned_model_generated_outputs = vllm_inference(
        ckpt_path, gpu_memory_utilization, tensor_parallel_size, prompts
    )
    if metric == "exact_match":
        score = exact_match_score(GROUND_TRUTH, tuned_model_generated_outputs)
    else:
        score = rouge_l_score(GROUND_TRUTH, tuned_model_generated_outputs)
    print(
        f"\n\nresult of {ckpt_path}\n\n------------------------------------------------\n\n{score}\n\n------------------------------------------------\n\n"
    )
    with open(evaluate_result_path, "r") as json_file:
        evaluate_result = json.load(json_file)
    evaluate_result["test_result"] = score
    print(f"The best ckpt on test set gain {score}")
    with open(evaluate_result_path, "w") as f:
        json.dump(evaluate_result, f, indent=4)
    store_evaluation_content(
        test_content_store_path,
        tuned_model_generated_outputs,
        prompts,
        GROUND_TRUTH,
    )

def main(config_path: str):
    set_seed(int(time.time()))

    with open(config_path, "r") as json_file:
        loaded_params = json.load(json_file)
    gpu_memory_utilization = loaded_params["gpu_memory_utilization"]
    TENSOR_SIZE = loaded_params["tensor_parallel_size"]
    expected_content = loaded_params["expected_content"]
    prompt_spec = MockPromptSpec(
        task_type=TaskType.TEXT_GENERATION,
        instruction=loaded_params["instruction"],
        examples=loaded_params["examples"],
    )
    log_and_data_path = Path(loaded_params["log_and_data_path"])
    ckpt_path = Path(loaded_params["ckpt_path"])

    assert log_and_data_path.exists()
    if not (log_and_data_path / "inputs").exists():
        print("generate_and_write_inputs!")
        logging.log(logging.INFO, "generate_and_write_inputs!")

        generate_and_write_inputs(
            prompt_spec = prompt_spec,
            generation_epochs = loaded_params["generation_epochs"],
            generation_batch_size = loaded_params["generation_batch_size"],
            parameter_dict = dict(
                top_k=loaded_params["generation_top_k"],
                temperature=loaded_params["generation_temperature"],
            ),
            log_and_data_path = log_and_data_path,
            gpu_memory_utilization = gpu_memory_utilization,
            expected_content = expected_content,
            optional_list = loaded_params["optional_list"],
            intput_length_constraint=loaded_params["intput_length_constraint"],
            conditional_labels=loaded_params["conditional_labels"],
            reannotate=True,
            extraction_examples=loaded_params["extraction_examples"],
            TENSOR_SIZE=TENSOR_SIZE,
        )

    if not (log_and_data_path / "dataset").exists():
        print("annotate_and_write_outputs!")
        logging.log(logging.INFO, "annotate_and_write_outputs!")

        annotate_and_write_outputs(
            log_and_data_path,
            gpu_memory_utilization,
            loaded_params["min_frequency"],
            TENSOR_SIZE,
            prompt_spec,
            loaded_params["optional_list"],
            loaded_params["output_length_constraint"],
            loaded_params["conditional_labels"],
        )

    # TODO: uncomment later, after 281 dataset generation
    if len(datasets.load_from_disk(log_and_data_path / "dataset")) == 0:
        return None

    pretrain_model_path = Path(MODEL_PATH)

    complete_ckpts = check_and_remove_checkpoints(ckpt_path)
    if (
        complete_ckpts < loaded_params["training_epochs"]
    ):
        print("finetune_vicuna!")
        logging.log(logging.INFO, "finetune_vicuna!")
        print(f'complete_ckpts = {complete_ckpts}')
        finetune_vicuna(
            prompt_spec,
            log_and_data_path,
            ckpt_path,
            pretrain_model_path,
            loaded_params["training_epochs"],
            resume_from_checkpoint=False if complete_ckpts == 0 else True,
            run_name=config_path.split("/")[-2],
            task_name=loaded_params["task_name"],
            per_device_train_batch_size=loaded_params["per_device_train_batch_size"],
        )
    
    return 0
