"""The main pipeline of Prompt2Model-Self-Distill."""

import argparse
import gc, os, re
import json
from functools import partial
from pathlib import Path
import logging
import datasets
import shutil
import torch, ray, random, numpy
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from vllm import LLM, SamplingParams
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
from prompt2model.input_generator import VLLMPromptBasedInputGenerator
from prompt2model.output_annotator import (
    VLLMPromptBasedOutputAnnotator,
    construct_meta_prompt,
)
from prompt2model.prompt_parser import MockPromptSpec, TaskType
from collections import OrderedDict


def set_seed(seed=42):
    # set seed for all possible avenues of stochasticity
    numpy.random.seed(seed=seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def generate_and_write_inputs(
    prompt_spec,
    generation_epochs,
    generation_batch_size,
    parameter_dict,
    log_and_data_path,
    gpu_memory_utilization,
    tensor_parallel_size,
):
    ray.init(ignore_reinit_error=True)
    input_generator = VLLMPromptBasedInputGenerator(
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
    )
    inputs = input_generator.batch_generation_inputs(
        prompt_spec, generation_epochs, generation_batch_size, parameter_dict
    )
    with open(log_and_data_path / f"inputs.txt", "w") as file:
        for index, item in enumerate(inputs):
            file.write(
                f"{index}:\n\n------------------------------------------------\n\n{item}\n\n------------------------------------------------\n\n"
            )
    dataset = datasets.Dataset.from_dict({"input_col": inputs})
    dataset.save_to_disk(log_and_data_path / "inputs")
    del input_generator
    destroy_model_parallel()
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()


def annotate_and_write_outputs(
    log_and_data_path, gpu_memory_utilization, min_frequency, tensor_parallel_size, prompt_spec
):
    if (log_and_data_path / "dataset").exists():
        return
    ray.init(ignore_reinit_error=True)
    output_annotator = VLLMPromptBasedOutputAnnotator(
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
    )
    dataset = datasets.load_from_disk(log_and_data_path / "inputs")
    inputs = dataset["input_col"]
    output_dataset = output_annotator.annotate_outputs(
        input_strings=inputs,
        prompt_spec=prompt_spec,
        hyperparameter_choices={"min_frequency": min_frequency},
    )
    output_dataset.save_to_disk(log_and_data_path / f"dataset")
    with open(log_and_data_path / f"dataset.txt", "w") as file:
        for index, item in enumerate(output_dataset):
            file.write(
                f"{index}:\n\n------------------------------------------------\n\n[INPUT]\n\n{item['input_col']}\n\n[OUPUT]\n\n{item['output_col']} \n\n------------------------------------------------\n\n"
            )
    del output_annotator
    destroy_model_parallel()
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()


def check_and_remove_checkpoints(ckpt_path):
    required_files = [
        "config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "pytorch_model-00001-of-00002.bin",
        "pytorch_model-00002-of-00002.bin",
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


def finetune_vicuna(
    prompt_spec,
    log_and_data_path,
    ckpt_path,
    pretrain_model_path,
    training_epochs,
    resume_from_checkpoint,
):
    construct_prompt = partial(
        construct_meta_prompt,
        instruction=prompt_spec.instruction,
        examples=prompt_spec.examples,
    )

    def filter_func(example):
        return example["output_col"] is not None and example["input_col"] is not None

    dataset = datasets.load_from_disk(log_and_data_path / "dataset").filter(filter_func)

    def map_func(example):
        example["model_input"] = construct_prompt(new_input=example["input_col"])
        example["model_output"] = example["output_col"]
        example["text"] = (
            example["model_input"] + example["model_output"] + tokenizer.eos_token
        )
        return example

    tokenizer = AutoTokenizer.from_pretrained(
        pretrain_model_path,
        local_files_only=True,
        padding_side="left",
        trust_remote_code=True,
    )
    mapped_dataset = dataset.map(map_func, load_from_cache_file=False)
    response_template_with_context = "\n### Your Output:\n\n"
    response_template_ids = tokenizer.encode(
        response_template_with_context, add_special_tokens=False
    )[2:]
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template_ids, tokenizer=tokenizer
    )
    ckpt_path.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        pretrain_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=True,
    )
    training_args = TrainingArguments(
        report_to="none",
        output_dir=str(ckpt_path),
        do_eval=False,
        save_strategy="epoch",
        evaluation_strategy="no",
        num_train_epochs=training_epochs,
        seed=42,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=mapped_dataset,
        dataset_text_field="text",
        data_collator=data_collator,
        max_seq_length=1500,
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    for dirpath, _, filenames in os.walk(str(ckpt_path), topdown=True):
        # Delete optimizer
        for filename in filenames:
            if filename == "optimizer.pt":
                file_path = os.path.join(dirpath, filename)
                print(f"Deleting {file_path}")
                os.system(f"rm -rf {(str(file_path))}")
    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()


def count_occurrences(word, filename):
    count = 0
    with open(filename, "r") as file:
        for line in file:
            count += line.lower().count(word.lower())
    return count


def extract_number(s):
    return int(re.search(r"\d+", s).group())


def validate(
    validation_set_path,
    log_and_data_path,
    ckpt_path,
    prompt_spec,
    gpu_memory_utilization,
    tensor_parallel_size,
    evaluate_result_path,
):
    assert evaluate_result_path.exists()
    with open(evaluate_result_path, "r") as json_file:
        evaluate_result = json.load(json_file)
    construct_prompt = partial(
        construct_meta_prompt,
        instruction=prompt_spec.instruction,
        examples=prompt_spec.examples,
    )

    def map_func(example):
        example["model_input"] = construct_prompt(new_input=example["input_col"])
        example["model_output"] = example["output_col"]
        return example

    test_dataset = datasets.load_from_disk(validation_set_path)
    # test_dataset = datasets.Dataset.from_dict(test_dataset[:20])
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

    sorted_list = sorted(
        [each for each in os.listdir(ckpt_path) if "checkpoint" in each],
        key=extract_number,
    )
    last_evaluate = len(list(evaluate_result.keys()))
    print(f"last validate {last_evaluate}.")
    for ckpt_index, each in enumerate(sorted_list):
        if ckpt_index < last_evaluate:
            print(f"skip the evaluation of the {ckpt_index + 1} epoch.")
            continue
        ray.init(ignore_reinit_error=True)
        model_path = ckpt_path / each
        tuned_model = LLM(
            model=str(model_path),
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
        )
        tuned_model_outputs = tuned_model.generate(prompts, sampling_params)
        tuned_model_generated_outputs = [
            each.outputs[0].text for each in tuned_model_outputs
        ]
        index = 0
        for i in range(len(GROUND_TRUTH)):
            if (
                GROUND_TRUTH[i] in tuned_model_generated_outputs[i]
                or tuned_model_generated_outputs[i] in GROUND_TRUTH[i]
            ):
                index += 1
        name = str(log_and_data_path).split("/")[-1]
        exact_match = index / len(GROUND_TRUTH)
        evaluate_result[f"{ckpt_index + 1}"] = exact_match
        print(
            f"\n\nresult of {name} epoch {ckpt_index + 1}\n\n------------------------------------------------\n\n{exact_match}\n\n------------------------------------------------\n\n"
        )
        with open(evaluate_result_path, "w") as f:
            json.dump(evaluate_result, f, indent=4)
        del tuned_model
        evaluate_generated_content_path = log_and_data_path / "generated_contents"
        evaluate_generated_content_path.mkdir(parents=True, exist_ok=True)
        content_store_path = str(evaluate_generated_content_path / str(ckpt_index + 1))
        print(f"Genrated contents are stored in {content_store_path}")
        datasets.Dataset.from_dict(
            dict(
                model_output=tuned_model_generated_outputs,
                model_input=prompts,
                groud_truth=GROUND_TRUTH,
            )
        ).save_to_disk(content_store_path)
        gc.collect()
        torch.cuda.empty_cache()
        destroy_model_parallel()
        ray.shutdown()


def get_ckpt_paths_and_result(ckpt_path, evaluate_result_path):
    sorted_list = sorted(
        [each for each in os.listdir(ckpt_path) if "checkpoint" in each],
        key=extract_number,
    )
    with open(evaluate_result_path, "r") as json_file:
        evaluate_result = json.load(json_file)
    assert len(sorted_list) == len(list(evaluate_result.keys()))
    return OrderedDict(
        (str(ckpt_path / each), evaluate_result[str(ckpt_index + 1)])
        for ckpt_index, each in enumerate(sorted_list)
    )


def search_against_parameter(config_path: str):
    set_seed(42)
    print(str(config_path))
    with open(config_path, "r") as json_file:
        loaded_params = json.load(json_file)
    gpu_memory_utilization = loaded_params["gpu_memory_utilization"]
    tensor_parallel_size = loaded_params["tensor_parallel_size"]
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
            prompt_spec,
            loaded_params["generation_epochs"],
            loaded_params["generation_batch_size"],
            dict(
                top_k=loaded_params["generation_top_k"],
                temperature=loaded_params["generation_temperature"],
                min_input_length=loaded_params["min_input_length"],
            ),
            log_and_data_path,
            gpu_memory_utilization,
            tensor_parallel_size,
        )
    if not (log_and_data_path / "dataset").exists():
        print("annotate_and_write_outputs!")
        logging.log(logging.INFO, "annotate_and_write_outputs!")
        min_frequency = loaded_params["min_frequency"]
        annotate_and_write_outputs(
            log_and_data_path,
            gpu_memory_utilization,
            min_frequency,
            tensor_parallel_size,
            prompt_spec,
        )

    pretrain_model_path = Path(
        "/data/ckpts/huggingface/models/models--lmsys--vicuna-7b-v1.5/snapshots/de56c35b1763eaae20f4d60efd64af0a9091ebe5"
    )

    complete_ckpts = check_and_remove_checkpoints(ckpt_path)
    evaluate_result_path = (
        log_and_data_path / loaded_params["evaluation_result_file_tail"]
    )

    assert evaluate_result_path.exists()
    with open(evaluate_result_path, "r") as json_file:
        evaluate_result = json.load(json_file)

    assert len(list(evaluate_result.keys())) <= loaded_params["training_epochs"]

    if (
        complete_ckpts < loaded_params["training_epochs"]
        and len(list(evaluate_result.keys())) < loaded_params["training_epochs"]
    ):
        print("finetune_vicuna!")
        logging.log(logging.INFO, "finetune_vicuna!")
        finetune_vicuna(
            prompt_spec,
            log_and_data_path,
            ckpt_path,
            pretrain_model_path,
            loaded_params["training_epochs"],
            resume_from_checkpoint=False if complete_ckpts == 0 else True,
        )

    validation_set_path = Path(
        "/home/cyzhao/prompt2model_test/testdataset/SQuAD_transformed"
    )

    if len(list(evaluate_result.keys())) < loaded_params["training_epochs"]:
        print("validate!")
        logging.log(logging.INFO, "validate!")
        validate(
            validation_set_path,
            log_and_data_path,
            ckpt_path,
            prompt_spec,
            gpu_memory_utilization,
            tensor_parallel_size,
            evaluate_result_path,
        )

    return get_ckpt_paths_and_result(ckpt_path, evaluate_result_path)
