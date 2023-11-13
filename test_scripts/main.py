"""The main pipeline of Prompt2Model-Self-Distill."""

import argparse
import gc, os, re
import json
from functools import partial
from pathlib import Path

import datasets
import torch
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

root_dir = Path("/home/cyzhao/ckpt_data_p2ms")
root_dir.mkdir(parents=True, exist_ok=True)


def generate_and_write_inputs(
    prompt_spec,
    generation_epochs,
    generation_batch_size,
    parameter_dict,
    store_path,
    gpu_memory_utilization,
):
    input_generator = VLLMPromptBasedInputGenerator(
        gpu_memory_utilization=gpu_memory_utilization
    )
    inputs = input_generator.batch_generation_inputs(
        prompt_spec, generation_epochs, generation_batch_size, parameter_dict
    )
    with open(store_path / f"inputs.txt", "w") as file:
        for index, item in enumerate(inputs):
            file.write(
                f"{index}:\n\n------------------------------------------------\n\n{item}\n\n------------------------------------------------\n\n"
            )
    dataset = datasets.Dataset.from_dict({"input_col": inputs})
    dataset.save_to_disk(store_path / "inputs")
    del input_generator
    destroy_model_parallel()
    gc.collect()
    torch.cuda.empty_cache()


def annotate_and_write_outputs(store_path, gpu_memory_utilization, min_frequency):
    if (store_path / "dataset").exists():
        return
    output_annotator = VLLMPromptBasedOutputAnnotator(
        gpu_memory_utilization=gpu_memory_utilization
    )
    dataset = datasets.load_from_disk(store_path / "inputs")
    inputs = dataset["input_col"]
    output_dataset = output_annotator.annotate_outputs(
        input_strings=inputs,
        prompt_spec=prompt_spec,
        hyperparameter_choices={"min_frequency": min_frequency},
    )
    output_dataset.save_to_disk(store_path / f"dataset")
    with open(store_path / f"dataset.txt", "w") as file:
        for index, item in enumerate(output_dataset):
            file.write(
                f"{index}:\n\n------------------------------------------------\n\n[INPUT]\n\n{item['input_col']}\n\n[OUPUT]\n\n{item['output_col']} \n\n------------------------------------------------\n\n"
            )
    del output_annotator
    destroy_model_parallel()
    gc.collect()
    torch.cuda.empty_cache()


def finetune_vicuna(prompt_spec, store_path, pretrain_model_path, training_epochs):
    construct_prompt = partial(
        construct_meta_prompt,
        instruction=prompt_spec.instruction,
        examples=prompt_spec.examples,
    )

    def filter_func(example):
        return example["output_col"] is not None and example["input_col"] is not None

    dataset = datasets.load_from_disk(store_path / "dataset").filter(filter_func)

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
    print(mapped_dataset[1]["text"])
    model = AutoModelForCausalLM.from_pretrained(
        pretrain_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=True,
    )
    response_template_with_context = "\n### Your Output:\n\n"
    response_template_ids = tokenizer.encode(
        response_template_with_context, add_special_tokens=False
    )[2:]
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template_ids, tokenizer=tokenizer
    )
    ckpt_path = store_path / "model"
    ckpt_path.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        report_to="none",
        output_dir=str(ckpt_path),
        do_eval=False,
        save_strategy="epoch",
        num_train_epochs=training_epochs,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=mapped_dataset,
        dataset_text_field="text",
        data_collator=data_collator,
        max_seq_length=1500,
    )
    trainer.train()
    for dirpath, _, filenames in os.walk(str(ckpt_path), topdown=True):
    # Delete optimizer
        for filename in filenames:
            if filename == "optimizer.pt":
                file_path = os.path.join(dirpath, filename)
                print(f"Deleting {file_path}")
                os.remove(file_path)
    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()


def evaluate(
    test_set_path,
    store_path,
    prompt_spec,
    gpu_memory_utilization,
):
    construct_prompt = partial(
        construct_meta_prompt,
        instruction=prompt_spec.instruction,
        examples=prompt_spec.examples,
    )

    def map_func(example):
        example["model_input"] = construct_prompt(new_input=example["input_col"])
        example["model_output"] = example["output_col"]
        return example

    test_dataset = datasets.load_from_disk(test_set_path)
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
    ckpt_path = store_path / "model"
    def extract_number(s):
        return int(re.search(r'\d+', s).group())

    sorted_list = sorted([each for each in os.listdir(ckpt_path) if "checkpoint" in each], key=extract_number)
    for ckpt_index, each in enumerate(sorted_list):
        model_path = ckpt_path / each
        tuned_model = LLM(
            model=str(model_path), gpu_memory_utilization=gpu_memory_utilization
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
        name = str(store_path).split("/")[-1]
        with open(store_path / f"result.txt", "a+") as file:
            file.write(
                f"\n\nresult of {name} epoch {ckpt_index + 1}\n\n------------------------------------------------{index / len(GROUND_TRUTH)}------------------------------------------------\n\n"
            )
        del tuned_model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    config_path = parser.parse_args().config
    with open(parser.parse_args().config, "r") as json_file:
        loaded_params = json.load(json_file)
    gpu_memory_utilization = loaded_params["gpu_memory_utilization"]
    prompt_spec = MockPromptSpec(
        task_type=TaskType.TEXT_GENERATION,
        instruction=loaded_params["instruction"],
        examples=loaded_params["examples"],
    )
    store_path = Path(loaded_params["store_path"])
    assert store_path.exists()
    if not (store_path / "inputs").exists():
        generate_and_write_inputs(
            prompt_spec=prompt_spec,
            generation_epochs=loaded_params["generation_epochs"],
            generation_batch_size=loaded_params["generation_batch_size"],
            parameter_dict=dict(
                top_k=loaded_params["generation_top_k"],
                temperature=loaded_params["generation_temperature"],
                min_input_length=loaded_params["min_input_length"],
            ),
            store_path=store_path,
            gpu_memory_utilization=gpu_memory_utilization,
        )
    if not (store_path / "dataset").exists():
        min_frequency = loaded_params["min_frequency"]
        annotate_and_write_outputs(store_path, gpu_memory_utilization, min_frequency)

    pretrain_model_path = Path(
        "/data/ckpts/huggingface/models/models--lmsys--vicuna-7b-v1.5/snapshots/de56c35b1763eaae20f4d60efd64af0a9091ebe5"
    )

    if not (store_path / "model").exists():
        finetune_vicuna(prompt_spec, store_path, pretrain_model_path, loaded_params["training_epochs"])

    test_set_path = Path("/home/cyzhao/prompt2model_test/testdataset/SQuAD_transformed")

    if not (store_path / "result.txt").exists():
        evaluate(test_set_path, store_path, prompt_spec, gpu_memory_utilization)
