"""The main pipeline of Prompt2Model-Self-Distill."""

import argparse
import gc
import json
from pathlib import Path
from prompt2model.output_annotator import construct_meta_prompt
import datasets
import torch
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from functools import partial
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
from prompt2model.input_generator import VLLMPromptBasedInputGenerator
from prompt2model.output_annotator import VLLMPromptBasedOutputAnnotator
from prompt2model.prompt_parser import MockPromptSpec, TaskType
from vllm import LLM, SamplingParams

root_dir = Path("/home/cyzhao/ckpt_data_p2ms")
root_dir.mkdir(parents=True, exist_ok=True)

def generate_and_write_inputs(
    prompt_spec, epochs, per_epoch_num, parameter_dict, store_path
):
    input_generator = VLLMPromptBasedInputGenerator()
    inputs = input_generator.batch_generation_inputs(
        prompt_spec,
        epochs,
        per_epoch_num,
        parameter_dict,
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
    destroy_model_parallel()


def annotate_and_write_outputs(store_path):
    output_annotator = VLLMPromptBasedOutputAnnotator()
    dataset = datasets.load_from_disk(store_path / "inputs")
    print(dataset)
    inputs = dataset["input_col"]
    output_dataset = output_annotator.annotate_outputs(
        input_strings=inputs,
        prompt_spec=prompt_spec,
        hyperparameter_choices={},
    )
    output_dataset.save_to_disk(store_path / f"dataset")
    with open(store_path / f"dataset.txt", "w") as file:
        for index, item in enumerate(output_dataset):
            file.write(
                f"{index}:\n\n[INPUT]\n\n------------------------------------------------\n\n{item['input_col']}\n\n[OUPUT]\n\n{item['output_col']} \n\n------------------------------------------------\n\n"
            )


def finetune_vicuna(prompt_spec, store_path, model_path):
    construct_prompt = partial(
        construct_meta_prompt,
        instruction=prompt_spec.instruction,
        examples=prompt_spec.examples,
    )
    def filter_func(example):
        return example["output_col"] is not None and example["input_col"] is not None

    dataset = datasets.load_from_disk(store_path / "dataset").filter(
            filter_func
        )

    def map_func(example):
        example["model_input"] = construct_prompt(new_input=example["input_col"])
        example["model_output"] = example["output_col"]
        example["text"] = (
            example["model_input"] + example["model_output"] + tokenizer.eos_token
        )
        return example

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        padding_side="left",
        trust_remote_code=True,
    )
    mapped_dataset = dataset.map(map_func, load_from_cache_file=False)
    print(mapped_dataset[1]["text"])
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
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
    training_args = TrainingArguments(
        report_to="none",
        output_dir=str(store_path),
        do_eval=False,
        save_strategy="no",
        num_train_epochs=1,
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
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    model.save_pretrained(store_path / "model")
    tokenizer.save_pretrained(store_path / "model")

def evaluate(test_set_path, store_path, prompt_spec, ):
    model_path = store_path / "model"
    construct_prompt = partial(
        construct_meta_prompt,
        instruction=prompt_spec.instruction,
        examples=prompt_spec.examples,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        padding_side="left",
        trust_remote_code=True,
    )
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
    tuned_model = LLM(model=str(model_path))
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
    print(index / len(GROUND_TRUTH))
    name = str(store_path).split("/")[-1]
    with open(store_path / f"result.txt", "w") as file:
        file.write(
            f"result of {name}\n\n------------------------------------------------{index / len(GROUND_TRUTH)}------------------------------------------------\n\n"
        )
    print(name)
    del tuned_model
    gc.collect()
    torch.cuda.empty_cache()

    def map_func(example):
        example["model_input"] = construct_prompt(new_input=example["input_col"])
        example["model_output"] = example["output_col"]
        example["text"] = (
            example["model_input"] + example["model_output"] + tokenizer.eos_token
        )
        return example

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    config_path = parser.parse_args().config
    with open(parser.parse_args().config, "r") as json_file:
        loaded_params = json.load(json_file)
    prompt_spec = MockPromptSpec(
        task_type=TaskType.TEXT_GENERATION,
        instruction=loaded_params["instruction"],
        examples=loaded_params["examples"],
    )
    store_path = Path(loaded_params["store_path"])
    assert store_path.exists()
    generate_and_write_inputs(
        prompt_spec=prompt_spec,
        epochs=loaded_params["epochs"],
        per_epoch_num=loaded_params["per_epoch_num"],
        parameter_dict=dict(
            top_k=loaded_params["top_k"], temperature=loaded_params["temperature"]
        ),
        store_path=store_path,
    )
    annotate_and_write_outputs(store_path)

    model_path = Path(
    "/data/ckpts/huggingface/models/models--lmsys--vicuna-7b-v1.5/snapshots/de56c35b1763eaae20f4d60efd64af0a9091ebe5")

    finetune_vicuna(prompt_spec, store_path, model_path)

