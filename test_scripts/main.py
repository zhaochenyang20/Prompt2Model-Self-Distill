"""The main pipeline of Prompt2Model-Self-Distill."""

import argparse
import gc
import json
from pathlib import Path

import datasets
import torch
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

from prompt2model.input_generator import VLLMPromptBasedInputGenerator
from prompt2model.prompt_parser import MockPromptSpec, TaskType

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
    file_name = "inputs"
    with open(store_path / f"{file_name}.txt", "w") as file:
        for index, item in enumerate(inputs):
            file.write(
                f"{index}:\n\n------------------------------------------------{item}------------------------------------------------\n\n"
            )
    dataset = datasets.Dataset.from_dict({"input_col": inputs})
    dataset.save_to_disk(store_path / file_name)
    del input_generator
    destroy_model_parallel()
    gc.collect()
    torch.cuda.empty_cache()


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
