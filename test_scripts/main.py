"""The main pipeline of Prompt2Model-Self-Distill."""

import argparse
import gc
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
    #! epochs, per_epoch_num, top_k, temperature
    experiments = [
        (20, 20, 50, 1.0),
        (40, 10, 50, 1.0),
        (20, 20, 30, 1.0),
        (20, 20, 10, 1.0),
        (20, 20, 50, 0.5),
        (20, 20, 50, 1.5),
    ]
    for epochs, per_epoch_num, top_k, temperature in experiments:
        parameter_dict = dict(top_k=top_k, temperature=temperature)
        generate_and_write_inputs(epochs, per_epoch_num, parameter_dict)


if __name__ == "__main__":
    file_name = f"inputs_{epochs}_{per_epoch_num}_{parameter_dict['top_k']}_{parameter_dict['temperature']}"
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="")
    parser.add_argument("--instruction", type=str, default="")
    parser.add_argument("--examples", type=str, default="")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--per_epoch_num", type=int, default=20)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()
    assert all(
        [args.task_name, args.instruction, args.examples]
    ), "Task name, instruction, and examples cannot be empty."

    prompt_spec = MockPromptSpec(
        task_type=TaskType.TEXT_GENERATION,
        instruction=args.instruction,
        examples=args.examples,
    )
    store_path = (
        root_dir
        / f"{args.task_name}_{args.epochs}_{args.per_epoch_num}_{args.top_k}_{args.temperature}"
    )
    store_path.mkdir(parents=True, exist_ok=True)
    generate_and_write_inputs(
        prompt_spec=prompt_spec,
        epochs=args.epochs,
        per_epoch_num=per_epoch_num,
        parameter_dict=dict(top_k=args.top_k, temperature=args.temperature),
        store_path=store_path,
    )
