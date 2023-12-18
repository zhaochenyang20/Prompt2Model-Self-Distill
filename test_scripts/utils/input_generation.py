import gc

import datasets
import ray
import torch
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

from prompt2model.input_generator import VLLMPromptBasedInputGenerator


def generate_and_write_inputs(
    prompt_spec,
    generation_epochs,
    generation_batch_size,
    parameter_dict,
    log_and_data_path,
    gpu_memory_utilization,
    tensor_parallel_size,
    expected_content,
    optional_list,
    portion,
    intput_length_constraint,
    conditional_labels,
):
    ray.init(ignore_reinit_error=True)
    input_generator = VLLMPromptBasedInputGenerator(
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
    )
    input_tuples = input_generator.batch_generation_inputs(
        prompt_spec,
        generation_epochs,
        generation_batch_size,
        parameter_dict,
        expected_content,
        optional_list,
        portion,
        intput_length_constraint,
        conditional_labels,
    )
    inputs = [each[0] for each in input_tuples]
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
