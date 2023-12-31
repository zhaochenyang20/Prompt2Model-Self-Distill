import gc

import datasets
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
    reannotate=True,
    extraction_examples=[],
):
    input_generator = VLLMPromptBasedInputGenerator(
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=1,
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
        extraction_examples,
    )
    inputs = [each[0] for each in input_tuples]
    pesudo_labels = [each[1] for each in input_tuples]
    with open(log_and_data_path / f"inputs.txt", "w", encoding="utf-8") as file:
        for index, item in enumerate(inputs):
            file.write(
                fr"{index}:\n\n------------------------------------------------\n\n{item}\n\n------------------------------------------------\n\n"
            )
    dataset = datasets.Dataset.from_dict({"input_col": inputs})
    dataset.save_to_disk(log_and_data_path / "inputs")
    if conditional_labels != [] and reannotate == False:
            dataset = datasets.Dataset.from_dict({"input_col": inputs, "output_col": pesudo_labels})
            dataset.save_to_disk(log_and_data_path / "dataset")
            with open(log_and_data_path / f"dataset.txt", "w") as file:
                for index, item in enumerate(inputs):
                    file.write(
                        f"{index}:\n\n------------------------------------------------\n\n{item}\n\n------------------------------------------------\n\n"
                    )
                    file.write(
                        f"------------------------------------------------\n\n{pesudo_labels[index]}\n\n------------------------------------------------\n\n"
                    )
    del input_generator
    destroy_model_parallel()
    gc.collect()
    torch.cuda.empty_cache()
