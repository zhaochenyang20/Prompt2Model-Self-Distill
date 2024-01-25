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
    expected_content,
    optional_list,
    intput_length_constraint,
    conditional_labels,
    reannotate=True,
    extraction_examples=[],
    tensor_parallel_size=1,
):
    input_generator = VLLMPromptBasedInputGenerator(
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
    )
    input_tuples = input_generator.batch_generation_inputs(
        prompt_spec=prompt_spec,
        generation_epochs=generation_epochs,
        per_epoch_num=generation_batch_size,
        hyperparameter_choices=parameter_dict,
        expected_content=expected_content,
        optional_list=optional_list,
        intput_length_constraint=intput_length_constraint,
        conditional_labels=conditional_labels,
        extraction_examples=extraction_examples,
    )
    inputs = [each[0] for each in input_tuples]
    pesudo_labels = [each[1] for each in input_tuples]
    with open(log_and_data_path / f"inputs.txt", "w", encoding="utf-8") as file:
        for index, item in enumerate(inputs):
            file.write(
                f"{index}:\n\n------------------------------------------------\n\n{item}\n\n------------------------------------------------\n\n"
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
