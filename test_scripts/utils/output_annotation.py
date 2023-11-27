import gc
import json

import datasets
import ray
import torch
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

from prompt2model.output_annotator import VLLMPromptBasedOutputAnnotator


def annotate_and_write_outputs(
    log_and_data_path,
    gpu_memory_utilization,
    min_frequency,
    tensor_parallel_size,
    prompt_spec,
    optional_list,
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
        hyperparameter_choices=dict(
            min_frequency=min_frequency,
        ),
        optional_list=optional_list,
    )
    output_dataset.save_to_disk(log_and_data_path / f"dataset")
    with open(log_and_data_path / f"dataset.txt", "w") as file:
        for index, item in enumerate(output_dataset):
            file.write(
                f"{index}:\n\n------------------------------------------------\n\n[INPUT]\n\n{item['input_col']}\n\n[OUPUT]\n\n{item['output_col']} \n\n------------------------------------------------\n\n"
            )
    with open(log_and_data_path / "config.json", "r") as json_file:
        loaded_params = json.load(json_file)
    loaded_params["generated_example_num"] = len(output_dataset)
    loaded_params["expected_example_num"] = (
        loaded_params["generation_epochs"] * loaded_params["generation_batch_size"]
    )
    loaded_params["selection_ratio"] = (
        loaded_params["generated_example_num"] / loaded_params["expected_example_num"]
    )
    print(f"generated_example_num: {loaded_params['generated_example_num']}")
    print(f"expected_example_num: {loaded_params['expected_example_num']}")
    print(f"selection_ratio: {loaded_params['selection_ratio']}")
    with open(log_and_data_path / "config.json", "w") as f:
        json.dump(loaded_params, f, indent=4)
    del output_annotator
    destroy_model_parallel()
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()
