import gc
import json

import datasets
import random
from collections import Counter
from datasets import Dataset
import torch
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
from functools import partial
from prompt2model.output_annotator import VLLMPromptBasedOutputAnnotator
from IPython import embed

def filter_func(example, conditional_labels):
    return example["output_col"] in conditional_labels

def mapping_func(example, conditional_labels):
    for conditional_label in conditional_labels:
        if conditional_label in example["output_col"]:
            example["output_col"] = conditional_label
            return

def reverse_func(example, conditional_labels):
    if example['output_col'] == conditional_labels[0]:
        example['output_col'] = conditional_labels[1]
    else:
        example['output_col'] = conditional_labels[0]
    return example

def annotate_and_write_outputs(
    log_and_data_path,
    gpu_memory_utilization,
    min_frequency,
    TENSOR_SIZE,
    prompt_spec,
    optional_list,
    output_length_constraint,
    conditional_labels=[],
):
    if (log_and_data_path / "dataset").exists():
        return
    output_annotator = VLLMPromptBasedOutputAnnotator(
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=TENSOR_SIZE,
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
        output_length_constraint=output_length_constraint,
        is_generation= False if conditional_labels != [] else True,
        log_and_data_path = log_and_data_path
    )
    # from IPython import embed
    # embed()
    all_outputs = datasets.load_from_disk(log_and_data_path/'output_recording')
    all_outputs = all_outputs.to_dict()

    output_col = output_dataset['output_col']
    if conditional_labels != []:
        mapping_function = partial(mapping_func, conditional_labels=conditional_labels)
        output_dataset = output_dataset.map(mapping_function)
        # TODO reverse experiments
        # print('reverse')
        # reverse_function = partial(reverse_func, conditional_labels=conditional_labels)
        # output_dataset = output_dataset.map(reverse_function)
        # print(output_dataset[0])
        # print(f"before label filtering {len(output_dataset)}")
        filter_func_partial = partial(filter_func, conditional_labels=conditional_labels)
        output_dataset = output_dataset.filter(filter_func_partial)
        # print(f"after label filtering {len(output_dataset)}")

        #! 强行将生成的 label 平衡起来
        output_col = output_dataset['output_col']
        element_count = Counter(output_col)
        min_count = min(element_count.values())
        balanced_data = {'input_col': [], 'output_col': []}
        class_samples = {label: [] for label in element_count.keys()}
        for input_val, output_val in zip(output_dataset['input_col'], output_dataset['output_col']):
            class_samples[output_val].append((input_val, output_val))
        for _ in range(min_count):
            for label in class_samples.keys():
                sample = random.choice(class_samples[label])
                # print(sample)
                balanced_data['input_col'].append(sample[0])
                balanced_data['output_col'].append(sample[1])
                class_samples[label].remove(sample)
        output_dataset = Dataset.from_dict(balanced_data)

        for output in output_col:
            if output in balanced_data["output_col"]:
                all_outputs['output'].append(output)
                all_outputs['drop_reason'].append('random chosen')
            else:
                all_outputs['output'].append(output)
                all_outputs['drop_reason'].append('random filtered')
    else:
        for output in output_col:
            all_outputs['output'].append(output)
            all_outputs['drop_reason'].append('random chosen')
    
    all_outputs_dataset = Dataset.from_dict(all_outputs)
    all_outputs_dataset.save_to_disk(log_and_data_path/'output_recording')
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
    with open(log_and_data_path / "config.json", "w") as f:
        json.dump(loaded_params, f, indent=4)
    del output_annotator
    destroy_model_parallel()
    gc.collect()
    torch.cuda.empty_cache()
