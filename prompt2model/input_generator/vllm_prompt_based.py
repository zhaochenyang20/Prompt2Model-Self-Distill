"""Input Generator based on Prompts."""

import random
import re
from functools import partial
from typing import Any

import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams

from prompt2model.input_generator import InputGenerator
from prompt2model.utils.path import MODEL_PATH
from prompt2model.input_generator.prompt_template import (
    construct_meta_prompt,
    construct_verify_prompt,
)
from prompt2model.prompt_parser import PromptSpec
from prompt2model.quality_evaluator import (
    ablation_list_filter,
    get_middle_portion,
    min_max_length_filter,
)
from prompt2model.utils import count_tokens_from_string, get_formatted_logger

logger = get_formatted_logger("InputGenerator")
test = True


class VLLMPromptBasedInputGenerator(InputGenerator):
    """Generate inputs from prompts."""

    def __init__(
        self,
        pretrained_model_name: str = "lmsys/vicuna-7b-v1.5",
        gpu_memory_utilization=0.5,
        tensor_parallel_size=1,
    ) -> None:
        """Create a new instance of the HFPromptBasedInputGenerator.

        Args:
            pretrained_model_name: The name of a pre-trained decoder-only model.
            gpu_memory_utilization: The portion of CUDA memory to use on a single GPU.
            tensor_parallel_size: The number of GPUs to use for distributed execution.
        """
        if pretrained_model_name == "lmsys/vicuna-7b-v1.5":
            self.language_model = LLM(
                model=MODEL_PATH,
                gpu_memory_utilization=gpu_memory_utilization,
                tensor_parallel_size=tensor_parallel_size,
            )
        else:
            self.language_model = LLM(
                model=pretrained_model_name,
                gpu_memory_utilization=gpu_memory_utilization,
                tensor_parallel_size=tensor_parallel_size,
            )

    def construct_generation_prompt(
        self,
        instruction: str,
        few_shot_example_string: str,
        generated_inputs: list[str],
        context_cutoff: int = 3200,
        conditional_label: str = None,
    ) -> tuple[str]:
        """Generates a prompt string for generating a new input.

        Args:
            instruction: The natural language instruction for the prompt.
            few_shot_example_string: A string representing the few-shot examples
                parsed from the user's prompt, which quality is higher than the
                generated examples.
            generated_inputs: A list of currently generated inputs.
            context_cutoff: If the total length of the prompt in tokens exceeds this
                value, repeat the prompt generation process to generate a shorter one.
            conditional_label: The expected output for the new input. Used for
                classification tasks.

        Returns:
            The generated prompt string.
        """
        while True:
            # Choose a few inputs to add to the prompt if examples exist.
            if len(generated_inputs) == 0:
                low_quality_input_string = "N/A\n"
            else:
                low_quality_input_string = ""
                random_selected_generated_input_num = random.randint(
                    1, min(len(generated_inputs), 10)
                )
                random_inputs = random.sample(
                    generated_inputs, random_selected_generated_input_num
                )
                for input in random_inputs:
                    low_quality_input_string += f'[input]="{input}"\n\n'

            # Extract inputs from the few-shot examples.
            matches = re.findall(
                r'\[input\]="(.*?)"\s*\[output\]="(.*?)"',
                few_shot_example_string,
                re.DOTALL,
            )
            assert matches != []
            high_quality_inputs = [match[0] for match in matches]
            random.shuffle(high_quality_inputs)
            if len(high_quality_inputs) == 0:
                high_quality_input_string = "N/A\n"
            else:
                high_quality_input_string = ""
                for input in high_quality_inputs:
                    high_quality_input_string += f'[input]="{input}"\n\n'

            prompt = construct_meta_prompt(
                instruction=instruction,
                low_quality_input_string=low_quality_input_string,
                high_quality_input_string=high_quality_input_string,
                conditional_label=conditional_label,
            )

            if count_tokens_from_string(prompt, "vicuna") < context_cutoff:
                return (prompt, conditional_label)
            else:
                orginal_input_string = (
                    instruction + few_shot_example_string
                    if few_shot_example_string
                    else instruction
                )
                if (
                    count_tokens_from_string(orginal_input_string, "vicuna")
                    > context_cutoff
                ):
                    logger.warning(
                        "The original input prompt is too long. "
                        "Consider writing a shorter prompt."
                    )
                continue

    def generate_inputs(
        self,
        generated_inputs: list[str],
        prompt_spec: PromptSpec,
        inputs_num: int,
        hyperparameter_choices: dict[str, Any],
        conditional_labels: list = [],
    ) -> tuple[list[str]]:
        """Generate new inputs for a given prompt with a pre-trained model.

        Args:
            generated_inputs: A list of currently generated inputs.
            prompt_spec: A prompt we use to generate new inputs.
            inputs_num: The number of new inputs to generate.
            hyperparameter_choices: A dictionary of hyperparameter choices.
            conditional_labels: All the expected output labels for the new input.
                Used for classification tasks.
        """
        prompt_tuples = [
            self.construct_generation_prompt(
                instruction=prompt_spec.instruction,
                few_shot_example_string=prompt_spec.examples,
                generated_inputs=generated_inputs,
                context_cutoff=hyperparameter_choices.get("context_cutoff", 3000),
                conditional_label=random.choice(conditional_labels) if conditional_labels != [] else None,
            )
            for _ in range(inputs_num)
        ]
        # [(prompt, pseudo_label), ...]
        prompts = [each[0] for each in prompt_tuples]
        pseudo_labels = [each[1] for each in prompt_tuples]
        if conditional_labels != []:
            assert all(label in conditional_labels for label in pseudo_labels)
        else:
            assert all(label is None for label in pseudo_labels)

        sampling_params = SamplingParams(
            n=hyperparameter_choices.get("n", 1),
            best_of=hyperparameter_choices.get("best_of", 1),
            top_k=hyperparameter_choices.get("top_k", -1),
            temperature=hyperparameter_choices.get("temperature", 1),
            max_tokens=hyperparameter_choices.get("max_tokens", 500),
        )
        output_sequence = self.language_model.generate(prompts, sampling_params)
        new_inputs = [
            output.text for each in output_sequence for output in each.outputs
        ]
        return (new_inputs, pseudo_labels)

    def verify(
        self,
        prompt_spec: PromptSpec,
        new_inputs: list[str],
        labels: list[str] = [],
        expected_content: str = None,
        extraction_examples: str = None,
    ):
        """Check the generated inputs.

        Args:
            prompt_spec: A prompt we use to generate new inputs.
            new_inputs: The generated inputs.
            labels: The inputs' corresponding labels.
        """

        if new_inputs is None:
            return None

        def construct_filter_prompt(
            few_shot_example_string: str,
            new_input: str,
            label: str = None,
            instruction: str = None,
            extraction_examples: list[str] = None,
        ):
            matches = re.findall(
                r'\[input\]="(.*?)"\s*\[output\]="(.*?)"',
                few_shot_example_string,
                re.DOTALL,
            )
            assert matches != []
            high_quality_inputs = [match[0] for match in matches]
            high_quality_input_string = ""
            for input in high_quality_inputs:
                high_quality_input_string += f'"{input}"\n\n'
            if extraction_examples != []:
                extraction_example_string = ""
                for extraction_input, extraction_output in extraction_examples:
                    extraction_example_string += f"USER: {extraction_input}\n"
                    extraction_example_string += f"ASSISTANT: {extraction_output}\n"
                assert extraction_example_string != ""
            else:
                assert extraction_examples == [] and label is None
            return construct_verify_prompt(
                examples=high_quality_input_string,
                new_input=new_input,
                expected_content=expected_content,
                extraction_example_string=extraction_example_string
                if extraction_examples != []
                else None,
                label=label,
                instruction=instruction if label is not None else None,
            )

        filter_prompts = []
        for i in range(len(new_inputs)):
            assert (labels[i] is not None and extraction_examples != []) or (
                labels[i] is None and extraction_examples == []
            )
            prompt = construct_filter_prompt(
                prompt_spec.examples,
                new_inputs[i],
                labels[i],
                prompt_spec.instruction,
                extraction_examples,
            )
            filter_prompts.append(prompt.strip() if labels[i] is not None else prompt)
        sampling_params = SamplingParams(
            top_k=-1,
            top_p=1,
            temperature=0,
            max_tokens=500,
        )
        output_sequence = self.language_model.generate(filter_prompts, sampling_params)
        filtered_inputs = [
            output.text for each in output_sequence for output in each.outputs
        ]
        trancated_outputs = []
        for each in filtered_inputs:
            if "USER:" in each:
                trancated_outputs.append(each[: each.index("USER:")].strip())
            else:
                trancated_outputs.append(each.strip())
        return trancated_outputs

    def batch_generation_inputs(
        self,
        prompt_spec: PromptSpec,
        generation_epochs: int,
        per_epoch_num: int,
        hyperparameter_choices: dict[str, Any],
        expected_content,
        optional_list=[],
        intput_length_constraint=False,
        conditional_labels: list[str] = None,
        extraction_examples: list[(str, str)] = None,
        log_and_data_path: str = ''
    ) -> list[str]:
        """Generate new inputs for a given prompt with a pre-trained model.

        Args:
            generation_epochs: The number of epochs to generate inputs.
            prompt_spec: A prompt we use to generate new inputs.
            inputs_num: The number of new inputs to generate.
            hyperparameter_choices: A dictionary of hyperparameter choices.
        """

        def calculate_string_metrics(string_list):
            lengths = np.array([len(s) for s in string_list])
            mean_length = np.mean(lengths)
            std_dev = np.std(lengths)
            mean_plus_2std = mean_length + 2 * std_dev
            mean_minus_2std = mean_length - 2 * std_dev
            return mean_length, mean_plus_2std, mean_minus_2std

        ablation_filter = partial(ablation_list_filter, optional_list=optional_list)
        matches = re.findall(
            r'\[input\]="(.*?)"\s*\[output\]="(.*?)"',
            prompt_spec.examples,
            re.DOTALL,
        )
        assert matches != []
        _, mean_plus_2std, mean_minus_2std = calculate_string_metrics(
            [match[0] for match in matches]
        )
        # mean_plus_2std and mean_minus_2std of inputs
        # 要么要求 input 在  mean_plus_2std and mean_minus_2std 之内，要么不做要求
        # 不能只要求上界或者只要求下界
        length_filter = partial(
            min_max_length_filter,
            min_length=int(mean_minus_2std),
            max_length=int(mean_plus_2std),
        )
        generated_inputs = []
        # [(input, label)]
        for epoch in tqdm(range(generation_epochs)):
            input_tuples = self.generate_inputs(
                [each[0] for each in generated_inputs],
                prompt_spec,
                per_epoch_num,
                hyperparameter_choices,
                conditional_labels,
            )
            new_inputs = input_tuples[0]
            pseudo_labels = input_tuples[1]

            # record all the data
            last_part = log_and_data_path.split('/')[-1]
            task_name, temperature, intput_length_constraint, output_length_constraint, exp_number = last_part.split('_')
            print(epoch)
            ids = list(range(len(inputs)))

            data = {
                'task_name': [task_name]*len(input_tuples),  # Example task numbers
                'exp_number': [exp_number]*len(input_tuples),  # Example experiment numbers
                'id': ids,  # Auto-generated IDs
                'input': new_inputs,  
                'output': ['']*len(input_tuples),
                'drop_reason': ['']*len(input_tuples),  # Example drop reasons, None means no drop reason
                'task type': ['']*len(input_tuples)  # Example task types
            }


            input_to_label = dict(zip(new_inputs, pseudo_labels))
            filtered_new_inputs = [
                element
                for element in new_inputs
                if element is not None and element != ""
            ]
            filtered_new_inputs = ablation_filter(
                length_filter(filtered_new_inputs)
                if intput_length_constraint
                else filtered_new_inputs
            )
            if filtered_new_inputs is not None and filtered_new_inputs != []:
                filtered_pesudo_labels = [
                    input_to_label[input_item] for input_item in filtered_new_inputs
                ]
                verified_inputs = self.verify(
                    prompt_spec,
                    [each.strip() for each in filtered_new_inputs],
                    filtered_pesudo_labels,
                    expected_content=expected_content,
                    extraction_examples=extraction_examples,
                )

                assert len(filtered_new_inputs) == len(verified_inputs)
                filtered_input_to_label = dict(
                    zip(verified_inputs, filtered_pesudo_labels)
                )
                filtered_verified_inputs = [
                    element
                    for element in verified_inputs
                    if element is not None and element != ""
                ]
                filtered_verified_inputs = ablation_filter(
                    length_filter(filtered_verified_inputs)
                    if intput_length_constraint
                    else filtered_verified_inputs
                )

                if (
                    filtered_verified_inputs is not None
                    and filtered_verified_inputs != []
                ):
                    filtered_verified_labels = [
                        filtered_input_to_label[input_item]
                        for input_item in filtered_verified_inputs
                    ]
                    input_label_pairs = list(
                        zip(filtered_verified_inputs, filtered_verified_labels)
                    )
                    generated_inputs.extend(input_label_pairs)
                    unique_inputs = {}
                    filtered_generated_inputs = []
                    for input_item, label in generated_inputs:
                        if input_item not in unique_inputs:
                            unique_inputs[input_item] = label
                            filtered_generated_inputs.append(
                                (input_item.strip(), label)
                            )
                    generated_inputs = filtered_generated_inputs
        return generated_inputs
