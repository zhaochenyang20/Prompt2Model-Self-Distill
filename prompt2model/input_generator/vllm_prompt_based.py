"""Input Generator based on Prompts."""

import random
import re
from typing import Any

from tqdm import tqdm
from vllm import LLM, SamplingParams

from prompt2model.input_generator import InputGenerator
from prompt2model.input_generator.prompt_template import (
    construct_meta_prompt,
    construct_verify_prompt,
)
from prompt2model.prompt_parser import PromptSpec
from prompt2model.quality_evaluator import ablation_list_filter
from prompt2model.quality_evaluator.length_filter import length_filter
from prompt2model.utils import count_tokens_from_string, get_formatted_logger

logger = get_formatted_logger("InputGenerator")


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
                model="/data/ckpts/huggingface/models/models--lmsys--vicuna-7b-v1.5/snapshots/de56c35b1763eaae20f4d60efd64af0a9091ebe5",
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
    ) -> str:
        """Generates a prompt string for generating a new input.

        Args:
            instruction: The natural language instruction for the prompt.
            few_shot_example_string: A string representing the few-shot examples
                parsed from the user's prompt, which quality is higher than the
                generated examples.
            generated_inputs: A list of currently generated inputs.
            context_cutoff: If the total length of the prompt in tokens exceeds this
                value, repeat the prompt generation process to generate a shorter one.

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
            high_quality_inputs = [match[0] for match in matches]
            random.shuffle(high_quality_inputs)
            if len(high_quality_inputs) == 0:
                high_quality_input_string = "N/A\n"
            else:
                high_quality_input_string = ""
                for input in high_quality_inputs:
                    high_quality_input_string += f'[input]="{input}"\n\n'

            # Construct the prompt.
            prompt = construct_meta_prompt(
                instruction=instruction,
                low_quality_input_string=low_quality_input_string,
                high_quality_input_string=high_quality_input_string,
            )
            if count_tokens_from_string(prompt) < context_cutoff:
                return prompt
            else:
                orginal_input_string = (
                    instruction + few_shot_example_string
                    if few_shot_example_string
                    else instruction
                )
                if count_tokens_from_string(orginal_input_string) > context_cutoff:
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
    ) -> list[str]:
        """Generate new inputs for a given prompt with a pre-trained model.

        Args:
            generated_inputs: A list of currently generated inputs.
            prompt_spec: A prompt we use to generate new inputs.
            inputs_num: The number of new inputs to generate.
            hyperparameter_choices: A dictionary of hyperparameter choices.
        """
        prompts = [
            self.construct_generation_prompt(
                instruction=prompt_spec.instruction,
                few_shot_example_string=prompt_spec.examples,
                generated_inputs=generated_inputs,
                context_cutoff=hyperparameter_choices.get("context_cutoff", 3500),
            )
            for _ in range(inputs_num)
        ]
        sampling_params = SamplingParams(
            n=hyperparameter_choices.get("n", 1),
            # repetition_penalty=hyperparameter_choices.get("repetition_penalty", 2),
            # do_sample=hyperparameter_choices.get("do_sample", True),
            best_of=hyperparameter_choices.get("best_of", 1),
            top_k=hyperparameter_choices.get("top_k", -1),
            temperature=hyperparameter_choices.get("temperature", 1),
            max_tokens=hyperparameter_choices.get("max_tokens", 500),
        )
        output_sequence = self.language_model.generate(prompts, sampling_params)
        new_inputs = [
            output.text for each in output_sequence for output in each.outputs
        ]
        return new_inputs

    def verify(self, prompt_spec: PromptSpec, new_inputs: list[str], expected_content):
        """Check the generated inputs.

        Args:
            prompt_spec: A prompt we use to generate new inputs.
            new_inputs: The generated inputs.
        """

        def construct_filter_prompt(
            few_shot_example_string: str,
            new_input: str,
        ):
            matches = re.findall(
                r'\[input\]="(.*?)"\s*\[output\]="(.*?)"',
                few_shot_example_string,
                re.DOTALL,
            )
            high_quality_inputs = [match[0] for match in matches]
            high_quality_input_string = ""
            for input in high_quality_inputs:
                high_quality_input_string += f'"{input}"\n\n'
            return construct_verify_prompt(
                examples=high_quality_input_string,
                new_input=new_input,
                expected_content=expected_content,
            )

        if new_inputs is None:
            return None
        filter_prompts = [
            construct_filter_prompt(prompt_spec.examples, each) for each in new_inputs
        ]
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
        return filtered_inputs

    def batch_generation_inputs(
        self,
        prompt_spec: PromptSpec,
        generation_epochs: int,
        per_epoch_num: int,
        hyperparameter_choices: dict[str, Any],
        expected_content,
    ) -> list[str]:
        """Generate new inputs for a given prompt with a pre-trained model.

        Args:
            generation_epochs: The number of epochs to generate inputs.
            prompt_spec: A prompt we use to generate new inputs.
            inputs_num: The number of new inputs to generate.
            hyperparameter_choices: A dictionary of hyperparameter choices.
        """
        generated_inputs = []
        for _ in tqdm(range(generation_epochs)):
            new_inputs = self.generate_inputs(
                generated_inputs, prompt_spec, per_epoch_num, hyperparameter_choices
            )
            new_inputs = [
                element
                for element in new_inputs
                if element is not None and element != ""
            ]
            filtered_inputs = self.verify(
                prompt_spec,
                ablation_list_filter(
                    length_filter(
                        new_inputs, hyperparameter_choices.get("min_input_length", 120)
                    )
                ),
                expected_content=expected_content,
            )
            if filtered_inputs is not None:
                generated_inputs.extend(filtered_inputs)
                generated_inputs = list(set(generated_inputs))
        return generated_inputs
