"""Input Generator based on Prompts."""

import re
from functools import partial
from typing import Any

import datasets
import numpy as np
from vllm import LLM, SamplingParams

from prompt2model.output_annotator import OutputAnnotator
from prompt2model.output_annotator.prompt_template import construct_meta_prompt
from prompt2model.utils.path import MODEL_PATH
from prompt2model.prompt_parser import PromptSpec
from prompt2model.quality_evaluator import (
    ablation_list_filter,
    min_max_length_filter,
    self_consistency_filter,
)
from prompt2model.utils import count_tokens_from_string, get_formatted_logger
from IPython import embed

logger = get_formatted_logger("OutputAnnotator")

class VLLMPromptBasedOutputAnnotator(OutputAnnotator):
    """Generate outputs from prompts."""

    def __init__(
        self,
        pretrained_model_name: str = "lmsys/vicuna-7b-v1.5",
        gpu_memory_utilization=0.5,
        tensor_parallel_size=1,
    ) -> None:
        """Create a new instance of the VLLMPromptBasedOutputAnnotator.

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
                trust_remote_code=True,
            )
        else:
            self.language_model = LLM(
                model=pretrained_model_name,
                gpu_memory_utilization=gpu_memory_utilization,
                tensor_parallel_size=tensor_parallel_size,
                trust_remote_code=True,
            )

    def construct_prompt(
        self,
        instruction: str,
        few_shot_example_string: str,
        new_input: str,
        context_cutoff: int,
        is_generation: bool = True,
    ) -> str:
        """Generates a prompt string.

        Args:
            instruction: The natural language instruction for the prompt.
            few_shot_example_string: A string representing the few-shot examples
                parsed from the user's prompt, which quality is higher than the
                generated examples.
            context_cutoff: If the total length of the prompt in tokens exceeds this
                value, repeat the prompt generation process to generate a shorter one.

        Returns:
            The generated prompt string.
        """
        while True:
            # Construct the prompt.
            prompt = construct_meta_prompt(
                instruction=instruction,
                examples=few_shot_example_string,
                new_input=new_input,
                is_generation=is_generation,
            )
            if count_tokens_from_string(prompt, "vicuna") < context_cutoff:
                return prompt.strip()
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

    def annotate_outputs(
        self,
        input_strings: list[str],
        prompt_spec: PromptSpec,
        hyperparameter_choices: dict[str, Any],
        optional_list=[],
        output_length_constraint=False,
        is_generation: bool = True,
        log_and_data_path: str = ''
    ) -> datasets.Dataset:
        """Generate candidate outputs for each given input.

        Args:
            input_strings: A list of input strings from InputGenerator.
            prompt_spec: A parsed prompt spec.

        Returns:
            A dataset of `input_col` and `output_col`.
        """

        def calculate_string_metrics(string_list):
            # Calculate the lengths of each string
            lengths = np.array([len(s) for s in string_list])
            # Calculate mean and standard deviation
            mean_length = np.mean(lengths)
            std_dev = np.std(lengths)
            # Calculate mean ± 2σ
            mean_plus_2std = mean_length + 2 * std_dev
            mean_minus_2std = mean_length - 2 * std_dev

            return mean_length, mean_plus_2std, mean_minus_2std

        matches = re.findall(
            r'\[input\]="(.*?)"\s*\[output\]="(.*?)"',
            prompt_spec.examples,
            re.DOTALL,
        )
        assert matches != []
        _, mean_plus_2std, mean_minus_2std = calculate_string_metrics(
            [match[1] for match in matches]
        )

        prompts = []
        consistency_filter = partial(
            self_consistency_filter,
            min_frequency=hyperparameter_choices.get("min_frequency", 0.2),
        )
        length_filter = partial(
            min_max_length_filter,
            min_length=int(mean_minus_2std),
            max_length=int(mean_plus_2std),
        )
        ablation_filter = partial(ablation_list_filter, optional_list=optional_list)

        all_outputs = {
            'output': [],
            'drop_reason': [],
        }

        for input in input_strings:
            prompts += [
                self.construct_prompt(
                    prompt_spec.instruction,
                    prompt_spec.examples,
                    new_input=input,
                    context_cutoff=3000,
                    is_generation=is_generation,
                )
            ]
        sampling_params = SamplingParams(
            n=hyperparameter_choices.get("n", 10),
            best_of=hyperparameter_choices.get("best_of", 20),
            top_k=hyperparameter_choices.get("top_k", 10),
            temperature=hyperparameter_choices.get("temperature", 0.2),
            max_tokens=hyperparameter_choices.get("max_tokens", 500),
        )
        output_sequence = self.language_model.generate(prompts, sampling_params)
        input_cols = []
        output_cols = []
        for idx, input in enumerate(input_strings):
            outputs = []
            for output in output_sequence[idx].outputs:
                if (output.text is not None and output.text != ""):
                    outputs.append(output.text.strip())
                else:
                    all_outputs["drop_reason"].append('empty output')
                    all_outputs["output"].append(output)
            trancated_outputs = [each.strip() for each in outputs]
            # 3 filters: consistency_filter, ablation_filter, length_filter
            # length filter and ablation filter
            outputs = []
            for output in trancated_outputs:
                if output_length_constraint and not length_filter([output]):
                    all_outputs["drop_reason"].append('output length constrain')
                    all_outputs["output"].append(output)
                    continue
                if not ablation_filter([output]):
                    all_outputs["drop_reason"].append('ablation filter')
                    all_outputs["output"].append(output)
                    continue
                outputs.append(output)

            # consistency_filter
            consistent_output = consistency_filter(outputs)
            for output in outputs:
                if output != consistent_output:
                    all_outputs["drop_reason"].append('consistent filter')
                    all_outputs["output"].append(output)
 
            if (
                consistent_output is not None
                and consistent_output != ""
                and isinstance(consistent_output, str)
            ):
                input_cols.append(input)
                output_cols.append(consistent_output)
        
        data_path = log_and_data_path / "output_recording"
        outputs = datasets.Dataset.from_dict(all_outputs)
        outputs.save_to_disk(data_path)

        return datasets.Dataset.from_dict(
            dict(input_col=input_cols, output_col=output_cols)
        ).shuffle()
