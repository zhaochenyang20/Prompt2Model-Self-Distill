"""Input Generator based on Prompts."""

from typing import Any

import datasets
from vllm import LLM, SamplingParams

from prompt2model.output_annotator import OutputAnnotator
from prompt2model.output_annotator.prompt_template import construct_meta_prompt
from prompt2model.prompt_parser import PromptSpec
from prompt2model.quality_evaluator import (
    ablation_list_filter,
    check_paragraph_coherence,
    self_consistency_filter,
)
from prompt2model.utils import count_tokens_from_string, get_formatted_logger

logger = get_formatted_logger("OutputAnnotator")


class VLLMPromptBasedOutputAnnotator(OutputAnnotator):
    """Generate outputs from prompts."""

    def __init__(
        self,
        pretrained_model_name: str = "lmsys/vicuna-7b-v1.5",
        gpu_memory_utilization=0.5,
    ) -> None:
        """Create a new instance of the VLLMPromptBasedOutputAnnotator.

        Args:
            pretrained_model_name: The name of a pre-trained decoder-only model.
            gpu_memory_utilization: The portion of CUDA memory to use on a single GPU.
        """
        if pretrained_model_name == "lmsys/vicuna-7b-v1.5":
            self.language_model = LLM(
                model="/data/ckpts/huggingface/models/models--lmsys--vicuna-7b-v1.5/snapshots/de56c35b1763eaae20f4d60efd64af0a9091ebe5"
            , gpu_memory_utilization=gpu_memory_utilization)
        else:
            self.language_model = LLM(model=pretrained_model_name, gpu_memory_utilization=gpu_memory_utilization)

    def construct_prompt(
        self,
        instruction: str,
        few_shot_example_string: str,
        new_input: str,
        context_cutoff: int,
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

    def annotate_outputs(
        self,
        input_strings: list[str],
        prompt_spec: PromptSpec,
        hyperparameter_choices: dict[str, Any],
    ) -> datasets.Dataset:
        """Generate candidate outputs for each given input.

        Args:
            input_strings: A list of input strings from InputGenerator.
            prompt_spec: A parsed prompt spec.

        Returns:
            A dataset of `input_col` and `output_col`.
        """
        prompts = []
        for input in input_strings:
            prompts += [
                self.construct_prompt(
                    prompt_spec.instruction,
                    prompt_spec.examples,
                    new_input=input,
                    context_cutoff=3500,
                )
            ]

        sampling_params = SamplingParams(
            n=hyperparameter_choices.get("n", 10),
            # do_sample=hyperparameter_choices.get("do_sample", True),
            best_of=hyperparameter_choices.get("best_of", 20),
            top_k=hyperparameter_choices.get("top_k", 10),
            temperature=hyperparameter_choices.get("temperature", 0.2),
            max_tokens=hyperparameter_choices.get("max_tokens", 500),
        )
        output_sequence = self.language_model.generate(prompts, sampling_params)
        
        output_strings = [
            self_consistency_filter(
                check_paragraph_coherence(
                    ablation_list_filter([output.text for output in each.outputs])
                )
            )
            for each in output_sequence
        ]
        return datasets.Dataset.from_dict(
            dict(input_col=input_strings, output_col=output_strings)
        )
