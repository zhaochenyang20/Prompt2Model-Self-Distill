"""Input Generator based on Prompts."""

import random
import re
from typing import Any

import torch
from optimum.bettertransformer import BetterTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt2model.dataset_generator.prompt_based import Example
from prompt2model.input_generator import InputGenerator
from prompt2model.input_generator.prompt_template import construct_meta_prompt
from prompt2model.prompt_parser import PromptSpec
from prompt2model.utils import count_tokens_from_string, get_formatted_logger

logger = get_formatted_logger("InputGenerator")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PromptBasedInputGenerator(InputGenerator):
    """Generate inputs from prompts."""

    def __init__(
        self,
        pretrained_model_name: str = "lmsys/vicuna-7b-v1.5",
    ) -> None:
        """Create a new instance of the PromptBasedInputGenerator.

        Args:
            pretrained_model_name: The name of a pre-trained decoder-only model.
        """
        if pretrained_model_name == "lmsys/vicuna-7b-v1.5":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "/home/cyzhao/.vicuna-7b-1.5",
                local_files_only=True,
                padding_side='left',
                trust_remote_code=True,
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name,  padding_side='left')
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            use_flash_attention_2=True,
        )
        self.model = BetterTransformer.transform(model, keep_original_model=True)

    def construct_prompt(
        self,
        instruction: str,
        few_shot_example_string: str,
        generated_examples: list[Example],
        context_cutoff: int,
    ) -> str:
        """Generates a prompt string.

        Args:
            instruction: The natural language instruction for the prompt.
            few_shot_example_string: A string representing the few-shot examples
                parsed from the user's prompt, which quality is higher than the
                generated examples.
            generated_examples: A list of currently generated examples.
            context_cutoff: If the total length of the prompt in tokens exceeds this
                value, repeat the prompt generation process to generate a shorter one.

        Returns:
            The generated prompt string.
        """
        while True:
            # Choose a few inputs to add to the prompt if examples exist.
            if len(generated_examples) == 0:
                low_quality_input_string = "N/A\n"
                random_selected_generated_example_num = 0
            else:
                low_quality_input_string = ""
                random_selected_generated_example_num = random.randint(
                    1, min(len(generated_examples), 10)
                )
                random_examples = random.sample(
                    generated_examples, random_selected_generated_example_num
                )
                for example in random_examples:
                    low_quality_input_string += f'[input]="{example.input_col}"\n'

            # Extract inputs from the few-shot examples.
            high_quality_inputs = re.findall(r'input="(.*?)"', few_shot_example_string)
            if len(high_quality_inputs) == 0:
                high_quality_input_string = "N/A\n"
            else:
                high_quality_input_string = ""
                for input in high_quality_inputs:
                    high_quality_input_string += f'[input]="{input}"\n'

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

    def generate_input(
        self,
        prompt: str,
        hyperparameter_choices: dict[str, Any],
    ) -> list[str]:
        """Generate new inputs for a given prompt.

        Args:
            prompt: A prompt we use to generate new inputs.
            hyperparameter_choices: A dictionary of hyperparameter choices.
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)
        output_sequences = self.model.generate(
            input_ids=input_ids,
            do_sample=True,
            top_k=hyperparameter_choices.get("top_k", 50),
            num_return_sequences=hyperparameter_choices.get("num_return_sequences", 5),
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=hyperparameter_choices.get("max_new_tokens", 400),
            temperature=hyperparameter_choices.get("temperature", 0.7),
        )

        generated_strings = [
            self.tokenizer.decode(
                generated_sequence.tolist(), clean_up_tokenization_spaces=True
            )
            for generated_sequence in output_sequences
        ]

        def extract_tail(a, b):
            start_index = b.find(a)
            if start_index == -1 or start_index == 0:
                return ""
            end_index = start_index + len(a)
            encoded = self.tokenizer.encode(b[end_index:])
            return self.tokenizer.decode(
                encoded, clean_up_tokenization_spaces=True, skip_special_tokens=True
            )

        generated_inputs = [extract_tail(prompt, each) for each in generated_strings]
        return generated_inputs

    def generate_inputs(
        self,
        generated_examples: list[Example],
        prompt_spec: PromptSpec,
        inputs_num: int,
        hyperparameter_choices: dict[str, Any],
    ) -> list[str]:
        """Generate new inputs for a given prompt with a pre-trained model.

        Args:
            generated_examples: A list of currently generated examples.
            prompt_spec: A prompt we use to generate new inputs.
            inputs_num: The number of new inputs to generate.
            hyperparameter_choices: A dictionary of hyperparameter choices.
        """
        generated_inputs = []
        prompts_list = [
            self.construct_prompt(
                instruction=prompt_spec.instruction,
                few_shot_example_string=prompt_spec.examples,
                generated_examples=generated_examples,
                context_cutoff=hyperparameter_choices.get("context_cutoff", 3500),
            )
            for _ in range(inputs_num)
        ]
        for start_idx in range(
            0, inputs_num, hyperparameter_choices.get("batch_size", 5)
        ):
            end_idx = min(
                start_idx + hyperparameter_choices.get("batch_size", 5), inputs_num
            )
            batched_prompts = prompts_list[start_idx:end_idx]
            encoded_inputs = self.tokenizer.batch_encode_plus(
                batched_prompts,
                truncation=True,
                max_length=hyperparameter_choices.get("tokenizer_max_length", 3500),
                padding=True,
                return_tensors="pt",
            )
            device = self.model.device
            input_ids = encoded_inputs["input_ids"].to(device)
            attention_mask = encoded_inputs["attention_mask"].to(device)
            output_sequences = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                top_k=hyperparameter_choices.get("top_k", 50),
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=hyperparameter_choices.get("max_new_tokens", 400),
                temperature=hyperparameter_choices.get("temperature", 0.7),
            )
            generated_strings = [
                self.tokenizer.decode(
                    generated_sequence.tolist(), clean_up_tokenization_spaces=True
                )
                for generated_sequence in output_sequences
            ]

            def extract_tail(a, b):
                start_index = b.find(a)
                if start_index == -1 or start_index == 0:
                    return ""
                end_index = start_index + len(a)
                encoded = self.tokenizer.encode(b[end_index:])
                return self.tokenizer.decode(
                    encoded, clean_up_tokenization_spaces=True, skip_special_tokens=True
                )

            generated_inputs += [
                extract_tail(batched_prompts[idx], each)
                for (idx, each) in enumerate(generated_strings)
            ]
        return generated_inputs
