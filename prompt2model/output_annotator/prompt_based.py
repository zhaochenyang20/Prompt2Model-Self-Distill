"""Input Generator based on Prompts."""

import re
from typing import Any

import datasets
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from prompt2model.output_annotator import OutputAnnotator
from prompt2model.output_annotator.prompt_template import construct_meta_prompt
from prompt2model.prompt_parser import PromptSpec
from prompt2model.utils import count_tokens_from_string, get_formatted_logger

logger = get_formatted_logger("OutputAnnotator")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PromptBasedOutputAnnotator(OutputAnnotator):
    """Generate inputs from prompts."""

    def __init__(
        self,
        peft_model_id: str = None,
        pretrained_model_name: str = "meta-llama/Llama-2-7b-chat-hf",
    ) -> None:
        """Create a new instance of the PromptBasedOutputAnnotator.

        Args:
            peft_model_id: The path of lora.
            pretrained_model_name: The name of a pre-trained decoder-only model.
        """
        if peft_model_id is not None:
            config = PeftConfig.from_pretrained(peft_model_id)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                config.base_model_name_or_path
            )
            self.model = PeftModel.from_pretrained(model, peft_model_id).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.base_model_name_or_path
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                pretrained_model_name, torch_dtype=torch.float16, trust_remote_code=True
            ).to(device)

    def construct_prompt(
        self,
        instruction: str,
        input: str,
        few_shot_example_string: str,
        context_cutoff: int,
    ) -> str:
        """Generates a prompt string.

        Args:
            instruction: The natural language instruction for the prompt.
            input: A new input to be annotated.
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
                input=input,
                high_quality_input_string=few_shot_example_string,
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

    def annotate_single_input(
        self,
        input: str,
        prompt_spec: PromptSpec,
        hyperparameter_choices=dict[str, Any],
    ) -> list[str]:
        """Generate new inputs for a given prompt with a pre-trained model.

        Args:
            input: A new input to be annotated.
            prompt_spec: A prompt we use to generate new inputs.
            hyperparameter_choices: A dictionary of hyperparameter choices.
        """
        prompt = self.construct_prompt(
            instruction=prompt_spec.instruction,
            input=input,
            few_shot_example_string=prompt_spec.examples,
            context_cutoff=hyperparameter_choices.get("context_cutoff", 3500),
        )
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)
        output_sequences = self.model.generate(
            input_ids=input_ids,
            do_sample=True,
            top_k=hyperparameter_choices.get("top_k", 10),
            num_return_sequences=hyperparameter_choices.get("num_candidate_outputs", 5),
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=hyperparameter_choices.get("max_new_tokens", 400),
            temperature=hyperparameter_choices.get("temperature", 0.7),
        )

        def extract_new_input(input_string: str) -> str:
            # Extract the sentence after "[new input]:"
            matched = re.search(
                r"\[output\]:\s*(?:\n)?(.*?)(?=\n|</s>|<unk>|$)", input_string
            )
            extracted_sentence = matched.group(1) if matched else ""
            # Clean the extracted sentence by removing unwanted tokens
            cleaned_sentence = (
                extracted_sentence.replace("</s>", "").replace("<unk>", "").strip()
            )
            return cleaned_sentence

        annotated_outputs = [
            self.tokenizer.decode(
                generated_sequence.tolist(), clean_up_tokenization_spaces=True
            )
            for generated_sequence in output_sequences
        ]
        return [extract_new_input(each) for each in annotated_outputs]

    def annotate_outputs(
        self,
        input_strings: list[str],
        num_candidate_outputs: int,
        prompt_spec: PromptSpec,
    ) -> datasets.Dataset:
        """Generate candidate outputs for each given input.

        Args:
            input_strings: A list of input strings from InputGenerator.
            num_candidate_outputs: Number of candidate outputs for
                each input in input_strings.
            prompt_spec: A parsed prompt spec.

        Returns:
            A dictionary mapping input strings to a list of candidate
                outputs, i.e. dict[str, list[str]].
        """
        outputs_dict = {}
        for input in input_strings:
            outputs = self.annotate_single_input(
                input=input,
                prompt_spec=prompt_spec,
                hyperparameter_choices=dict(
                    num_candidate_outputs=num_candidate_outputs,
                    context_cutoff=3500,
                    top_k=10,
                    max_new_tokens=400,
                    temperature=0.7,
                ),
            )
            outputs_dict[input] = outputs
        return datasets.Dataset.from_dict(outputs_dict)
