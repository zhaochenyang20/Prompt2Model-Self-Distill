"""An interface for generating inputs."""

from __future__ import annotations  # noqa FI58

from abc import ABC, abstractmethod
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt2model.dataset_generator.prompt_based import Example
from prompt2model.prompt_parser import PromptSpec


class InputGenerator(ABC):
    """A class for generating inputs of examples from a prompt specification."""

    def __init__(self, pretrained_model_name: str) -> None:
        """Initializes OutputAnnotator with a pre-trained model.

        Args:
            pretrained_model_name: The name of a pre-trained
                middle-sized autoregressive model, ideally the
                same as the model of OutputAnnotator.
        """
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, padding_side="left"
        )

    @abstractmethod
    def generate_inputs(
        self,
        generated_examples: list[Example],
        prompt_spec: PromptSpec,
        inputs_num: int,
        hyperparameter_choices=dict[str, Any],
    ) -> list[str]:
        """Generate new inputs for a given prompt.

        Args:
            generated_examples: A list of currently generated examples.
            prompt_spec: A prompt we use to generate new inputs.
            inputs_num: The number of new inputs to generate.
            hyperparameter_choices: A dictionary of hyperparameter choices.

        Returns:
            A list of new input strings.
        """
