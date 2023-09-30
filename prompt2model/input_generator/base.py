"""An interface for generating inputs."""

from __future__ import annotations  # noqa FI58

from abc import ABC, abstractmethod

from transformers import AutoModelForCausalLM, AutoTokenizer

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
    def generate_inputs(self, num_examples: int, prompt_spec: PromptSpec) -> list[str]:
        """Generate new inputs for a given prompt.

        Args:
            num_examples: Expected number of inputs.
            prompt_spec: A parsed prompt spec.

        Returns:
            A list of new input strings.
        """
