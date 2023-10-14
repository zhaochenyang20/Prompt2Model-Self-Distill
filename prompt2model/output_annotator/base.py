"""An interface for generating inputs."""

from __future__ import annotations  # noqa FI58

from abc import ABC, abstractmethod

import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt2model.prompt_parser import PromptSpec


class OutputAnnotator(ABC):
    """A class for annotating outputs for each given input."""

    def __init__(self, pretrained_model_name: str) -> None:
        """Initializes OutputAnnotator with a pre-trained model.

        Args:
            pretrained_model_name: The name of a pre-trained
                middle-sized autoregressive model, ideally the
                same as the model of InputGenerator.
        """
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, padding_side="left"
        )

    @abstractmethod
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
