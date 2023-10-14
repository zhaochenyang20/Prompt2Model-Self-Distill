"""A class for generating mock inputs (for testing purposes)."""

from typing import Any

from prompt2model.dataset_generator.prompt_based import Example
from prompt2model.input_generator.base import InputGenerator
from prompt2model.prompt_parser import PromptSpec


class MockInputGenerator(InputGenerator):
    """A class for generating empty datasets (for testing purposes)."""

    def generate_inputs(
        self,
        generated_examples: list[Example],
        prompt_spec: PromptSpec,
        hyperparameter_choices=dict[str, Any],
    ) -> list[str]:
        """Generate mock inputs for a given prompt.

        Args:
            num_examples: Expected number of inputs.
            prompt_spec: A parsed prompt spec.

        Returns:
            A list of new input strings.
        """
        _ = prompt_spec, generated_examples, hyperparameter_choices
        return ["The mock input"] * hyperparameter_choices.get(
            "num_return_sequences", 5
        )
