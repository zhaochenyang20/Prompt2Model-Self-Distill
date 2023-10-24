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
        inputs_num: int,
        hyperparameter_choices=dict[str, Any],
    ) -> list[str]:
        """Generate new inputs for a given prompt with a pre-trained model.

        Args:
            generated_examples: A list of currently generated examples.
            prompt_spec: A prompt we use to generate new inputs.
            inputs_num: The number of new inputs to generate.
            hyperparameter_choices: A dictionary of hyperparameter choices.
        """
        _ = prompt_spec, generated_examples, hyperparameter_choices
        return ["The mock input"] * inputs_num
