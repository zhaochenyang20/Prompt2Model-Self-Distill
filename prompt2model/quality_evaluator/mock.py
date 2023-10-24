"""A class for generating mock inputs (for testing purposes)."""

from typing import Any

import datasets

from prompt2model.prompt_parser import PromptSpec
from prompt2model.quality_evaluator.base import QualityEvaluator


class MockQualityEvaluator(QualityEvaluator):
    """A class for generating empty datasets (for testing purposes)."""

    def evaluate_input_output_pairs(
        self,
        prompt_spec: PromptSpec,
        annotated_dataset: datasets.Dataset,
        hyperparameter_choices: dict[str, Any],
    ) -> datasets.Dataset:
        """Generate mock inputs for a given prompt.

        Args:
            num_examples: Expected number of inputs.
            prompt_spec: A parsed prompt spec.

        Returns:
            A list of new input strings.
        """
        _ = prompt_spec
        return ["The mock input"] * num_examples
