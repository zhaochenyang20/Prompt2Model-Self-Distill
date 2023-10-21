"""A class for generating mock inputs (for testing purposes)."""

from prompt2model.prompt_parser import PromptSpec
from prompt2model.quality_evaluator.base import QualityEvaluator


class MockQualityEvaluator(QualityEvaluator):
    """A class for generating empty datasets (for testing purposes)."""

    def generate_inputs(self, num_examples: int, prompt_spec: PromptSpec) -> list[str]:
        """Generate mock inputs for a given prompt.

        Args:
            num_examples: Expected number of inputs.
            prompt_spec: A parsed prompt spec.

        Returns:
            A list of new input strings.
        """
        _ = prompt_spec
        return ["The mock input"] * num_examples
