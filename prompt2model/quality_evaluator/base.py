"""An interface for generating inputs."""

from __future__ import annotations  # noqa FI58

from abc import ABC, abstractmethod

import datasets

from prompt2model.prompt_parser import PromptSpec


class QualityEvaluator(ABC):
    """A class for generating inputs of examples from a prompt specification."""

    @abstractmethod
    def evaluate_input_output_pairs(
        self, prompt_spec: PromptSpec, annotated_dataset: datasets.Dataset
    ) -> datasets.Dataset:
        """Generate new inputs for a given prompt.

        Args:
            prompt_spec: A parsed prompt spec.
            annotated_dataset: Annotated dataset with `model_input`
                and `model_output` columns.

        Returns:
            A dataset with `model_input`, `model_output`, and `score`
            columns, where `score` is a scaled float for RLAIF training.
        """
