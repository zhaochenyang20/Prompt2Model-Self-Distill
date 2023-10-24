"""An interface for generating inputs."""

from __future__ import annotations  # noqa FI58

from abc import ABC, abstractmethod
from typing import Any

import datasets

from prompt2model.prompt_parser import PromptSpec


class QualityEvaluator(ABC):
    """A class for evaluate generated exmaples and filter high-quality examples."""

    def filter_dataset(
        self,
        prompt_spec: PromptSpec,
        annotated_dataset: datasets.Dataset,
        hyperparameter_choices: dict[str, Any],
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
        _ = prompt_spec, annotated_dataset, hyperparameter_choices
