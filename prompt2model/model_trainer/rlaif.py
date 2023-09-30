"""A trainer class to train generation models."""

from __future__ import annotations  # noqa FI58

import os
from typing import Any

import datasets

from prompt2model.model_trainer.base import BaseTrainer
from prompt2model.output_annotator import OutputAnnotator
from prompt2model.utils import get_formatted_logger

logger = get_formatted_logger("ModelTrainer")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class AnnotatorTrainer(BaseTrainer):
    """Trainer for the OutputAnnotator utilizing RLAIF and Lora."""

    def train_annotator(
        self,
        hyperparameter_choices: dict[str, Any],
        output_annotator: OutputAnnotator,
        training_dataset: datasets.Dataset,
    ):
        """Utilize RLAIF and Lora to train the OutputAnnotator.

        Args:
            hyperparameter_choices: A dictionary of hyperparameters for training.
            output_annotator: The OutputAnnotator being trained.
            training_dataset: Training dataset with `model_input`, `model_output`,
                and `score` columns from the QualityEvaluator.

        Returns:
            A trained OutputAnnotator.
        """
        _ = hyperparameter_choices, output_annotator, training_dataset
