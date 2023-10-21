"""Import Input Generator classes."""
from prompt2model.output_annotator.base import OutputAnnotator
from prompt2model.output_annotator.mock import MockOutputAnnotator
from prompt2model.output_annotator.prompt_based import PromptBasedOutputAnnotator

__all__ = (
    "OutputAnnotator",
    "MockOutputAnnotator",
    "PromptBasedOutputAnnotator",
)
