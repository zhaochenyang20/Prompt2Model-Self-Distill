"""Import Input Generator classes."""
from prompt2model.output_annotator.base import OutputAnnotator
from prompt2model.output_annotator.mock import MockOutputAnnotator
from prompt2model.output_annotator.prompt_template import construct_meta_prompt
from prompt2model.output_annotator.vllm_prompt_based import (
    VLLMPromptBasedOutputAnnotator,
)

__all__ = (
    "OutputAnnotator",
    "MockOutputAnnotator",
    "VLLMPromptBasedOutputAnnotator",
    "construct_meta_prompt",
)
