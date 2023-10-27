"""Import Input Generator classes."""
from prompt2model.output_annotator.base import OutputAnnotator
from prompt2model.output_annotator.mock import MockOutputAnnotator
from prompt2model.output_annotator.prompt_based import HFPromptBasedOutputAnnotator
from prompt2model.output_annotator.vllm_prompt_based import VLLMPromptBasedOutputAnnotator
from prompt2model.output_annotator.prompt_template import construct_meta_prompt

__all__ = (
    "OutputAnnotator",
    "MockOutputAnnotator",
    "HFPromptBasedOutputAnnotator",
    "VLLMPromptBasedOutputAnnotator",
    "construct_meta_prompt",
)
