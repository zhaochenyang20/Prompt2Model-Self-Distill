"""Import Input Generator classes."""
from prompt2model.input_generator.base import InputGenerator
from prompt2model.input_generator.mock import MockInputGenerator
from prompt2model.input_generator.prompt_based import PromptBasedInputGenerator

__all__ = (
    "PromptBasedInputGenerator",
    "InputGenerator",
    "MockInputGenerator",
)
