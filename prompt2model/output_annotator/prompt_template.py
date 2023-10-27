"""Utilities to construct an LLM "metaprompt" for our dataset generator."""

PROMPT_TEMPLATE = """### Instructions:

{instruction}

### Examples:

{examples}

### New Input:

{new_input}

[your output]="""  # noqa E501


def construct_meta_prompt(
    instruction: str = None,
    examples: str = None,
    new_input: str = None,
) -> str:
    """Constructs a prompt template for the dataset generator.

    Args:
        instruction: The natural language instruction for the prompt.
        input: A new input to be annotated.
        high_quality_input_string: A string representing the high quality examples.
    """
    if not new_input.startswith("[input]="):
        new_input = "[input]=" + new_input
    return PROMPT_TEMPLATE.format(
        instruction=instruction,
        new_input=new_input,
        examples=examples,
    )
