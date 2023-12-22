"""Utilities to construct an LLM "metaprompt" for our dataset generator."""

PROMPT_TEMPLATE_OLD = """
### Instructions:

{instruction}

### Examples:

{examples}

### New Input:

{new_input}

### Your Output:

"""  # noqa E501

PROMPT_TEMPLATE = """
A chat between a curious user and an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers to the user's questions.
USER: The artificial intelligence assistant only needs to help annotate label. The task is: {instruction} 
ASSISTANT: Okay. 
USER: [input] = "claim: Music containing lyrics that glorify violent and criminal lifestyles should be banned. perspective: hip hop artists have a right to free speech."
ASSISTANT: undermine.
USER: [input] = "claim: Abolish the US Electoral College. perspective: The electoral college weakens incentives for voting and party building."
ASSISTANT: support
USER: [input] = {new_input}
ASSISTANT: 
"""  # noqa E501


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
    return PROMPT_TEMPLATE.format(
        instruction=instruction,
        new_input=new_input,
        examples=examples,
    )
