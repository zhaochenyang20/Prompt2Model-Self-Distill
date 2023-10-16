"""Utilities to construct an LLM "metaprompt" for our dataset generator."""

PROMPT_TEMPLATE = """[INST]<<SYS>>
{META_PROMPT}
--------------------------------------------------------------------------------------------
Here is the requirement for the task:

[requirement]:
{requirement}
---------------------------------------------------------------------------------------------
Here are some [examples] for the [requirement]. These examples can provide you with very strict format requirements. You should pay extreme attention to them!!!

[examples]:
{examples}
---------------------------------------------------------------------------------------------
<</SYS>>
Here is the new input:

[input]:
{input}
----------------------------------------------------------------------------------------------
Before generating your response, ensure that you strictly adhere to the rules mentioned in the [requirement] and follow the format of the [examples].

[output]:
[/INST]
"""  # noqa E501

META_PROMPT = """As a smart AI agents, your task is to generate a response based on the [requirement] and some [examples].

Try you best to ensure that the response you generate are detailed, precise, comprehensive, and high-quality."""  # noqa E501


def construct_meta_prompt(
    instruction: str = None,
    input: str = None,
    high_quality_input_string: str = None,
) -> str:
    """Constructs a prompt template for the dataset generator.

    Args:
        instruction: The natural language instruction for the prompt.
        input: A new input to be annotated.
        high_quality_input_string: A string representing the high quality examples.
    """
    return PROMPT_TEMPLATE.format(
        META_PROMPT=META_PROMPT,
        requirement=instruction,
        examples=high_quality_input_string,
        input=input,
    )
