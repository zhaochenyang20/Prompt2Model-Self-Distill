"""Utilities to construct an LLM "metaprompt" for our dataset generator."""
import random

PROMPT_TEMPLATE = """[INST]<<SYS>>
{META_PROMPT}
---------------------------------------------------------------------------------------------
Here are some [examples] you may refer to:

- Example 1

{example_1}

- Example 2

{example_2}
--------------------------------------------------------------------------------------------
Here is the [requirement] for the task:

[requirement] = {requirement}
---------------------------------------------------------------------------------------------
Here is the new input:

[input] = {input}
----------------------------------------------------------------------------------------------
Before generating your response, ensure that you strictly adhere to the rules mentioned in the [requirement] and follow the format of the [examples].
<</SYS>>
[output]=[/INST]"""  # noqa E501

META_PROMPT = """As a smart AI agent, your task is to generate a response based on the [requirement] and some [examples].

Try your best to ensure that the responses you generate are detailed, precise, comprehensive, and high-quality."""  # noqa E501


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
    example_1, example_2 = high_quality_input_string.split("\n\n")[0:2]
    return PROMPT_TEMPLATE.format(
        META_PROMPT=META_PROMPT,
        requirement=instruction,
        example_1=example_1,
        example_2=example_2,
        input=input,
    )
