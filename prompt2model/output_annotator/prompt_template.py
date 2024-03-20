"""Utilities to construct an LLM "metaprompt" for our dataset generator."""

import re

GENERATION_PROMPT_TEMPLATE = """
### Instructions:

{instruction}

### Examples:

{examples}{few_shots_prompt}

### New Input:

{new_input}

### Your Output:

"""  # noqa E501

CLASSIFICATION_PROMPT_TEMPLATE = """
A chat between a curious user and an artificial intelligence assistant.
The assistant gives concise answers to the user's questions.
USER: The artificial intelligence assistant only needs to help annotate label. The task is: {instruction}
ASSISTANT: Okay.
{examples}{few_shots_prompt}
USER: [input] = {new_input}
ASSISTANT:
"""  # noqa E501

def construct_meta_prompt(
    instruction: str = None,
    examples: str = None,
    new_input: str = None,
    is_generation: bool = True,
    few_shots_prompt: str = None,
) -> str:
    """Constructs a prompt template for the dataset generator.

    Args:
        instruction: The natural language instruction for the prompt.
        input: A new input to be annotated.
        high_quality_input_string: A string representing the high quality examples.
    """
    matches = re.findall(
        r'\[input\]="(.*?)"\s*\[output\]="(.*?)"',
        examples,
        re.DOTALL,
    )
    assert matches != []
    annotation_prompt_string = ""
    if is_generation:
        for input, output in matches:
            annotation_prompt_string += f"[input] = {input}\n"
            annotation_prompt_string += f"[output] = {output}\n"
        assert annotation_prompt_string != ""
        prompt = GENERATION_PROMPT_TEMPLATE.format(
            instruction=instruction,
            new_input=new_input,
            examples=annotation_prompt_string.strip(),
            few_shots_prompt=few_shots_prompt
        )
    else:
        annotation_prompt_string = ""
        for input, output in matches:
            annotation_prompt_string += f"USER: [input] = {input}\n"
            annotation_prompt_string += f"ASSISTANT: {output}\n"
        assert annotation_prompt_string != ""
        prompt = CLASSIFICATION_PROMPT_TEMPLATE.format(
            instruction=instruction,
            new_input=new_input,
            examples=annotation_prompt_string.strip(),
            few_shots_prompt=few_shots_prompt
        )
    return prompt
