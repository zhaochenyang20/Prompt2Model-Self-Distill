"""Utilities to construct an LLM "metaprompt" for our dataset generator."""

import re
import random

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
USER: [input] {notion} {new_input}
ASSISTANT:
"""  # noqa E501

def reverse_func(labels):
    number_labels = len(labels)
    new_indexes = list(range(number_labels))
    new_index = random.choice(new_indexes)
    return labels[new_index]

def construct_meta_prompt(
    instruction: str = None,
    examples: str = None,
    new_input: str = None,
    is_generation: bool = True,
    few_shots_prompt: str = None,
    notion: str = '=',
    suffix: str = '\n',
    reverse: bool = False,
    labels: list = []
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

    # reverse labels in matches (few-shots examples)
    fixed_few_shots_examples = []
    if reverse:
        for match in matches:
            random_label = reverse_func(labels=labels)
            example = (match[0], random_label)
            fixed_few_shots_examples.append(example)
    else:
        fixed_few_shots_examples = matches


    annotation_prompt_string = ""
    # is_generation is always false
    # if is_generation:
    #     for input, output in matches:
    #         annotation_prompt_string += f"[input] {notion} {input}{suffix}"
    #         annotation_prompt_string += f"[output] {notion} {output}{suffix}"
    #     assert annotation_prompt_string != ""
    #     prompt = GENERATION_PROMPT_TEMPLATE.format(
    #         instruction=instruction,
    #         new_input=new_input,
    #         examples=annotation_prompt_string.strip(),
    #         few_shots_prompt=few_shots_prompt
    #     )
    # else:
    annotation_prompt_string = ""
    for input, output in fixed_few_shots_examples:
        annotation_prompt_string += f"USER: [input] {notion} {input}{suffix}"
        annotation_prompt_string += f"ASSISTANT: {output}{suffix}"
    assert annotation_prompt_string != ""
    prompt = CLASSIFICATION_PROMPT_TEMPLATE.format(
        instruction=instruction,
        new_input=new_input,
        examples=annotation_prompt_string.strip(),
        few_shots_prompt=few_shots_prompt,
        notion=notion
    )
    return prompt
