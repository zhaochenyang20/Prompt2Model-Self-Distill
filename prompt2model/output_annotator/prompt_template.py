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

OUTPUT_ANNOTATOR_PROMPT_TEMPLATE = """
A chat between a curious user and an artificial intelligence assistant.
The assistant gives concise answers to the user's questions.
USER: The artificial intelligence assistant only needs to help annotate label. The task is: {instruction} 
ASSISTANT: Okay. 
{examples}
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
    prompt = OUTPUT_ANNOTATOR_PROMPT_TEMPLATE.format(
        instruction=instruction,
        new_input=new_input,
        examples=examples,
    ).strip()
    # print(prompt)
    return prompt
