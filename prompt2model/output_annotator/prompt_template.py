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

PROMPT_TEMPLATE_199 = """
A chat between a curious user and an artificial intelligence assistant.
The assistant gives concise answers to the user's questions.
USER: The artificial intelligence assistant only needs to help annotate label. The task is: {instruction} 
ASSISTANT: Okay. 
USER: [input] = "Sentence 1: Next to the MGM Grand you will find M and M World. Sentence 2: The candy has many fans who love its attractions."
ASSISTANT: no
USER: [input] = "Sentence 1: I've forgotten his name now, confessed Tuppence. Sentence 2: Tuppence remembered his name later."
ASSISTANT: no
USER: [input] = "Sentence 1: One of the first organizational realignments taking place is in the Office of the Taxpayer Advocate. Sentence 2: The office of the taxpayer advocate is having an organizational realignment."
ASSISTANT: yes
USER: [input] = "Sentence 1: yeah I tell you what though if you go price some of those tennis shoes i can see why now you know they're getting up in the hundred dollar range. Sentence 2: The tennis shoes have only one price."
ASSISTANT: yes

USER: [input] = {new_input}
ASSISTANT: 
"""  # noqa E501

PROMPT_TEMPLATE = """
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
    prompt = PROMPT_TEMPLATE.format(
        instruction=instruction,
        new_input=new_input,
        examples=examples,
    ).strip()
    print(prompt)
    return prompt
