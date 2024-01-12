PROMPT_TEMPLATE = """A chat between a curious user and an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers to the user's questions.

USER:

{task_instruction}

ASSISTANT:

Okay.

{examples}

USER:

{new_input}

ASSISTANT:

"""

# all the prompt we have:

# input generator and input corrector: 
# for generation tasks: INPUT_GENERATOR_PROMPT_TEMPLATE, INPUT_FILTER_TEMPLATE
# for classification tasks: CONDITIOANL_INPUT_GENERATOR_PROMPT_TEMPLATE, CONDITIONAL_INPUT_FILTER_TEMPLATE

# output anatator: 
# OUTPUT_ANNOTATOR_PROMPT_TEMPLATE

# inference (main.py): during test and evaluate
# PROMPT_TEMPLATE: providing input with prompts

# trainer:
# PROMPT_TEMPLATE: providing training dataset