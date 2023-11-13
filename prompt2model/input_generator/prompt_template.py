"""Utilities to construct an LLM "metaprompt" for our dataset generator."""

INPUT_PROMPT_TEMPLATE = """
As an InputGenerator, your task is to generate a new [input] based on the [instruction] and some example [input].

Try your best to ensure that the new [input] you generate is distinct from the provided [input] while maintaining a diverse, detailed, precise, comprehensive, and high-quality response.

Avoid generating a new [input] that is the same as the provided [input].
--------------------------------------------------------------------------------------------
[instruction]

{instruction}
--------------------------------------------------------------------------------------------
Here are some high-quality [input] for the [instruction]. These [input] can provide you with very strict format requirements. You should pay extreme attention to them!!!

Some high-quality [input]:

{high_quality_input_string}
--------------------------------------------------------------------------------------------
These are some addtional [input]. Their formats and contents may not be accurate. However, you may also refer to the content of them.

Some low-quality [input]:

{low_quality_input_string}
--------------------------------------------------------------------------------------------
Afters seeing example inputs, generate a new [input]. Before generating the new [input], ensure that you strictly adhere to the rules of the new [instruction] and follow the format of high-quality [input].

Prioritize the new [instruction] guidelines to maintain consistency and quality.

Think twice before generating a new [input]. Only response the new [input] without any other information.

[input]=
"""  # noqa E501


# INPUT_FILTER_TEMPLATE = """
# ### [EXAMPLE]

# "Question: What city did Super Bowl 50 take place in? Context: Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50."
# "Question: What river runs through Warsaw? Context: Warsaw (Polish: Warszawa [varˈʂava] ( listen); see also other names) is the capital and largest city of Poland. It stands on the Vistula River in east-central Poland, roughly 260 kilometres (160 mi) from the Baltic Sea and 300 kilometres (190 mi) from the Carpathian Mountains. Its population is estimated at 1.740 million residents within a greater metropolitan area of 2.666 million residents, which makes Warsaw the 9th most-populous capital city in the European Union. The city limits cover 516.9 square kilometres (199.6 sq mi), while the metropolitan area covers 6,100.43 square kilometres (2,355.39 sq mi)."
# "Question: The Ottoman empire controlled territory on three continents, Africa, Asia and which other? Context: The Ottoman Empire was an imperial state that lasted from 1299 to 1923. During the 16th and 17th centuries, in particular at the height of its power under the reign of Suleiman the Magnificent, the Ottoman Empire was a powerful multinational, multilingual empire controlling much of Southeast Europe, Western Asia, the Caucasus, North Africa, and the Horn of Africa. At the beginning of the 17th century the empire contained 32 provinces and numerous vassal states. Some of these were later absorbed into the empire, while others were granted various types of autonomy during the course of centuries."

# ### [NEW INPUT]

# {new_input}

# ### [YOUR TASK]

# Extract "Question" and "Context" from [NEW INPUT] and output as your response. 

# ### [YOUR RESPONSE]

# """  # noqa E501

INPUT_FILTER_TEMPLATE = """
### [EXAMPLE]

{few_shot_examples}

### [NEW INPUT]

{new_input}

### [YOUR TASK]

Extract "Question" and "Context" from [NEW INPUT] and output as your response. 

### [YOUR RESPONSE]

"""  # noqa E501


def construct_verify_prompt(
    examples: str = None,
    new_input: str = None,
):
    return INPUT_FILTER_TEMPLATE.format(
        few_shot_examples=examples,
        new_input=new_input,
    )


def construct_meta_prompt(
    instruction: str = None,
    low_quality_input_string: str = None,
    high_quality_input_string: str = None,
) -> str:
    """Constructs a prompt template for the dataset generator.

    Args:
        instruction: The natural language instruction for the prompt.
        low_quality_input_string: A string representing the low quality examples.
        high_quality_input_string: A string representing the high quality examples.

    Returns:
        str: A prompt template, where the `instruction` and `examples` fields
            are filled in.
    """
    return INPUT_PROMPT_TEMPLATE.format(
        instruction=instruction,
        high_quality_input_string=high_quality_input_string,
        low_quality_input_string=low_quality_input_string,
    )
