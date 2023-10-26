"""Utilities to construct an LLM "metaprompt" for our dataset generator."""

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


"""[input]="Question: What is the capital city of France? Context: France, officially the French Republic, is a country primarily located in Western Europe, consisting of metropolitan France and several overseas regions and territories. The capital city of France is Paris, which is also its largest city. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral."
[output]="Paris"

[input]="Question: What is the capital city of Australia? Context: Australia, officially known as the Commonwealth of Australia, is a sovereign country comprising the mainland of the Australian continent, the island of Tasmania, and numerous smaller islands. It is the largest country in Oceania and the world's sixth-largest country by total area. Canberra is the capital city of Australia and is located in the Australian Capital Territory (ACT). The city was chosen as the capital in 1908 as a compromise between Sydney and Melbourne, the two largest cities in Australia."
[output]="Canberra"

[input]="Question: What is the largest desert in the world? Context: Deserts are dry, barren landscapes that receive very little rainfall. The largest desert in the world is the Sahara Desert, which covers an area of approximately 9.2 million square kilometers. It is located in northern Africa and spans across several countries, including Algeria, Chad, Egypt, Libya, Mali, Mauritania, Morocco, Niger, Sudan, and Tunisia. The Sahara Desert is known for its vast stretches of sand dunes and extreme temperatures."
[output]="Sahara Desert"

[input]="Question: What is the largest ocean in the world? Context: Oceans are large bodies of saltwater that cover more than 70% of the Earth's surface. The largest ocean in the world is the Pacific Ocean, which stretches across approximately 63 million square miles. It is located between the Western Hemisphere and the Eastern Hemisphere and is bordered by Asia and Australia to the west, and the Americas to the east."

[output]="""

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
