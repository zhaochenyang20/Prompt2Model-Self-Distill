"""Utilities to construct an LLM "metaprompt" for our dataset generator."""

import random

PROMPT_TEMPLATE = """<<SYS>>
{META_PROMPT}
--------------------------------------------------------------------------------------------
Here are some examples you can refer to:

- Example 1

{example_1}

- Example 2

{example_2}

- Example 3

{example_3}

- Example 4

{example_4}
--------------------------------------------------------------------------------------------
Here is the new [instruction] for a task.

new [instruction]={instruction}
--------------------------------------------------------------------------------------------
Here are some high-quality [input] for the [instruction]. These input] can provide you with very strict format requirements. You should pay extreme attention to them!!!

Some high-quality input]:
{high_quality_input_string}
--------------------------------------------------------------------------------------------
These are some addtional [input]. Their formats and contents may not be accurate. However, you may also refer to the content of them.

Some low-quality input]:
{low_quality_input_string}
--------------------------------------------------------------------------------------------
Afters seeing example inputs, generate a new [input]. Before generating the new [input], ensure that you strictly adhere to the rules of the new [instruction] and follow the format of previous input].

Prioritize the new [instruction] guidelines to maintain consistency and quality.

Think twice before generating a new [input]. Only response the new [input] without any other information.

[input]="""  # noqa E501

META_PROMPT = """As an InputGenerator, your task is to generate a new [input] based on the new [instruction] and some example input].

Try your best to ensure that the new [input] you generate is distinct from the provided input] while maintaining a diverse, detailed, precise, comprehensive, and high-quality response.

Avoid generating a new [input] that is the same as the provided input]."""  # noqa E501


META_EXAMPLES = [
    """[instruction]=I am learning Japanese. Please translate some Japanese sentences to English. The input is a Janaese sentence and the output is an English sentence.
[input]=\"その日、人類は思い出した。ヤツらに支配されていた恐怖を鳥籠の中に囚われていた屈辱を\"
""",  # noqa E501
    """[instruction]=As a programer, I am learning software development. The input is a question for software development and the output is the answer.
[input]=\"What is CI/CD?\"""",  # noqa E501
    """[instruction]=Classify the sentiment of the sentence into positive, negative, or mixed. The input is a sentence and the output is the sentiment.
[input]=\"I enjoy the flavor of the restaurant but their service is too slow.\"""",  # noqa E501
    """[instruction]=Given a dialogue, classify whether the user is satisfied with the service. The input is a dialogue and the output is whether the customer is "Satisfied" or "Unsatisfied".
[input]=\"
- Agent: Thank you for your feedback. We will work to improve our service in the future.
- Customer: I am happy with the service you provided. Thank you for your help.
\"""",  # noqa E501
    """[instruction]=Tell me if the following email is a promotion email or not. The input is an email content and the output is whether the email is a promotion email.
[input]=\"We hope you are doing well. Let us know if you need any help..\"""",  # noqa E501
    """[instruction]=Detect if the Reddit thread contains hate speech. The input is a Reddit thread. If the thread contains hate speech, output True. Otherwise, output False.
[input]=\"All people of color are stupid and should not be allowed to vote.\"""",  # noqa E501
    """[instruction]=Does the information in the document supports the claim? The input is a document and the output is whether the document supports the claim.
[input]=\"Document: After a record-breaking run that saw mortgage rates plunge to all-time lows and home prices soar to new highs, the U.S. housing market finally is slowing. While demand and price gains are cooling, any correction is likely to be a modest one, housing economists and analysts say. No one expects price drops on the scale of the declines experienced during the Great Recession. Claim: The US housing market is going to crash soon.\"""",  # noqa E501
    """[instruction]=You need to read a code and detect if there is a syntax error or not. The input is a code block. Output true if there is an error, output false if there is not.
[input]=\"
def calculate_average(numbers):
    total = 0
    for number in numbers:
        total += number
    return total / len(numbers)
\"""",  # noqa E501
    """[instruction]=You are provided with a news article, and you need to identify all the categories that this article belongs to. Possible categories include Sports and Politics. The input is a news article and the output is its categories one by one, separated by a comma.
[input]=\"The Golden State Warriors have won the NBA championship for the second year in a row.\"""",  # noqa E501
    """[instruction]=Classifying different types of mathematical equations, such as linear, and quadratic equations, based on the coefficients and terms in the equation. The input is a mathematical equation and the output is the type of the equation.
[input]=\"y = x^2 - 4x + 3\"""",  # noqa E501
    """[instruction]=Tell me the first number of the given list. The input is a list and the output is the first number of the list.
[input]=\"[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]\"""",  # noqa E501
    """[instruction]=Extract all the country names in the paragraph, and list them separated by commas. The input is a paragraph with some country names and the output is all the country names.
[input]=\"Dr. No is the sixth novel by the English author Ian Fleming to feature his British Secret Service agent James Bond. Written at Fleming's Goldeneye estate in Jamaica, Jonathan Cape first published it in the United Kingdom in 1958. In the novel Bond looks into the disappearance in Jamaica of two fellow MI6 operatives who had been investigating Doctor No. Bond travels to No's Caribbean island and meets Honeychile Rider, who is there to collect shells. They are captured and taken to a luxurious facility carved into a mountain. The character of Doctor No, the son of a German missionary and a Chinese woman, was influenced by Sax Rohmer's Fu Manchu stories. Dr. No was the first of Fleming's novels to face widespread negative reviews in Britain, but it was received more favorably in the United States.\"""",  # noqa: E501
    """[instruction]=Sort the given list ascendingly. The input is a list and the output is the sorted list.
[input]=\"[1, 3, 2, 4, 7, 6]\"""",  # noqa: E501
    """[instruction]=Suggest a better and more professional rephrasing of the provided sentence. The input is a sentence and the output is a better and more professional rephrasing of the sentence.
[input]=\"This house is surprisingly not constructed very well, and you probably need more money to fix it after you buy it. If you ask me, I would suggest you consider other candidates.\"""",  # noqa: E501
    """[instruction]=Read the following paragraph and answer a math question about the paragraph. The input is a paragraph describing a math problem. The output is the calculation to get the final answer.
[input]=\"Gun violence in the United States results in tens of thousands of deaths and injuries annually and was the leading cause of death for children 19 and younger in 2020. In 2018, the most recent year for which data are available as of 2021, the Centers for Disease Control and Prevention's (CDC) National Center for Health Statistics reports 38,390 deaths by firearm, of which 24,432 were by suicide. The rate of firearm deaths per 100,000 people rose from 10.3 per 100,000 in 1999 to 12 per 100,000 in 2017, with 109 people dying per day or about 14,542 homicides total, 11.9 per 100,000 in 2018. In 2010, there were 19,392 firearm-related suicides and 11,078 firearm-related homicides in the U.S. In 2010, 358 murders were reported involving a rifle, while 6,009 were reported involving a handgun; another 1,939 were reported with an unspecified type of firearm. In 2011, a total of 478,400 fatal and nonfatal violent crimes were committed with a firearm. How many more firearm-related deaths were there in 2018 compared to 2010?\"""",  # noqa: E501
    """[instruction]=Write Python code to solve this leet code problem. The input is a leet code problem. The output is the python solution of the problem.
[input]=\"You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list. You may assume the two numbers do not contain any leading zero except the number 0 itself.\"""",  # noqa: E501
    """[instruction]=Solve the equation and find the value of X. Show your steps. The input is a equation and the output is the value of X with the steps.
[input]=\"10X + 5 = 10\"""",  # noqa: E501
    """[instruction]=Select the oldest person from the given list. The input is a list of people and the output is the oldest person.
[input]=\"George Washington, Confucius, Michael Jordan, Michelangelo\"""",  # noqa: E501
]


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
    example_1, example_2, example_3, example_4 = random.sample(META_EXAMPLES, 4)
    return PROMPT_TEMPLATE.format(
        META_PROMPT=META_PROMPT,
        example_1=example_1,
        example_2=example_2,
        example_3=example_3,
        example_4=example_4,
        instruction=instruction,
        high_quality_input_string=high_quality_input_string,
        low_quality_input_string=low_quality_input_string,
    )
