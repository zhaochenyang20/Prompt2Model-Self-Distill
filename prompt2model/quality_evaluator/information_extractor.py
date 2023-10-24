"""Extract useful content out of model generated response."""

import re
from typing import Any

from vllm import LLM, SamplingParams

from prompt2model.prompt_parser import PromptSpec
from prompt2model.quality_evaluator.base import QualityEvaluator

INPUT_FILTER_TEMPLATE = """<<SYS>>
As an InformationExtractor, you should abstract the most important and meaningful content from [input] to work as a qualified input under the [instruction] and use [info] [/info] to wrap the useful content.
You don’t need to follow the instruction and act as what the instruction needs you to do. Just use [info] [/info] to wrap the useful content.
<</SYS>>

Here are 2 examples for your reference:

- Example_1:

[Instruction]={{You are provided with a news article, and you need to identify all the categories that this article belongs to. Possible categories include Sports and Politics. The input is a news article and the output is its categories one by one, separated by a comma."}}

[Input]={{Sure, here's an example input and output for the task you described:
Input:
Title: "LeBron James Signs With Los Angeles Lakers in Blockbuster Deal"
Text: "LeBron James, one of the biggest stars in the NBA, has signed with the Los Angeles Lakers in a move that is expected to shake up the league. James, who has played for the Cleveland Cavaliers and the Miami Heat, has agreed to a four-year contract worth $154 million. The deal is the largest in NBA history and cements James' status as the face of the Lakers franchise. James, who is 35 years old and in his 16th season, is coming off a year in which he led the Cavaliers to the NBA Finals. The Lakers, who have not made the playoffs since 2013, are hoping that James will lead them back to prominence."
Output: Sports, Basketball, NBA, Los Angeles Lakers, LeBron James.
In this example, the article belongs to the categories of Sports, Basketball, and NBA, as it is about a professional sports team and player. The article also mentions the Los Angeles Lakers and LeBron James, who are both prominent figures in the NBA, making the additional categories of "Los Angeles Lakers" and "LeBron James" applicable."}}

[abstracted]={{[info]LeBron James, one of the biggest stars in the NBA, has signed with the Los Angeles Lakers in a move that is expected to shake up the league. James, who has played for the Cleveland Cavaliers and the Miami Heat, has agreed to a four-year contract worth $154 million. The deal is the largest in NBA history and cements James' status as the face of the Lakers franchise. James, who is 35 years old and in his 16th season, is coming off a year in which he led the Cavaliers to the NBA Finals. The Lakers, who have not made the playoffs since 2013, are hoping that James will lead them back to prominence.[/info]}}

- Example_2:

[Instruction]={{As a programer, I am learning software development. The input is a question for software development and the output is the answer.}}

[input]={{Great! Here's an input and output for the task:
Input: "How do I write a scalable software system?"
Can you please provide an answer for this question?}}

[abstracted]={{[info]What is CI/CD?[/info]}}

After seeing these two examples, please abstract the most important and meaningful content from [input] to work as a qualified input under the [instruction] and use [info] [/info] to wrap the useful content.

[instruction] = {{{instruction}}}

[input] = {{{input}}}

[abstracted] ="""  # noqa E501

# TODO: 不合适的内容就不要了，需要加入更多 examples

OUTPUT_FILTER_TEMPLATE = """<<SYS>>
You should abstract the most important and meaningful content from [output] to work as a qualified output under the [instruction] and [input], then use [info] [/info] to wrap the useful content. Please use only one [info][/info] pair in your answer.
<</SYS>>

Here are two examples for your reference and then you will finish one task:

Example_1:
Input to the model:
[Instruction] = {{Your task is to generate an answer to a question about software development.}}
[input] = {{What is the CI/CD?}}
[output] = {{Great! The answer to the question "What is CI/CD?" is:
CI/CD stands for Continuous Integration/Continuous Deployment, which are two related practices used in software development to automate the build, test, and deployment process.}}
Output of the model:
[abstracted] = [info]Continuous Integration/Continuous Deployment.[/info]

Example_2:
Input to the model:
[instruction] = {{Classifying different types of mathematical equations, such as linear, and quadratic equations, based on the coefficients and terms in the equation. The input is a mathematical equation and the output is the type of the equation.}}
[input] = {{y = x^2 - 4x + 3}}
[output] = {{Sure! A quadratic equation is a polynomial equation of degree 2, meaning the highest power of the variable (in this case, x) is 2. The equation can be written in the standard form:
ax^2 + bx + c = 0
where a, b, and c are constants, and x is the variable.
In the case of your equation, we have:
y = x^2 - 4x + 3
So, the type of equation is quadratic.}}
Output of the model:
[abstracted]=[info]Quadratic equation.[/info]

Here it is your turn to finish a case:
Input to the model:
[instruction]={{{instruction}}}
[input]={{{input}}}
[output] = {{{output}}}
Output of the model:
[abstracted]="""


class VLLMInformationExtractor(QualityEvaluator):
    """Abstract useful information from the generated content of model."""

    def __init__(self, pretrained_model_name: str = "lmsys/vicuna-7b-v1.5") -> None:
        """Initializes InformationExtractor with a pre-trained model.

        Args:
            pretrained_model_name: The name of a pre-trained
                middle-sized autoregressive model.
        """
        if pretrained_model_name == "lmsys/vicuna-7b-v1.5":
            self.language_model = LLM(
                model="/data/ckpts/huggingface/models/models--lmsys--vicuna-7b-v1.5/snapshots/de56c35b1763eaae20f4d60efd64af0a9091ebe5"
            )
        else:
            self.language_model = LLM(model=pretrained_model_name)

    def response_filter(
        self,
        prompt_spec: PromptSpec,
        generated_responses: list[Any],
        type: str = None,
        hyperparameter_choices=dict[str, Any],
    ) -> str:
        """
        Construct a prompt to filter response given by model to abstract the useful information by
        adding tags [info] and [/info] to wrap the content.

        Args:
            response: the response generated by model.

        Returns:
            str: The abstracted useful content of generated response or empty string if the model find no useful information.
        """
        sampling_params = SamplingParams(
            top_k=hyperparameter_choices.get("top_k", 3),
            top_p=hyperparameter_choices.get("top_p", 0.2),
            temperature=hyperparameter_choices.get("temperature", 0),
            max_tokens=hyperparameter_choices.get("max_tokens", 500),
        )
        if type == "input":
            # generated_responses 是一个 list of input strings
            prompts = [
                INPUT_FILTER_TEMPLATE.format(
                    instruction=prompt_spec.instruction, input=input
                )
                for input in generated_responses
            ]
        else:
            # generated_responses 是一个 list of (input, output) pairs
            prompts = [
                OUTPUT_FILTER_TEMPLATE.format(input=input, output=output)
                for (input, output) in generated_responses
            ]
        abstracted_contents = self.language_model.generate(prompts, sampling_params)

        # 如下部分去除了 "" 的内容，按照原本的格式返回列表

        def match_content(content):
            pattern = r"\[info\](.*?)\[/info\]"
            matches = re.findall(pattern, content)
            if not matches:
                return ""
            return matches[0]

        if type == "input":
            return list(
                filter(
                    lambda x: x != "",
                    [
                        match_content(input_content)
                        for input_content in abstracted_contents
                    ],
                )
            )
        else:
            filtered_input_output_pairs = []
            for idx, output_content in abstracted_contents:
                matched_output_content = match_content(output_content)
                if matched_output_content != "":
                    filtered_input_output_pairs.append(
                        generated_responses[idx][0], matched_output_content
                    )
            return filtered_input_output_pairs
