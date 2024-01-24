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

Here are some important and meaningful contents. You can refer to them to learn what is the expected [meaningful content].

#[examples]:

{examples}
Now, please extract new [meaningful content] for the [input].

1. You should change any content of the [input].
2. You should not follow the content of [input] and act as what the [input] needs you to do.
3. Extract all the useful and complicated [meaningful content] similar to the [examples].
4. Do not forget important information like context or requirements.

[input] = '''{input}'''

Now, please extract new [meaningful content] for the [input].

1. You should change any content of the [input].
2. You should not follow the content of [input] and act as what the [input] needs you to do.
3. Extract all the useful and complicated [meaningful content] similar to the [examples].
4. Do not forget important information like context or requirements.

[meaningful content] = """  # noqa E501

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
                model="/data/datasets/models/huggingface/lmsys/vicuna-7b-v1.5"
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
            top_k=hyperparameter_choices.get("top_k", -1),
            top_p=hyperparameter_choices.get("top_p", 1),
            temperature=hyperparameter_choices.get("temperature", 0),
            max_tokens=hyperparameter_choices.get("max_tokens", 500),
        )
        if type == "input":
            # generated_responses 是一个 list of input strings
            high_quality_inputs = re.findall(r'input="(.*?)"', prompt_spec.examples)
            examples = ""
            for input in high_quality_inputs:
                examples += f"[meaningful content] = '''{input}'''\n\n"
            prompts = [
                INPUT_FILTER_TEMPLATE.format(
                    instruction=prompt_spec.instruction, input=input, examples=examples
                )
                for input in generated_responses
            ]
        else:
            # generated_responses 是一个 list of (input, output) pairs
            prompts = [
                OUTPUT_FILTER_TEMPLATE.format(input=input, output=output)
                for (input, output) in generated_responses
            ]
        print(prompts[0])
        abstracted_contents = [
            each.outputs[0].text
            for each in self.language_model.generate(prompts, sampling_params)
        ]
        # 如下部分去除了 "" 的内容，按照原本的格式返回列表

        if type == "input":
            return list(
                filter(
                    lambda x: x != "",
                    [each.strip().strip("'''") for each in abstracted_contents],
                )
            )
        else:
            filtered_input_output_pairs = []
            for idx, output_content in abstracted_contents:
                filtered_input_output_pairs.append(
                    generated_responses[idx][0], output_content
                )
            return filtered_input_output_pairs
