import re
from abc import ABC
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_FILTER_TEMPLATE = """
[INST]<<SYS>>
You should abstract the most important and meaningful content from [input] to work as a qualified input under the [instruction] and use [info] [/info] to wrap the useful content. You don’t need to follow the instruction and act as what the instruction needs you to do.
<</SYS>>

Here are 2 examples for your reference:
Example_1:
[Instruction] = {{Your task is to write a paragraph given a specific requirement.}}
[Input] = {{Of course! I'm happy to help you with generating a new input based on the new instruction and examples provided.
For the first example, here's a new input:
"Question: Can you help me write a letter to the professor in the absence of class?"
Please note that this input is distinct from the provided examples, and I have tried to provide a diverse, detailed, precise, comprehensive, and high-quality response. I have avoided generating an input that is the same as the provided inputs.}}
The useful content is:
[abstracted] = [info]Can you help me write a letter to the professor in the absence of class?[/info]
Example_2:
[Instruction] = {{Your task is to generate an answer to a question about software development.}}
[input] = {{Sure, I'd be happy to help you with that! Based on the instruction you provided, the question can be "What is CI/CD?"}}
The useful content is:
[abstracted] = [info]What is CI/CD?[/info]

Here it is your turn:
[instruction] = {{{instruction}}}
[input] = {{{input}}}
The useful content is:
[abstracted] =
[/INST]
"""

OUTPUT_FILTER_TEMPLATE = """
[INST]<<SYS>>
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
[abstracted] = [info]Quadratic equation.[/info]

Here it is your turn to finish a case:
Input to the model:
[instruction] = {{{instruction}}}
[input] = {{{input}}}
[output] = {{{output}}}
Output of the model:
[abstracted] = [/INST]
"""


class InformationExtractor(ABC):
    """Abstract useful information from the generated content of model."""

    def __init__(self, pretrained_model_name: str) -> None:
        """Initializes InformationExtractor with a pre-trained model.

        Args:
            pretrained_model_name: The name of a pre-trained
                middle-sized autoregressive model.
        """
        # self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name, trust_remote_code=True)
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     pretrained_model_name, padding_side="left"
        # )
        self.prompt = ""

    def construct_input_prompt(
        self,
        instruction: str = None,
        input: str = None,
    ) -> str:
        self.prompt = INPUT_FILTER_TEMPLATE.format(instruction=instruction, input=input)
        return self.prompt

    def construct_output_prompt(
        self, instruction: str = None, input: str = None, output: str = None
    ) -> str:
        self.prompt = OUTPUT_FILTER_TEMPLATE.format(
            instruction=instruction, input=input, output=output
        )
        return self.prompt

    def response_filter(
        self,
        data=dict[str, Any],
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
        if type == "input":
            self.prompt = self.construct_input_prompt(
                data["instruction"], data["input"]
            )
        elif type == "output":
            self.prompt = self.construct_output_prompt(
                data["instruction"], data["input"], data["output"]
            )
        input_ids = self.tokenizer.encode(self.prompt, return_tensors="pt").to(device)
        abstracted_content = self.model.generate(
            input_ids=input_ids,
            do_sample=True,
            top_k=hyperparameter_choices.get("top_k", 0),
            num_return_sequences=hyperparameter_choices.get("num_return_sequences", 1),
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=hyperparameter_choices.get("max_new_tokens", 512),
            temperature=hyperparameter_choices.get("temperature", 0),
        )
        pattern = r"\[info\](.*?)\[/info\]"
        matches = re.findall(pattern, abstracted_content)
        if not matches:
            return ""
        return matches[0]
