"""Test Input Generator."""

from prompt2model.dataset_generator.prompt_based import Example
from prompt2model.input_generator import PromptBasedInputGenerator
from prompt2model.prompt_parser import MockPromptSpec, TaskType

generated_examples = [
    Example(
        input_col="Question: What is the capital city of France?", output_col="Paris"
    ),
    Example(
        input_col="Question: When was the American Declaration of Independence signed?",
        output_col="July 4, 1776",
    ),
    Example(
        input_col="Question: What is the tallest mountain in the world?",
        output_col="Mount Everest",
    ),
]

prompt_spec = MockPromptSpec(
    task_type=TaskType.TEXT_GENERATION,
    instruction="Your task is to generate an answer to a natural question. In this task, the input is a question string. and the output is the corresponding answer string. The question can range from Math, Cultural, Social, Geometry, Biology, History, Sports, Technology, Science, and so on.",  # # noqa E501
    examples="""input="Question: What is the capital city of China?", output="Shanghai"\n\ninput="Question: When was People's Republic of China founded?", output="October 1, 1949"\n\n""",  # noqa E501
)

input_generator = PromptBasedInputGenerator(
    pretrained_model_name="meta-llama/Llama-2-7b-chat-hf"
)

inputs = input_generator.generate_inputs(
    generated_examples=generated_examples,
    prompt_spec=prompt_spec,
    hyperparameter_choices={"num_return_sequences": 1},
)
