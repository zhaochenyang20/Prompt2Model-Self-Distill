"""Test Input Generator."""

from pathlib import Path

import datasets

from prompt2model.input_generator import VLLMPromptBasedInputGenerator
from prompt2model.prompt_parser import MockPromptSpec, TaskType

# 测试

inputs_dir = Path("/home/cyzhao/generated_datasets")
inputs_dir.mkdir(parents=True, exist_ok=True)

generated_inputs = []

prompt_spec = MockPromptSpec(
    task_type=TaskType.TEXT_GENERATION,
    instruction="Given an abstract, generate a keyword (a noun phrase) that best describes the focus or contribution of the paper. Such keywords can be directly from the given abstract or outside it.",  # # noqa E501
    examples="""
[input]=Abstract: Some patients converted from ventricular fibrillation to organized rhythms by defibrillation - trained ambulance technicians(EMT - Ds) will refibrillate before hospital arrival.The authors analyzed 271 cases of ventricular fibrillation managed by EMT - Ds working without paramedic back - up.Of 111 patients initially converted to organized rhythms, 19(17 % ) refibrillated, 11(58 % ) of whom were reconverted to perfusing rhythms, including nine of 11(82 % ) who had spontaneous pulses prior to refibrillation.Among patients initially converted to organized rhythms, hospital admission rates were lower for patients who refibrillated than for patients who did not(53 % versus 76 % , P = NS), although discharge rates were virtually identical(37 % and 35 % , respectively).Scene - to - hospital transport times were not predictively associated with either the frequency of refibrillation or patient outcome.Defibrillation - trained EMTs can effectively manage refibrillation with additional shocks and are not at a significant disadvantage when paramedic back - up is not available.
[output]=Ventricular Fibrillation

[input]=Abstract: Our results suggest that ethylene oxide retention after sterilization is increased in cuprammonium cellulose plate dialyzers containing potting compound. In contrast, cuprammonium cellulose plate dialyzers without potting compound were characterized by a rapid disappearance of retained ethylene oxide after sterilization. Whether these findings explain the low incidence of SARD with cuprammonium cellulose plate dialyzers that do not contain potting material is a matter for continued study and experimentation.
[output]=Sterilization


""",  # noqa E501
)

input_generator = VLLMPromptBasedInputGenerator(gpu_memory_utilization=0.9)

# prompt = input_generator.construct_generation_prompt(
#     instruction=prompt_spec.instruction,
#     few_shot_example_string=prompt_spec.examples,
#     generated_inputs=["How are you?", "I am fine."],
#     context_cutoff=3500,
# )

# filter_prompt = input_generator.construct_filter_prompt(
#     few_shot_example_string=prompt_spec.examples, new_input="how are you?"
# )
# generation_epochs,generation_batch_size,generation_top_k,generation_temperature,min_frequency,min_input_length,training_epochs
# 10                15                      50              0.3                     0.5         125                5
inputs = input_generator.batch_generation_inputs(
    prompt_spec,
    5,
    5,
    dict(
        top_k=20,
        temperature=0.3,
        min_input_length=150,
    ),
    expected_content='"Abstract"',
    optional_list=['[input]', '[NEW INPUT]','[ABSTRACT]', 'Abstract']
)
