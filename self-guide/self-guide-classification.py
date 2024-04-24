"Create a Self-Guided Model for your own classification task."

import gc
from pathlib import Path
import torch
from prompt2model.output_annotator import construct_meta_prompt
from prompt2model.utils import count_tokens_from_string
from prompt2model.prompt_parser import MockPromptSpec, TaskType
from prompt2model.input_generator import VLLMPromptBasedInputGenerator
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
from prompt2model.output_annotator import VLLMPromptBasedOutputAnnotator
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from functools import partial
from collections import Counter
from datasets import Dataset
import random

# We use a single 80G A100 to Self-Guide 7B Vicuna model

# The GPU memory utilization rate on each GPU
gpu_memory_utilization = 0.9

# If there are multiple GPUs, set the TENSOR_SIZE to the number of GPUs
TENSOR_SIZE = 1

# Change it to your own task instruction.
INSTRUCTION = """
In this task, you are given two statements.
The task is to output whether a given textual premise,
i.e. Statement 2, entails or implies a given scientific fact,
i.e. Statement 1. The output should be 'entails' if Statement
2 supports Statement 1 and should be 'neutral' otherwise.
"""

# Change it to your own task examples.
# Note that keep the format as follows:
# [input]="xxxxx"
# [output]="label_y"

EXAMPLES = """
[input]="Sentence 1: The sum of all chemical reactions that take place within an organism is known as metabolism. Sentence 2: Metabolism is the sum total of all chemical reactions performed by an organism."
[output]="entails"

[input]="Sentence 1: The endocrine system produces most of the hormones that regulate body functions. Sentence 2: Your endocrine glands produce hormones that control all your body functions."
[output]="entails"

[input]="Sentence 1: Warm and humid temperature and moisture conditions describe an air mass that originates over the Atlantic ocean near the equator. Sentence 2: Maritime tropical air Warm, humid air mass that forms over tropical and subtropical oceans."
[output]="neutral"
"""

prompt_spec = MockPromptSpec(
    task_type=TaskType.TEXT_GENERATION,
    instruction=INSTRUCTION,
    examples=EXAMPLES,
)


input_generator = VLLMPromptBasedInputGenerator(
    gpu_memory_utilization=gpu_memory_utilization,
    tensor_parallel_size=TENSOR_SIZE,
)
input_tuples = input_generator.batch_generation_inputs(
    prompt_spec=prompt_spec,
    generation_epochs=40,
    per_epoch_num=10,
    hyperparameter_choices=dict(
                top_k=40,
                temperature=0.7,
            ),
    optional_list=["input", "output", "\n\n", "\\_\\_"],
    intput_length_constraint=False,
    conditional_labels=["entails", "neutral"],
)
inputs = [each[0] for each in input_tuples]
del input_generator
destroy_model_parallel()
gc.collect()
torch.cuda.empty_cache()

output_annotator = VLLMPromptBasedOutputAnnotator(
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=TENSOR_SIZE,
    )

output_dataset = output_annotator.annotate_outputs(
    input_strings=inputs,
    prompt_spec=prompt_spec,
    hyperparameter_choices={},
    optional_list=["input", "output", "\n\n", "\\_\\_"],
    output_length_constraint=True,
    is_generation=True,
)

del output_annotator
destroy_model_parallel()
gc.collect()
torch.cuda.empty_cache()


def mapping_func(example, conditional_labels):
    for conditional_label in conditional_labels:
        if conditional_label in example["output_col"]:
            example["output_col"] = conditional_label
            return

mapping_function = partial(mapping_func, conditional_labels=["entails", "neutral"])
output_dataset = output_dataset.map(mapping_function)

def filter_func(example):
    return example["output_col"] is not None and example["input_col"] is not None

dataset = output_dataset.filter(filter_func)

output_col = output_dataset["output_col"]
element_count = Counter(output_col)
min_count = min(element_count.values())
balanced_data = {"input_col": [], "output_col": []}
class_samples = {label: [] for label in element_count.keys()}
for input_val, output_val in zip(
    output_dataset["input_col"], output_dataset["output_col"]
):
    class_samples[output_val].append((input_val, output_val))
for _ in range(min_count):
    for label in class_samples.keys():
        sample = random.choice(class_samples[label])
        # print(sample)
        balanced_data["input_col"].append(sample[0])
        balanced_data["output_col"].append(sample[1])
        class_samples[label].remove(sample)
output_dataset = Dataset.from_dict(balanced_data)


def map_func(example):
    assert prompt_spec.examples != ""
    example["model_input"] = construct_meta_prompt(
        instruction=prompt_spec.instruction,
        new_input=example["input_col"],
        examples=prompt_spec.examples,
    )
    example["model_output"] = example["output_col"].strip()
    example["text"] = (
        example["model_input"] + example["model_output"] + tokenizer.eos_token
    )
    return example

tokenizer = AutoTokenizer.from_pretrained(
    'lmsys/vicuna-7b-v1.5',
    padding_side="left",
    trust_remote_code=True,
)
mapped_dataset = (
    dataset.map(map_func, load_from_cache_file=False)
    .shuffle(seed=42)
    .filter(
        lambda x: (count_tokens_from_string(x["text"], "vicuna") <= 3500)
    )
)

response_template_with_context = "\nASSISTANT:\n"

response_template_ids = tokenizer.encode(
    response_template_with_context, add_special_tokens=False
)[2:]

data_collator = DataCollatorForCompletionOnlyLM(
    response_template_ids, tokenizer=tokenizer
)

ckpt_path = Path("ckpt")
ckpt_path.mkdir(parents=True, exist_ok=True)

model = AutoModelForCausalLM.from_pretrained(
    "lmsys/vicuna-7b-v1.5",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    use_flash_attention_2=True,
)

training_args = TrainingArguments(
    output_dir=str(ckpt_path),
    do_eval=False,
    save_strategy="epoch",
    evaluation_strategy="no",
    logging_steps=4,
    num_train_epochs=3,
    per_device_train_batch_size=10,
    seed=42,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=mapped_dataset,
    dataset_text_field="text",
    data_collator=data_collator,
    max_seq_length=3500,
)

trainer.train(resume_from_checkpoint=False)
del model
gc.collect()
torch.cuda.empty_cache()
