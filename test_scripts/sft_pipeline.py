import gc
import os
from functools import partial
from pathlib import Path

import datasets
import torch
import wandb

os.environ["WANDB_PROJECT"] = "test"
import wandb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainingArguments,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer


class WandbLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            # Filter out the desired metrics if needed
            wandb.log(logs)


from prompt2model.output_annotator import construct_meta_prompt
from prompt2model.prompt_parser import MockPromptSpec, TaskType

model_path = Path(
    "/data/ckpts/huggingface/models/models--lmsys--vicuna-7b-v1.5/snapshots/de56c35b1763eaae20f4d60efd64af0a9091ebe5"
)
ckpt_path = Path("/home/cyzhao/ckpt")
generated_dataset_path = Path("/home/cyzhao/generated_datasets")

prompt_spec = MockPromptSpec(
    task_type=TaskType.TEXT_GENERATION,
    instruction="Your task is to generate an answer to a natural question. In this task, the input is a string that consists of both a question and a context passage. The context is a descriptive passage related to the question and contains the answer. And the question can range from Math, Cultural, Social, Geometry, Biology, History, Sports, Technology, Science, and so on.",  # # noqa E501
    examples="""
[input]="Question: What city did Super Bowl 50 take place in? Context: Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50."
[output]="Santa Clara"

[input]="Question: What river runs through Warsaw? Context: Warsaw (Polish: Warszawa [varˈʂava] ( listen); see also other names) is the capital and largest city of Poland. It stands on the Vistula River in east-central Poland, roughly 260 kilometres (160 mi) from the Baltic Sea and 300 kilometres (190 mi) from the Carpathian Mountains. Its population is estimated at 1.740 million residents within a greater metropolitan area of 2.666 million residents, which makes Warsaw the 9th most-populous capital city in the European Union. The city limits cover 516.9 square kilometres (199.6 sq mi), while the metropolitan area covers 6,100.43 square kilometres (2,355.39 sq mi)."
[output]="Vistula River"

[input]="Question: The Ottoman empire controlled territory on three continents, Africa, Asia and which other? Context: The Ottoman Empire was an imperial state that lasted from 1299 to 1923. During the 16th and 17th centuries, in particular at the height of its power under the reign of Suleiman the Magnificent, the Ottoman Empire was a powerful multinational, multilingual empire controlling much of Southeast Europe, Western Asia, the Caucasus, North Africa, and the Horn of Africa. At the beginning of the 17th century the empire contained 32 provinces and numerous vassal states. Some of these were later absorbed into the empire, while others were granted various types of autonomy during the course of centuries."
[output]="Europe"
"""
    # noqa E501
)

construct_prompt = partial(
    construct_meta_prompt,
    instruction=prompt_spec.instruction,
    examples=prompt_spec.examples,
)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True,
    padding_side="left",
    trust_remote_code=True,
)


def filter_func(example):
    return example["output_col"] is not None and example["input_col"] is not None


def map_func(example):
    example["model_input"] = construct_prompt(new_input=example["input_col"])
    example["model_output"] = example["output_col"]
    example["text"] = (
        example["model_input"] + example["model_output"] + tokenizer.eos_token
    )
    return example


for each in os.listdir(generated_dataset_path):
    name = each[8:]
    dataset = datasets.load_from_disk(generated_dataset_path / each).filter(filter_func)
    mapped_dataset = dataset.map(map_func, load_from_cache_file=False)
    print(mapped_dataset[1]["text"])
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=True,
    )
    response_template_with_context = "\n### Your Output:\n\n"
    response_template_ids = tokenizer.encode(
        response_template_with_context, add_special_tokens=False
    )[2:]

    data_collator = DataCollatorForCompletionOnlyLM(
        response_template_ids, tokenizer=tokenizer
    )
    training_args = TrainingArguments(
        report_to="wandb",
        output_dir="/home/cyzhao/cache",
        do_eval=False,
        save_strategy="no",
        num_train_epochs=1,
    )
    wandb.init(project="your_project_name", name="your_run_name")
    wandb.config.update(training_args.to_dict())
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=mapped_dataset,
        dataset_text_field="text",
        data_collator=data_collator,
        max_seq_length=1500,
        callbacks=[WandbLoggingCallback()],  # Add your custom callback here
    )
    trainer.train()
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    model.save_pretrained(ckpt_path / name)
    tokenizer.save_pretrained(ckpt_path / name)
