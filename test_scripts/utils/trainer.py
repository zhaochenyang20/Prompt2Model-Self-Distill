import gc
import re
import os
from functools import partial
from IPython import embed

import datasets
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

from prompt2model.output_annotator import construct_meta_prompt
from prompt2model.utils import count_tokens_from_string
from prompt2model.utils.prompt import PROMPT_TEMPLATE


def finetune_vicuna(
    prompt_spec,
    log_and_data_path,
    ckpt_path,
    pretrain_model_path,
    training_epochs,
    resume_from_checkpoint,
    run_name,
    task_name,
    max_seq_length=2000,
    per_device_train_batch_size=6,
):

    def filter_func(example):
        return example["output_col"] is not None and example["input_col"] is not None

    dataset = datasets.load_from_disk(log_and_data_path / "dataset").filter(filter_func)

    def map_func(example):
        matches = re.findall(
            r'\[input\]="(.*?)"\s*\[output\]="(.*?)"',
            prompt_spec.examples,
            re.DOTALL,
        )
        assert matches != []
        annotation_prompt_string = ""
        for input, output in matches:
            annotation_prompt_string += f"USER:\n\n{input}\n\n"
            annotation_prompt_string += f"ASSISTANT:\n\n{output}\n\n"
        assert annotation_prompt_string != ""
        example["model_input"] = PROMPT_TEMPLATE.format(
            task_instruction=prompt_spec.instruction,
            new_input=example["input_col"],
            examples=annotation_prompt_string.strip()
        )
        #! 此处绝对不可以 strip，否则 token 定位失效
        example["model_output"] = example["output_col"].strip()
        example["text"] = (
            example["model_input"] + example["model_output"] + tokenizer.eos_token
        )
        #! 此处其实也不用 strip，因为 eos 必定不是 \n
        return example

    tokenizer = AutoTokenizer.from_pretrained(
        pretrain_model_path,
        local_files_only=True,
        padding_side="left",
        trust_remote_code=True,
    )
    mapped_dataset = (
        dataset.map(map_func, load_from_cache_file=False)
        .shuffle(seed=42)
        .filter(
            lambda x: (count_tokens_from_string(x["text"], "vicuna") <= max_seq_length)
        )
    )
    response_template_with_context = "ASSISTANT:\n\n"
    response_template_ids = tokenizer.encode(
        response_template_with_context, add_special_tokens=False
    )[2:]
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template_ids, tokenizer=tokenizer
    )
    ckpt_path.mkdir(parents=True, exist_ok=True)
    # embed()
    model = AutoModelForCausalLM.from_pretrained(
        pretrain_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    wandb.init(project=task_name, name=run_name)
    wandb.watch(model)
    training_args = TrainingArguments(
        report_to="wandb",
        run_name=run_name,
        output_dir=str(ckpt_path),
        do_eval=False,
        save_strategy="epoch",
        evaluation_strategy="no",
        logging_steps=4,
        num_train_epochs=training_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        seed=42,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=mapped_dataset,
        dataset_text_field="text",
        data_collator=data_collator,
        max_seq_length=max_seq_length,
    )
    wandb.config.update(training_args.to_dict(), allow_val_change=True)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    for dirpath, _, filenames in os.walk(str(ckpt_path), topdown=True):
        # Delete optimizer
        for filename in filenames:
            if filename == "optimizer.pt":
                file_path = os.path.join(dirpath, filename)
                print(f"Deleting {file_path}")
                os.system(f"rm -rf {(str(file_path))}")
    wandb.finish()
    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()
