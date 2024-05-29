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

    is_generation=False
    
    def map_func(example):
        assert prompt_spec.examples != ""
        example["model_input"] = construct_meta_prompt(
            instruction=prompt_spec.instruction,
            new_input=example["input_col"],
            examples=prompt_spec.examples,
            is_generation=is_generation,
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
        # local_files_only=True,
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
    if is_generation:
        response_template_with_context = "\n\n### Your Output:\n\n"
    else:
        response_template_with_context = "\nASSISTANT:\n"
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
        use_flash_attention_2=True,
        # baichuan 不行
        trust_remote_code=True,
    )
    wandb.init(project=task_name, name=run_name)
    wandb.watch(model)
    training_args = TrainingArguments(
        report_to="wandb",
        run_name=run_name,
        output_dir=str(ckpt_path),
        do_eval=False,
        save_strategy="no",
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
    mock_ckpt_path = ckpt_path / "checkpoint-1"
    model.save_pretrained(mock_ckpt_path)
    tokenizer.save_pretrained(mock_ckpt_path)
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
