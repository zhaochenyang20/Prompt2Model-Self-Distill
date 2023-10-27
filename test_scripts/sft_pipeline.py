dataset_path = "/home/cyzhao/prompt2model_test/generation/generated_dataset/SQuAD_0.3_1.4_with_filtering"
model_path = "/data/ckpts/huggingface/models/models--lmsys--vicuna-7b-v1.5/snapshots/de56c35b1763eaae20f4d60efd64af0a9091ebe5"
ckpt_path = "/home/cyzhao/ckpt"

import gc

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

dataset = load_from_disk(dataset_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    use_flash_attention_2=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True,
    padding_side="left",
    trust_remote_code=True,
)


def formatting_func(example):
    return dict(
        text=f"### INPUT:  {example['input_col']}\n### OUTPUT: {example['output_col']}"
    )


mapped_dataset = dataset.map(formatting_func)

response_template_with_context = "\n### OUTPUT:"
response_template_ids = tokenizer.encode(
    response_template_with_context, add_special_tokens=False
)[2:]

data_collator = DataCollatorForCompletionOnlyLM(
    response_template_ids, tokenizer=tokenizer
)
training_args = TrainingArguments(
    output_dir=ckpt_path,
    do_eval=False,
    save_strategy="no",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=mapped_dataset,
    dataset_text_field="text",
    data_collator=data_collator,
)

trainer.train()

gc.collect()
torch.cuda.empty_cache()

model.save_pretrained("/home/cyzhao/cache")
