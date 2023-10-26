dataset_path = "/home/cyzhao/prompt2model_test/generation/generated_dataset/SQuAD_0.3_1.4_with_filtering"
model_path = "/data/ckpts/huggingface/models/models--lmsys--vicuna-7b-v1.5/snapshots/de56c35b1763eaae20f4d60efd64af0a9091ebe5"

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

dataset = load_from_disk(dataset_path)
model = AutoModelForCausalLM.from_pretrained(model_path,
                                             device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True,
    padding_side="left",
    trust_remote_code=True,
)

def formatting_func(example):
    return dict(text=f"### INPUT:  {example['input_col']}\n### OUTPUT: {example['output_col']}")

mapped_dataset = dataset.map(formatting_func)

response_template_with_context = "\n### OUTPUT:"  # We added context here: "\n". This is enough for this tokenizer
response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]  # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]`

data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
trainer = SFTTrainer(
    model,
    train_dataset=mapped_dataset,
    dataset_text_field="text",
    data_collator=data_collator,
)

trainer.train()

model.save("/home/cyzhao/cache")