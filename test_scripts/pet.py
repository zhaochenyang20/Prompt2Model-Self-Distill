import os, json
from IPython import embed
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_MODE"] = "offline"

from datasets import load_from_disk, Dataset
from pathlib import Path
import os, re
from utils.tasks import task1516, task1529, task1612, task1615, task284, task329, task346
import gc
import re
from vllm import LLM, SamplingParams
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
import datasets
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from unit_test.baseline_test_ni import exact_match_score, rouge_l_score

dataset_root = Path("/home/azureuser/p2mss/prompt2model_test/testdataset/NI/test")

classification_tasks = [
    "task1516", # 直接在结尾加入 The Premise is [TODO] with the Hypothesis. [END]
    "task1529", # 直接在结尾加入 The Premise is [TODO] with the Hypothesis. [END]
    "task1612", # 直接在结尾加入 The sentence_A is in [TODO] relationship with the sentence_B. [END]
    "task1615", # 直接在结尾加入 The sentence_A is in [TODO] relationship with the sentence_B. [END]
    "task284", # 直接在结尾加上 ”This Is a [TODO] movie review.“ [END]
    "task329", # 结尾加上 The pronoun refers to [TODO] in the sentence. [END]
    "task346", # 结尾加上 The part-of-speech tag of the given word in the question is [TODO] equal to the given POS tag. [END]
]

generation_tasks = [
    "task1345", # The paraphrased sentence is [TODO] to the given sentence. [END]
    "task281", #  [TODO] is shared between all three sentences. [END]
    "task1562", # The paraphrased sentence is [TODO] to the given sentence. [END]
    "task1622" # The converted sentence is [TODO] to the given sentence. [END]
    ]


tokenizer = AutoTokenizer.from_pretrained(
    'lmsys/vicuna-7b-v1.5',
    padding_side="left",
    trust_remote_code=True,
)

for task_name in generation_tasks:
    task_config_for_generation_tasks = None
    with open("/home/azureuser/p2mss/p2mss/main/NI_tasks/tasks.json", "r", encoding="utf-8") as json_file:
        all_tasks = json.load(json_file)
    for task in all_tasks:
        if task["task_name"] == task_name:
            task_config_for_generation_tasks = (
                task["task_name"],
                task["task_instruction"],
                task["examples"],
                task["expected_content"],
                f"/home/azureuser/p2mss/prompt2model_test/testdataset/NI/eval/{task_name}",
                f"/home/azureuser/p2mss/prompt2model_test/testdataset/NI/test/{task_name}",
                task.get("optional_list", []),
                task.get("metric", "rouge"),
            )
            break

    task_name, instruction, examples, expected_content, evaluation_dataset_path, test_set_path, optional_list, metric  = task_config_for_generation_tasks
    
    def construct_cls_training_dataset():
        matches = re.findall(
            r'\[input\]="(.*?)"\s*\[output\]="(.*?)"',
            examples,
            re.DOTALL,
        )
        assert matches != []
        training_inputs = []
        if task in [task1516, task1529]:
            for example in matches:
                training_inputs.append(task.task_instruction + " " + example[0] + f" The Premise is [TODO] with the Hypothesis. [END] " + example[1])
        elif task in [task1612, task1615]:
            for example in matches:
                training_inputs.append(task.task_instruction + " " + example[0] + f" The sentence_A is in [TODO] relationship with the sentence_B. [END] " + example[1])
        elif task in [task284]:
            for example in matches:
                training_inputs.append(task.task_instruction + " " + example[0] + f" This Is a [TODO] movie review. [END] " + example[1])
        elif task in [task329]:
            for example in matches:
                training_inputs.append(task.task_instruction + " " + example[0] + f" The pronoun refers to [TODO] in the sentence. [END] " + example[1])
        elif task in [task346]:
            for example in matches:
                training_inputs.append(task.task_instruction + " " + example[0] + f" The part-of-speech tag of the given word in the question is [TODO] equal to the given POS tag. [END] " + example[1])
        
        training_outputs = [example[1] for example in matches]
        return Dataset.from_dict({"text": training_inputs, "output_col": training_outputs})

    def construct_gen_training_dataset():
        matches = re.findall(
            r'\[input\]="(.*?)"\s*\[output\]="(.*?)"',
            examples,
            re.DOTALL,
        )
        assert matches != []
        training_inputs = []
        if task_name in ["task1345", "task1562"]:
            for example in matches:
                training_inputs.append(instruction + " " + example[0] + f" The paraphrased sentence is [TODO] to the given sentence. [END] " + example[1] + "</s>")
        elif task_name in ["task281"]:
            for example in matches:
                training_inputs.append(instruction + " " + example[0] + f" [TODO] is shared between all three sentences. [END] " + example[1] + "</s>")
        elif task_name in ["task1622"]:
            for example in matches:
                training_inputs.append(instruction + " " + example[0] + f" The converted sentence is [TODO] to the given sentence. [END] " + example[1] + "</s>")
        
        training_outputs = [example[1] for example in matches]
        return Dataset.from_dict({"text": training_inputs, "output_col": training_outputs})

    training_dataset = construct_gen_training_dataset()

    tokenizer = AutoTokenizer.from_pretrained(
        'lmsys/vicuna-7b-v1.5',
        padding_side="left",
        trust_remote_code=True,
    )

    response_template_with_context = "[END]"
    response_template_ids = tokenizer.encode(
        response_template_with_context, add_special_tokens=False
    )[2:]
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template_ids, tokenizer=tokenizer
    )
    ckpt_path = Path("/home/azureuser/p2mss/p2mss/ckpt_data_p2ms/pet_gen") / task_name
    ckpt_path.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        'lmsys/vicuna-7b-v1.5',
        device_map="auto",
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=True,
    )
    wandb.init(project=task_name, name=task_name)
    wandb.watch(model)

    training_args = TrainingArguments(
        report_to="wandb",
        run_name=task_name,
        output_dir=str(ckpt_path),
        do_eval=False,
        save_strategy="no",  # 不在训练过程中保存检查点
        evaluation_strategy="no",
        logging_steps=4,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=training_dataset,
        dataset_text_field="text",
        data_collator=data_collator,
        max_seq_length=2000,
    )

    wandb.config.update(training_args.to_dict(), allow_val_change=True)
    trainer.train(resume_from_checkpoint=False)
    model.save_pretrained(str(ckpt_path))
    tokenizer.save_pretrained(str(ckpt_path))
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


    dataset = load_from_disk(dataset_root / task_name)
    
    def map_func_cls(example):
        if task.task_name in ["task1516", "task1529"]:
            example["model_input"] = task.task_instruction + " " + example["input_col"] + " The Premise is [TODO] with the Hypothesis. [END]"
        elif task.task_name in ["task1612", "task1615"]:
            example["model_input"] = task.task_instruction + " " + example["input_col"] + " The sentence_A is in [TODO] relationship with the sentence_B. [END]"
        elif task.task_name in ["task284"]:
            example["model_input"] = task.task_instruction + " " + example["input_col"] + " This Is a [TODO] movie review. [END]"
        elif task.task_name in ["task329"]:
            example["model_input"] = task.task_instruction + " " + example["input_col"] + " The pronoun refers to [TODO] in the sentence. [END]"
        elif task.task_name in ["task346"]:
            example["model_input"] = task.task_instruction + " " + example["input_col"] + " The part-of-speech tag of the given word in the question is [TODO] equal to the given POS tag. [END]"
        example["model_output"] = example["output_col"]
        return example

    def map_func_gen(example):
        if task_name in ["task1345", "task1562"]:
            example["model_input"] = instruction + " " + example["input_col"] + f" The paraphrased sentence is [TODO] to the given sentence. [END] "
        elif task_name in ["task281"]:
            example["model_input"] = instruction + " " + example["input_col"] + f" [TODO] is shared between all three sentences. [END] "
        elif task_name in ["task1622"]:
            example["model_input"] = instruction + " " + example["input_col"] + f" The converted sentence is [TODO] to the given sentence. [END] "
        example["model_output"] = example["output_col"]
        return example

    dataset = dataset.map(map_func_gen, load_from_cache_file=False)

    # def find_ckpt_path():
    #     ckpt_path = Path("/home/azureuser/p2mss/p2mss/ckpt_data_p2ms/pet_gen") / task_name
    #     ckpt_files = list(ckpt_path.glob("checkpoint-*"))
    #     if ckpt_files == []:
    #         return None
    #     ckpt_files.sort(key=lambda x: int(re.search(r"checkpoint-(\d+)", str(x)).group(1)))
    #     return str(ckpt_files[-1])

    model = LLM(
        model= str(Path("/home/azureuser/p2mss/p2mss/ckpt_data_p2ms/pet_gen") / task_name),
        gpu_memory_utilization=0.9,
        swap_space = 16,
        tensor_parallel_size=1,
    )

    length = len(dataset)
    
    if length >= 1000:
        length = 1000

    prompts = dataset["model_input"][:1000]
    GROUND_TRUTH = dataset["model_output"][:1000]
    hyperparameter_choices = {}

    # change this to the same params as inference
    sampling_params = SamplingParams(
        top_k=hyperparameter_choices.get("top_k", -1),
        top_p=hyperparameter_choices.get("top_p", 1),
        temperature=hyperparameter_choices.get("temperature", 0),
        max_tokens=hyperparameter_choices.get("max_tokens", 50),
        presence_penalty = 1.0,
        frequency_penalty = 1.0,
    )

    model_outputs = model.generate(prompts, sampling_params)
    decoded_outputs = []
    for idx, _ in enumerate(model_outputs):
        outputs = [
            output.text.strip()
            for output in model_outputs[idx].outputs
            if (output.text is not None and output.text != "")
        ]
        decoded_outputs.append(outputs[0])

    print(decoded_outputs[:10])

    evaluate_result = rouge_l_score(GROUND_TRUTH, decoded_outputs)
    
    print(f"{task_name} test => {evaluate_result}")
    # #TODO change file name every day
    evaluate_generated_content_path = Path("/home/azureuser/p2mss/p2mss/ckpt_data_p2ms/pet_gen") / f"generated_answer_{task_name}"
    datasets.Dataset.from_dict(
        dict(
            model_output=decoded_outputs,
            model_input=prompts,
            groud_truth=GROUND_TRUTH,
        )
    ).save_to_disk(evaluate_generated_content_path)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    destroy_model_parallel()
    
    # def map_func(example):
    #     def extract_before_punctuation(text):
    #         # 找到第一个 . 或 ? 的位置
    #         dot_index = text.find('.')
    #         question_index = text.find('?')
            
    #         # 如果都没有找到，返回原字符串
    #         if dot_index == -1 and question_index == -1:
    #             return text
            
    #         # 处理没有找到 . 或 ? 的情况
    #         if dot_index == -1:
    #             return text[:question_index]
    #         if question_index == -1:
    #             return text[:dot_index]
            
    #         # 返回最靠前的一个位置之前的部分
    #         return text[:min(dot_index, question_index)]
    #     example["model_output"] = extract_before_punctuation(example["model_output"])
    #     return example

    # dataset = load_from_disk(evaluate_generated_content_path).map(map_func, load_from_cache_file=False)
    # dataset = load_from_disk(evaluate_generated_content_path)
    # evaluate_result = rouge_l_score(dataset["groud_truth"], dataset["model_output"])
    # print(f"{task_name} test => {evaluate_result}")