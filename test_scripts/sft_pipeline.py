import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WANDB_MODE"] = "offline"

import gc
import re
import wandb
from functools import partial
from pathlib import Path
import json
import datasets
import torch
import shutil
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

from prompt2model.output_annotator import construct_meta_prompt
from prompt2model.prompt_parser import MockPromptSpec, TaskType
from prompt2model.utils.path import ROOT, STORE_ROOT, MODEL_PATH
from prompt2model.utils import count_tokens_from_string
from utils.inference import vllm_inference

import string
import re
from collections import Counter

def store_evaluation_content(
    content_store_path, tuned_model_generated_outputs, prompts, GROUND_TRUTH
):
    print(f"Genrated contents are stored in {content_store_path}")
    datasets.Dataset.from_dict(
        dict(
            model_output=tuned_model_generated_outputs,
            model_input=prompts,
            groud_truth=GROUND_TRUTH,
        )
    ).save_to_disk(content_store_path)

def find_last_occurrence(model_output: str, labels: list[str]) -> str:
    pattern = '|'.join(re.escape(label) for label in labels)
    regex = re.compile(pattern)
    matches = list(regex.finditer(model_output))
    return matches[-1].group() if matches else None

# cited from https://github.com/allenai/natural-instructions/blob/55a365637381ce7f3748fa2eac7aef1a113bbb82/eval/automatic/evaluation.py#L24
def normalize_answer(s):
    """Lower text and remove punctuation, and extra whitespace."""

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))

def exact_match(prediction, ground_truth, xlingual=False):
    # small changed based on our current code
    if prediction is None:
        return 0
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def exact_match_score(
    GROUND_TRUTH,
    tuned_model_generated_outputs,
):
    labels = list(Counter(GROUND_TRUTH).keys())
    index = 0
    n = len(GROUND_TRUTH)
    for i in range(n):
        index += exact_match(find_last_occurrence(tuned_model_generated_outputs[i], labels), GROUND_TRUTH[i])
    score = index / len(GROUND_TRUTH)
    return score

def lcs_length_dp(x, y):
    """Compute the length of the longest common subsequence between two strings using dynamic programming."""
    m, n = len(x), len(y)
    dp_table = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                dp_table[i][j] = 0
            elif x[i - 1] == y[j - 1]:
                dp_table[i][j] = dp_table[i - 1][j - 1] + 1
            else:
                dp_table[i][j] = max(dp_table[i - 1][j], dp_table[i][j - 1])

    return dp_table[m][n]

def rouge_l_score(GROUND_TRUTH, tuned_model_generated_outputs):
    scores = []
    for gt, gen in zip(GROUND_TRUTH, tuned_model_generated_outputs):
        lcs = lcs_length_dp(gt, gen)
        if lcs == 0:
            scores.append(0)
            continue
        precision = lcs / len(gen)
        recall = lcs / len(gt)
        f_measure = (2 * precision * recall) / (precision + recall)
        scores.append(f_measure)
    return sum(scores) / len(scores)

def extract_number(s):
    return int(re.search(r"\d+", s).group())

def check_and_remove_checkpoints(ckpt_path):
    required_files = [
        "config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
        "tokenizer.json",
        "tokenizer.model",
    ]

    sorted_list = sorted(
        [each for each in os.listdir(ckpt_path) if each.startswith("checkpoint-")],
        key=extract_number,
    )

    for each in sorted_list:
        checkpoint_path = os.path.join(ckpt_path, each)
        if all(
            os.path.exists(os.path.join(checkpoint_path, file))
            for file in required_files
        ):
            print(f"Checkpoint '{each}' is complete.")
        else:
            print(f"Checkpoint '{each}' is incomplete and will be removed.")
            shutil.rmtree(checkpoint_path)

    return len(
        sorted(
            [each for each in os.listdir(ckpt_path) if each.startswith("checkpoint-")],
            key=extract_number,
        )
    )

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
    per_device_train_batch_size=1,
    exact_match=True
):

    def filter_func(example):
        return example["output_col"] is not None and example["input_col"] is not None

    dataset = datasets.load_from_disk(log_and_data_path +"/"+ "dataset").filter(filter_func)

    is_generation=False

    result_path = f"/home/azureuser/p2mss/p2mss/classification_14/NI_{task_name}_exp_14/best_validation_result.json"
    if not exact_match:
        result_path = f"/home/azureuser/p2mss/p2mss/generation_11/NI_{task_name}_exp_11/best_validation_result.json"
        is_generation = True
    with open(result_path, 'r') as file:
        data = json.load(file)
        evaluate_result_path = data.get("evaluate_result_path", "")
        dataset_path = '/'.join(evaluate_result_path.split('/')[:-1]) + '/dataset'
    dataset = load_from_disk(dataset_path)
    # TODO change n
    n = 16
    dataset = dataset.select(range(n))

    def map_func(example):
        assert prompt_spec.examples != ""
        example["model_input"] = construct_meta_prompt(
            instruction=prompt_spec.instruction,
            examples=prompt_spec.examples,
            new_input=example["input_col"],
            is_generation=is_generation,
            few_shots_prompt = ''
        )
        example["model_output"] = example["output_col"]
        example["text"] = (
            example["model_input"] + example["model_output"] + tokenizer.eos_token
        )
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
    
    complete_ckpts = check_and_remove_checkpoints(ckpt_path)
    resume_from_checkpoint = False if complete_ckpts == 0 else True
    
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

def validate_or_test(
    evaluation_dataset_path,
    test_dataset_path,
    ckpt_path,
    instruction,
    examples,
    gpu_memory_utilization,
    tensor_parallel_size,
    test_content_store_path=None,
    log_and_data_path=None,
    metric="exact_match",
):


    def map_func(example):
        example["model_input"] = construct_meta_prompt(
            instruction=instruction,
            new_input=example["input_col"],
            examples=examples,
            is_generation= (metric == 'exact_match'),
            few_shots_prompt=''
        )
        example["model_output"] = example["output_col"]
        return example

    # first evaluate 

    loaded_dataset = datasets.load_from_disk(evaluation_dataset_path)
    test_dataset = loaded_dataset.map(map_func, load_from_cache_file=False)
    test_dataset = test_dataset.filter(
        lambda x: (
            count_tokens_from_string(x["model_input"], "vicuna") <= 3000
            and count_tokens_from_string(x["model_output"], "vicuna") <= 500
        )
    )
    prompts = test_dataset["model_input"]
    GROUND_TRUTH = test_dataset["model_output"]

    sorted_list = sorted(
        [each for each in os.listdir(ckpt_path) if ("checkpoint" in each and "tmp" not in each)],
        key=extract_number,
    )
    
    evaluate_result = {}
    for ckpt_index, each in enumerate(sorted_list):
        model_path = ckpt_path / each
        tuned_model_generated_outputs = [each.strip() for each in vllm_inference(
            model_path, gpu_memory_utilization, tensor_parallel_size, prompts
        )]
        if metric == "exact_match":
            score = exact_match_score(GROUND_TRUTH, tuned_model_generated_outputs)
        else:
            score = rouge_l_score(GROUND_TRUTH, tuned_model_generated_outputs)
        evaluate_result[model_path] = score
        name = str(log_and_data_path).split("/")[-1]
        print(
            f"\n\nresult of {name} \n path = {model_path} \n epoch {ckpt_index + 1}\n\n------------------------------------------------\n\n{score}\n\n------------------------------------------------\n\n"
        )
        evaluate_generated_content_path = log_and_data_path / "generated_contents"
        evaluate_generated_content_path.mkdir(parents=True, exist_ok=True)
        content_store_path = str(
            evaluate_generated_content_path / str(ckpt_index + 1)
        )
        store_evaluation_content(
            content_store_path, tuned_model_generated_outputs, prompts, GROUND_TRUTH
        )

    best_model_path = max(evaluate_result, key=evaluate_result.get)


    #! test
        
    loaded_dataset = datasets.load_from_disk(test_dataset_path)
    test_dataset = loaded_dataset.map(map_func, load_from_cache_file=False)
    # remove since we're trying to get as much as possible, more than 3000 + 500
    test_dataset = test_dataset.filter(
        lambda x: (
            count_tokens_from_string(x["model_input"], "vicuna") <= 3000
            and count_tokens_from_string(x["model_output"], "vicuna") <= 500
        )
    )
    prompts = test_dataset["model_input"]
    GROUND_TRUTH = test_dataset["model_output"]
    
    tuned_model_generated_outputs = vllm_inference(
        best_model_path, gpu_memory_utilization, tensor_parallel_size, prompts
    )
    if metric == "exact_match":
        score = exact_match_score(GROUND_TRUTH, tuned_model_generated_outputs)
    else:
        score = rouge_l_score(GROUND_TRUTH, tuned_model_generated_outputs)
    print(
        f"\n\nresult of {ckpt_path}\n\n------------------------------------------------\n\n{score}\n\n------------------------------------------------\n\n"
    )
    store_evaluation_content(
        test_content_store_path,
        tuned_model_generated_outputs,
        prompts,
        GROUND_TRUTH,
    )


# TODO: store ckpt or call test ahd store model output
model_path = Path(
    MODEL_PATH
)

# classification
# task 1529
# prompt_spec = MockPromptSpec(
#     task_type=TaskType.CLASSIFICATION,
#     instruction="You are given two sentences. You have to find if there is entailment or agreement of the Hypothesis by the Premise. From the given pair of sentences, you should identify if there is enough information in the Premise to support the claim made in the Hypothesis. The Premise may not exactly be the same as Hypothesis. Your task is to return 'entails' if the premise supports hypothesis else return 'neutral'.",
#     examples="""
# [input]="Premise: Lyme Disease is caused by a bacterium that's transmitted by tick bite, but many infected people don't remember a bite. \n Hypothesis: Lyme disease is caused by bacteria."
# [output]="entails"
# [input]="Premise: Corolla Collective term for all the petals of a flower, these petals may be separate or fused together. \n Hypothesis: All of the petals together are called a corolla."
# [output]="entails"
# [input]="Premise: This can be dangerous to both plants and animals. \n Hypothesis: Nematodes can be a parasite of both."
# [output]="neutral"
# [input]="Premise: The liver is divided into the right lobe and left lobes. \n Hypothesis: The gallbladder is near the right lobe of the liver."
# [output]="neutral"
# """
# )
# task 1612
prompt_spec = MockPromptSpec(
    task_type=TaskType.CLASSIFICATION,
    instruction="In this task, you're given a pair of sentences, sentence 1 and sentence 2. Your job is to choose whether the two sentences clearly agree (entailment)/disagree (contradiction) with each other, or if this cannot be determined (neutral). Your answer must be in the form of the numbers 0 (entailment), 1 (neutral), or 2(contradiction).",
    examples="""
[input]="sentence_A: A dancer is dancing on the stage. sentence_B: A girl is giving dance performance on the dais."
[output]="0"
[input]="sentence_A: The crowd is cheering at her dance performance. sentence_B: The group is enjoying while eating food."
[output]="1"
[input]="sentence_A: A man is standing and has tears of joy seeing the dance performance. sentence_B: There is no man standing with happiness seeing the dance."
[output]="2"
"""
)


# TODO change task name here
task_name = 'task1612'
# TODO change task pramas here
prompt_spec = prompt_spec
# TODO change log_and_data_path, choose best ckpt generated dataset to finetune, same as self-icl
# log_and_data_path = "/home/azureuser/p2mss/p2mss/classification_14/NI_task1529_exp_14/task1529_1.0_True_False_40_14"
log_and_data_path = "/home/azureuser/p2mss/p2mss/classification_14/NI_task1612_exp_14/task1612_1.0_False_False_40_14"
ckpt_path = Path(STORE_ROOT+f"/ckpt_data_p2ms/few_finetune_{task_name}")
pretrain_model_path = Path(MODEL_PATH)
training_epochs = 6
resume_from_checkpoint = False
run_name = log_and_data_path.split('/')[-2] + 'x' # to be different from previous runs
task_name = run_name.split('_')[1]

finetune_vicuna(
    prompt_spec,
    log_and_data_path,
    ckpt_path,
    pretrain_model_path,
    training_epochs,
    resume_from_checkpoint,
    run_name,
    task_name,
    max_seq_length=2000,
    per_device_train_batch_size=1,
    exact_match=True
)


evaluation_dataset_path = '/home/azureuser/p2mss/prompt2model_test/testdataset/NI/eval/' + task_name
test_dataset_path = '/home/azureuser/p2mss/prompt2model_test/testdataset/NI/test/' + task_name
ckpt_path = Path(STORE_ROOT+f"/ckpt_data_p2ms/few_finetune_{task_name}")
instruction = prompt_spec.instruction
examples = prompt_spec.examples
gpu_memory_utilization = 0.9
tensor_parallel_size = 1
# change path
# log_and_data_path = Path(ROOT) / 'task1529_x_finetune'
log_and_data_path = Path(ROOT) / 'task1612_x_finetune'
test_content_store_path = log_and_data_path / 'best_few_shot_result'
metric="exact_match"

validate_or_test(
    evaluation_dataset_path,
    test_dataset_path,
    ckpt_path,
    instruction,
    examples,
    gpu_memory_utilization,
    tensor_parallel_size,
    test_content_store_path=test_content_store_path,
    log_and_data_path=log_and_data_path,
    metric="exact_match",
)


# add test and evaluation later and that can work! Before run you need to make sure the exact
# metric is correct

