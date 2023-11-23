import gc
from functools import partial
from pathlib import Path

# import os

import datasets
import ray
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

from prompt2model.output_annotator.prompt_template import construct_meta_prompt
from prompt2model.prompt_parser import MockPromptSpec, TaskType

# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

#! TODO 改为新任务的 test set path

task_name = "task937" #test result 0.6583333333333333
experiment_name = 'NI_'+task_name+'_exp_1'

test_dataset = datasets.load_from_disk(
    f"/home/cyzhao/prompt2model_test/testdataset/NI/test/{task_name}"
)

#!     "/home/cyzhao/prompt2model_test/testdataset/SQuAD_transformed_test" SQuAD

# ckpt_path = Path("/home/cyzhao/ckpt")
inputs_dir = Path("/home/cyzhao/")

prompt_spec = MockPromptSpec(
    task_type=TaskType.TEXT_GENERATION,
    #! TODO 改 instruction 和 examples
    # TODO 改成自动化更改
    instruction="In this task, you are given a hypothesis and an update. The hypothesis sentence is a statement that speaks of a socially normative behavior. In other words, it is a generalizing statement about how we expect people to behave in society. The update provides additional contexts about the situation that might UNDERMINE or SUPPORT the generalization. An undermining context provides a situation that weakens the hypothesis. A supporting context provides a situation that strengthens the generalization. Your task is to output 'strengthener' or 'weakener' if the update supports or undermines the hypothesis, respectively",
    examples="""
[input]="Hypothesis: You should help your family with funeral expenses.\nUpdate: They have asked you to chip in"
[output]="strengthener"

[input]="Hypothesis: It's good to protect your property.\nUpdate: you don't care what happens to your property."
[output]="weakener"

[input]=“Hypothesis: You should help your family with funeral expenses.\nUpdate: You are not financially stable to help out”
[output]=“weakener”
""",
)

construct_prompt = partial(
    construct_meta_prompt,
    instruction=prompt_spec.instruction,
    examples=prompt_spec.examples,
)

tokenizer = AutoTokenizer.from_pretrained(
    "/data/ckpts/huggingface/models/models--lmsys--vicuna-7b-v1.5/snapshots/de56c35b1763eaae20f4d60efd64af0a9091ebe5",
    local_files_only=True,
    padding_side="left",
    trust_remote_code=True,
)


def map_func(example):
    example["model_input"] = construct_prompt(new_input=example["input_col"])
    example["model_output"] = example["output_col"]
    example["text"] = (
        example["model_input"] + example["model_output"] + tokenizer.eos_token
    )
    return example


test_dataset = test_dataset.map(map_func, load_from_cache_file=False)
prompts = test_dataset["model_input"]
GROUND_TRUTH = test_dataset["model_output"]

hyperparameter_choices = {}
sampling_params = SamplingParams(
    top_k=hyperparameter_choices.get("top_k", -1),
    top_p=hyperparameter_choices.get("top_p", 1),
    temperature=hyperparameter_choices.get("temperature", 0),
    max_tokens=hyperparameter_choices.get("max_tokens", 500),
)
MODEL_INPUTS = prompts
VALIDATION_DATASET = datasets.Dataset.from_dict(
    {"model_ouput": GROUND_TRUTH, "model_input": MODEL_INPUTS}
)

#! 这里测试轮次比较多，是为了看结果是否稳定
for _ in range(3):
    # vicuna base model "/data/ckpts/huggingface/models/models--lmsys--vicuna-7b-v1.5/snapshots/de56c35b1763eaae20f4d60efd64af0a9091ebe5"
    base_model = "/data/ckpts/huggingface/models/models--lmsys--vicuna-7b-v1.5/snapshots/de56c35b1763eaae20f4d60efd64af0a9091ebe5"
    # 改了这里的名字
    path = f"/data2/cyzhao/best_ckpt/NI/{experiment_name}"
    ray.init(ignore_reinit_error=True)
    tuned_vicuna = LLM(
        model=base_model, gpu_memory_utilization=0.5, tensor_parallel_size=1
    )
    tuned_vicuna_outputs = tuned_vicuna.generate(prompts, sampling_params)
    tuned_vicuna_generated_outputs = [
        each.outputs[0].text for each in tuned_vicuna_outputs
    ]
    index = 0
    for i in range(len(GROUND_TRUTH)):
        if (
            GROUND_TRUTH[i] in tuned_vicuna_generated_outputs[i]
            or tuned_vicuna_generated_outputs[i] in GROUND_TRUTH[i]
        ):
            index += 1
    print(index / len(GROUND_TRUTH))
    with open(inputs_dir / f"evaluate_10_times.txt", "a+") as file:
        file.write(
            f"\n\nresult of {_} th:\n\n------------------------------------------------{index / len(GROUND_TRUTH)}------------------------------------------------\n\n"
        )
    del tuned_vicuna
    #! 记得改名字
    evaluate_generated_content_path = inputs_dir / f"base_vicuna_{task_name}"
    print(f"Genrated contents are stored in {str(evaluate_generated_content_path)}")
    datasets.Dataset.from_dict(
        dict(
            model_output=tuned_vicuna_generated_outputs,
            model_input=prompts,
            groud_truth=GROUND_TRUTH,
        )
    ).save_to_disk(evaluate_generated_content_path)
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()
    destroy_model_parallel()
