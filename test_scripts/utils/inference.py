import gc

import torch
from vllm import LLM, SamplingParams
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel


def vllm_inference(model_path, gpu_memory_utilization, tensor_parallel_size, prompts):
    hyperparameter_choices = {}
    sampling_params = SamplingParams(
        top_k=hyperparameter_choices.get("top_k", -1),
        top_p=hyperparameter_choices.get("top_p", 1),
        temperature=hyperparameter_choices.get("temperature", 0),
        max_tokens=hyperparameter_choices.get("max_tokens", 500),
    )
    model = LLM(
        model=str(model_path),
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=1,
    )
    model_outputs = model.generate(prompts, sampling_params)
    model_generated_outputs = [each.outputs[0].text for each in model_outputs]
    del model
    gc.collect()
    torch.cuda.empty_cache()
    destroy_model_parallel()
    return model_generated_outputs
