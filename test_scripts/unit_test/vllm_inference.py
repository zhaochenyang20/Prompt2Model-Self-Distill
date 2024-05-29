import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from vllm import LLM, SamplingParams
from prompt2model.utils.path import MODEL_PATH
from datasets import load_from_disk




default = MODEL_PATH

language_model = LLM(
    model="baichuan-inc/Baichuan2-13B-Chat",
    gpu_memory_utilization=0.9,
    swap_space = 16,
    tensor_parallel_size=1,
    trust_remote_code=True,
)

input_prompt = """
"\nA chat between a curious user and an artificial intelligence assistant.\nThe assistant gives concise answers to the user's questions.\nUSER: The artificial intelligence assistant only needs to help annotate label. The task is: In this task, given 2 input sentences, you must classify the relation between them. If the second sentence has a similar meaning to that of the first sentence then the output is 'B_entails_A', if the second sentence has the opposite meaning to the first sentence then it is classified as 'B_contradicts_A'. If you cannot clearly ascertain agreement/disagreement between the two sentences, the label is 'B_neutral_A'.\nASSISTANT: Okay.\nUSER: [input] = sentence_A: man is wearing a hard hat and dancing. sentence_B: There is no man with a hard hat dancing.\nASSISTANT: B_contradicts_A\nUSER: [input] = sentence_A: A baby is crying. sentence_B: A man is exercising.\nASSISTANT: B_neutral_A\nUSER: [input] = sentence_A: A tiger is pacing around a cage. sentence_B: A tiger is walking around a cage\nASSISTANT: B_entails_A\nUSER: [input] = sentence_A: A big cat is opening a plastic drawer with its paws and is jumping inside. sentence_B: A cat is opening a drawer and climbing inside\nASSISTANT:\n"
""".strip()
prompts = [input_prompt]

hyperparameter_choices = {}

sampling_params = SamplingParams(
    n=hyperparameter_choices.get("n", 1),
    best_of=hyperparameter_choices.get("best_of", 1),
    top_k=hyperparameter_choices.get("top_k", -1),
    temperature=hyperparameter_choices.get("temperature", 0),
    max_tokens=hyperparameter_choices.get("max_tokens", 500),
    logprobs=hyperparameter_choices.get("logprobs", 10),
)

output_sequence = language_model.generate(prompts, sampling_params)

generated_inputs = [
    [output.text for output in each.outputs] for each in output_sequence
]
generated_inputs
