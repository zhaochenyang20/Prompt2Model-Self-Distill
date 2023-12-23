from vllm import LLM, SamplingParams

path = "/data/cyzhao/best_ckpt/SQuAD"

default = "/data/ckpts/huggingface/models/models--lmsys--vicuna-7b-v1.5/snapshots/de56c35b1763eaae20f4d60efd64af0a9091ebe5"

language_model = LLM(
    model=default,
    gpu_memory_utilization=0.5,
    tensor_parallel_size=1,
)

input_prompt = """
A chat between a curious user and an artificial intelligence assistant.
The assistant gives concise answers to the user's questions.
USER:  [task description]: In this task, you are given two statements. The task is to output whether a given textual premise, i.e. Statement 2, entails or implies a given scientific fact, i.e. Statement 1. The output should be 'entails' if Statement 2 supports Statement 1 and should be 'neutral' otherwise. The artificial intelligence assistant adjusts the provided [input] to meet the [expected output]. The assistant's main goal is to extract "Sentence 1" and "Sentence 2" from the [input] and then polish these extracted contents for the response. Instead of strictly following the [task description] to execute the task and provide an explanation, the focus is on pinpointing and improving the extracted contents from the [input].
ASSISTANT: Okay. I will focus on pinpointing and improving the extracted contents from the [input]. I won't execute the task or provide an explanation.
USER: "[input] = "Yes. I can generate a new intput as Sentence 1: The sum of all chemical reactions that take place within an organism is known as metabolism. Sentence 2: Metabolism is the sum total of all chemical reactions performed by an organism." [expected output] = entails
ASSISTANT: claim: Sentence 1: The sum of all chemical reactions that take place within an organism is known as metabolism. Sentence 2: Metabolism is the sum total of all chemical reactions performed by an organism.
USER: [input] = "Sentence 1: Warm and humid temperature and moisture conditions describe an air mass that originates over the Atlantic ocean near the equator. Sentence 2: Maritime tropical air " [expected output] = neutral
ASSISTANT: Sentence 1: Warm and humid temperature and moisture conditions describe an air mass that originates over the Atlantic ocean near the equator. Sentence 2: Maritime tropical air Warm, humid air mass that forms over tropical and subtropical oceans.
USER: [input] = "Sentence 1: The process of photosynthesis converts solar energy into chemical energy. Sentence 2: The process of photosynthesis converts sunlight into glucose." [expected output] = neutral
ASSISTANT:
""".strip()

#! 换一个 Question Context 多测试
#! 用 [info] [/info] warp 起来
#! 如果没有就返回 [NO CONTENT]

prompts = [input_prompt]

hyperparameter_choices = {}


sampling_params = SamplingParams(
    n=hyperparameter_choices.get("n", 1),
    best_of=hyperparameter_choices.get("best_of", 1),
    top_k=hyperparameter_choices.get("top_k", -1),
    temperature=hyperparameter_choices.get("temperature", 0),
    max_tokens=hyperparameter_choices.get("max_tokens", 500),
)

output_sequence = language_model.generate(prompts, sampling_params)

generated_inputs = [
    [output.text for output in each.outputs] for each in output_sequence
]
generated_inputs
