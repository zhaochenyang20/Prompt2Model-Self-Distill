from vllm import LLM, SamplingParams

path = "/data/cyzhao/best_ckpt/SQuAD"

default = "/data/ckpts/huggingface/models/models--lmsys--vicuna-7b-v1.5/snapshots/de56c35b1763eaae20f4d60efd64af0a9091ebe5"

language_model = LLM(
    model=default,
    gpu_memory_utilization=0.5,
    tensor_parallel_size=1,
)

input_prompt = """\nA chat between a curious user and an artificial intelligence assistant.\nThe assistant gives concise answers to the user's questions.\nUSER: The artificial intelligence assistant only needs to help annotate label. The task is: In this task, you are given a premise, a hypothesis, and an update. The premise sentence describes a real-world situation and is always assumed to be true. The hypothesis sentence describes an assumption or inference that you might make about that situation having read the premise. The update provides additional information about the situation that might weaken or strengthen the hypothesis. A weakener is a statement that weakens the hypothesis. It makes you much less likely to believe the hypothesis is true. A strengthener is a statement that strengthens the hypothesis. It makes you much more likely to believe the hypothesis is true. Your task is to output 'strengthener' or 'weakener' if the update strengths or weakens the hypothesis, respectively. \nASSISTANT: Okay. \nUSER: [input] = Premise: PersonX seems interested\nHypothesis: PersonX then good activity\nUpdate: PersonX was a good student\nASSISTANT: strengthener\nUSER: [input] = Premise: PersonX seems interested\nHypothesis: PersonX then good activity\nUpdate: PersonX was faking to get close to a girl\nASSISTANT: weakener\nUSER: [input] = \nPremise: PersonX seems interested\nHypothesis: PersonX then good activity\nUpdate: PersonX was faking to get close to a girl\nASSISTANT: \n"""

#! 换一个 Question Context 多测试
#! 用 [info] [/info] warp 起来
#! 如果没有就返回 [NO CONTENT]

prompts = [input_prompt]

hyperparameter_choices = {}

sampling_params = SamplingParams(
    n=hyperparameter_choices.get("n", 10),
    best_of=hyperparameter_choices.get("best_of", 20),
    top_k=-1,
    top_p=1,
    temperature=0.2,
    max_tokens=500,
)

sampling_params = SamplingParams(
    n=hyperparameter_choices.get("n", 10),
    best_of=hyperparameter_choices.get("best_of", 20),
    top_k=hyperparameter_choices.get("top_k", 10),
    temperature=hyperparameter_choices.get("temperature", 0.2),
    max_tokens=hyperparameter_choices.get("max_tokens", 500),
)

output_sequence = language_model.generate(prompts, sampling_params)

generated_inputs = [
    [output.text for output in each.outputs] for each in output_sequence
]
generated_inputs
