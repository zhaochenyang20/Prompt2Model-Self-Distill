from vllm import LLM, SamplingParams
from prompt2model.utils.path import MODEL_PATH

path = "/data/cyzhao/best_ckpt/SQuAD"

default = MODEL_PATH

language_model = LLM(
    model=default,
    gpu_memory_utilization=0.9,
    swap_space = 16,
    tensor_parallel_size=1,
    enforce_eager = True,
)

input_prompt = """
A chat between a curious user and an artificial intelligence assistant.\nThe assistant gives concise answers to the user's questions.\nUSER: The artificial intelligence assistant only needs to help annotate label. The task is: In this task, you're given a statement, and three sentences as choices. Your job is to determine which sentence clearly disagrees with the statement. Indicate your answer as '1', '2', or '3' corresponding to the choice number of the selected sentence. \nASSISTANT: Okay. \nUSER: [input] = Statement: Next to the MGM Grand you will find M and M World, four stories of merchandise and memorabilia dedicated to the candy that doesn't melt in your hand. Choices: 1. The candy has many fans who love its attractions. 2. There's four stories of memorabilia dedicated to a candy. 3. That particular candy melts and becomes difficult to eat.\nASSISTANT: 3\nUSER: [input] = Statment: I've forgotten his name now, confessed Tuppence. Choices: 1. Tuppence forgot his name. 2.Tuppence remembered his name later. 3. Tuppence never could forget his name.\nASSISTANT: 3\nUSER: [input] = Statement: One of the first organizational realignments taking place is in the Office of the Taxpayer Advocate. Choices: 1. The office of the taxpayer advocate is the last to be realigned. 2. The realignment is taking place over a few weeks. 3. The office of the taxpayer advocate is having an organizational realignment.\nASSISTANT: 1\nUSER: [input] = Statement: yeah i tell you what though if you go price some of those tennis shoes i can see why now you know they're getting up in the hundred dollar range. Choices: 1. The tennis shoes have a range of prices. 2. The tennis shoes can be in the hundred dollar range. 3. The tennis shoes are not over hundred dollars.\nASSISTANT: 3\nUSER: [input] = Statement: The new company policy states that all employees must work from home on Fridays. Choices: 1. The policy states that employees must work from home on Fridays. 2. The policy states that employees must work from home on Mondays. 3. The policy states that employees must work from home every day.\nASSISTANT:
""".strip()

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
