from vllm import LLM, SamplingParams

language_model = LLM(
    model="/data/ckpts/huggingface/models/models--lmsys--vicuna-7b-v1.5/snapshots/de56c35b1763eaae20f4d60efd64af0a9091ebe5"
)


sampling_params = SamplingParams(
    top_k=-1,
    top_p=0.1,
    temperature=1,
    max_tokens=500,
)

prompt = """### Instructions:

Your task is to generate an answer to a natural question. In this task, the input is a string that consists of both a question and a context passage. The context is a descriptive passage related to the question and contains the answer. And the question can range from Math, Cultural, Social, Geometry, Biology, History, Sports, Technology, Science, and so on.

### Examples:

[input]="Question: What is the capital city of France? Context: France, officially the French Republic, is a country primarily located in Western Europe, consisting of metropolitan France and several overseas regions and territories. The capital city of France is Paris, which is also its largest city. Paris is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral."
[output]="Paris"

[input]="Question: What is the capital city of Australia? Context: Australia, officially known as the Commonwealth of Australia, is a sovereign country comprising the mainland of the Australian continent, the island of Tasmania, and numerous smaller islands. It is the largest country in Oceania and the world's sixth-largest country by total area. Canberra is the capital city of Australia and is located in the Australian Capital Territory (ACT). The city was chosen as the capital in 1908 as a compromise between Sydney and Melbourne, the two largest cities in Australia."
[output]="Canberra"

[input]="Question: What is the largest desert in the world? Context: Deserts are dry, barren landscapes that receive very little rainfall. The largest desert in the world is the Sahara Desert, which covers an area of approximately 9.2 million square kilometers. It is located in northern Africa and spans across several countries, including Algeria, Chad, Egypt, Libya, Mali, Mauritania, Morocco, Niger, Sudan, and Tunisia. The Sahara Desert is known for its vast stretches of sand dunes and extreme temperatures."
[output]="Sahara Desert"

### New Input:

"Question: What is the largest ocean in the world? Context: Oceans are large bodies of saltwater that cover more than 70% of the Earth's surface. The largest ocean in the world is the Pacific Ocean, which stretches across approximately 63 million square miles. It is located between the Western Hemisphere and the Eastern Hemisphere and is bordered by Asia and Australia to the west, and the Americas to the east."

### Your Output:
"""

prompts = [prompt]

outputs = language_model.generate(prompts, sampling_params)
generated_inputs = [each.outputs[0].text for each in outputs]
generated_inputs[0]
