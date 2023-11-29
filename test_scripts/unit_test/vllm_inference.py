from vllm import LLM, SamplingParams

path = "/data/cyzhao/best_ckpt/SQuAD"

default = "/data/ckpts/huggingface/models/models--deepseek-ai--deepseek-llm-7b-chat/snapshots/afbda8b347ec881666061fa67447046fc5164ec8"

language_model = LLM(
    model=default,
    gpu_memory_utilization=0.5,
    tensor_parallel_size=1,
)

input_prompt = """
Here are some [EXAMPLE] with important and meaningful content for your reference. You can refer to them to learn what the expected meaningful content is.

### [EXAMPLE]

"Question: What city did Super Bowl 50 take place in? Context: Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50."

"Question: What river runs through Warsaw? Context: Warsaw (Polish: Warszawa [varˈʂava] ( listen); see also other names) is the capital and largest city of Poland. It stands on the Vistula River in east-central Poland, roughly 260 kilometres (160 mi) from the Baltic Sea and 300 kilometres (190 mi) from the Carpathian Mountains. Its population is estimated at 1.740 million residents within a greater metropolitan area of 2.666 million residents, which makes Warsaw the 9th most-populous capital city in the European Union. The city limits cover 516.9 square kilometres (199.6 sq mi), while the metropolitan area covers 6,100.43 square kilometres (2,355.39 sq mi)."

"Question: The Ottoman empire controlled territory on three continents, Africa, Asia and which other? Context: The Ottoman Empire was an imperial state that lasted from 1299 to 1923. During the 16th and 17th centuries, in particular at the height of its power under the reign of Suleiman the Magnificent, the Ottoman Empire was a powerful multinational, multilingual empire controlling much of Southeast Europe, Western Asia, the Caucasus, North Africa, and the Horn of Africa. At the beginning of the 17th century the empire contained 32 provinces and numerous vassal states. Some of these were later absorbed into the empire, while others were granted various types of autonomy during the course of centuries."


-------------------------------------------------------------------------------------------

### [NEW INPUT]

""
Sure. I can help to generate a new input. Here is the new question and context:

Question: What was the capital of Ukraine during World War II?

Context: Ukraine is a country located at the crossroads of Central and Eastern Europe, beyond the Carpathian Mountains. Its capital, Kyiv (also known as Lviv at the time), was occupied by Nazi Germany from 1941 to 1944, following the German invasion of the USSR on 22 June 1941 during World War II."
""

-------------------------------------------------------------------------------------------

### [IMPORTANT]

Now, please extract the most important and meaningful content from the [NEW INPUT] that is similar to the [EXAMPLE] and reply with [info] [/info] to wrap the meaningful content

1. Extract all the most important and meaningful content similar to the [EXAMPLE].
2. Only reply with the most important and meaningful content from [NEW INPUT], you are not allowed to change any content of the [NEW INPUT].
3. If there is no meaningful content in the [NEW INPUT], reply with "[NO CONTENT]".

-------------------------------------------------------------------------------------------

### Your Response

"""

#! 换一个 Question Context 多测试
#! 用 [info] [/info] warp 起来
#! 如果没有就返回 [NO CONTENT]

prompts = [input_prompt]

sampling_params = SamplingParams(
    top_k=-1,
    top_p=1,
    temperature=0,
    max_tokens=500,
)

output_sequence = language_model.generate(prompts, sampling_params)

generated_inputs = [
    [output.text for output in each.outputs] for each in output_sequence
]
generated_inputs
