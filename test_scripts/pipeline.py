import json
import os
from itertools import product
from pathlib import Path

root_dir = Path("/home/cyzhao/ckpt_data_p2ms")
root_dir.mkdir(parents=True, exist_ok=True)
# 训练时能够用的显卡，加起来总共剩余的显存对于 7B model 需要接近 200G
CUDA_CONDITION = "0,1,2,3,4,5,6,7"
# 进行 inference，也即除了训练之外的任何步骤，所能占用的单卡比例
# inference 只会用 CUDA_CONDITION 的第一张卡
# 比如 CUDA_CONDITION 是 0,1,2, 则 inference 会占用 0 卡的 gpu_memory_utilization 这么多显存
# gpu_memory_utilization 越小，则 inference 越慢，理论上不该低于 28 / 80 = 0.35
gpu_memory_utilization = 0.8

tasks = [
    (
        "SQuAD",
        "Your task is to generate an answer to a natural question. In this task, the input is a string that consists of both a question and a context passage. The context is a descriptive passage related to the question and contains the answer. And the question can range from Math, Cultural, Social, Geometry, Biology, History, Sports, Technology, Science, and so on.",
        """
[input]="Question: What city did Super Bowl 50 take place in? Context: Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50."
[output]="Santa Clara"

[input]="Question: What river runs through Warsaw? Context: Warsaw (Polish: Warszawa [varˈʂava] ( listen); see also other names) is the capital and largest city of Poland. It stands on the Vistula River in east-central Poland, roughly 260 kilometres (160 mi) from the Baltic Sea and 300 kilometres (190 mi) from the Carpathian Mountains. Its population is estimated at 1.740 million residents within a greater metropolitan area of 2.666 million residents, which makes Warsaw the 9th most-populous capital city in the European Union. The city limits cover 516.9 square kilometres (199.6 sq mi), while the metropolitan area covers 6,100.43 square kilometres (2,355.39 sq mi)."
[output]="Vistula River"

[input]="Question: The Ottoman empire controlled territory on three continents, Africa, Asia and which other? Context: The Ottoman Empire was an imperial state that lasted from 1299 to 1923. During the 16th and 17th centuries, in particular at the height of its power under the reign of Suleiman the Magnificent, the Ottoman Empire was a powerful multinational, multilingual empire controlling much of Southeast Europe, Western Asia, the Caucasus, North Africa, and the Horn of Africa. At the beginning of the 17th century the empire contained 32 provinces and numerous vassal states. Some of these were later absorbed into the empire, while others were granted various types of autonomy during the course of centuries."
[output]="Europe"
""",
    )
]

# generation_epochs, generation_batch_size, generation_top_k, generation_temperature
# min_frequency_of_self_consitency, min_input_length
# training_epochs
parameter_tuples = [
    (1, 10, 50, 1.0, 0.2, 120, 3),
    # (20, 20, 50, 1.0, 0.3, 120, 3),
    # (20, 20, 50, 1.0, 0.4, 120, 3),
    # (20, 20, 50, 1.0, 0.6, 120, 3),
]
for task, parameter_tuple in product(tasks, parameter_tuples):
    task_name, instruction, examples = task
    (
        generation_epochs,
        generation_batch_size,
        generation_top_k,
        generation_temperature,
        min_frequency,
        min_input_length,
        training_epochs
    ) = parameter_tuple
    store_path = (
        root_dir
        / f"{task_name}_{generation_epochs}_{generation_batch_size}_{generation_top_k}_{generation_temperature}_{min_frequency}_{min_input_length}_{training_epochs}"
    )
    store_path.mkdir(parents=True, exist_ok=True)
    params = {
        "CUDA_CONDITION": CUDA_CONDITION,
        "task_name": task_name,
        "instruction": instruction,
        "examples": examples,
        "generation_epochs": generation_epochs,
        "generation_batch_size": generation_batch_size,
        "generation_top_k": generation_top_k,
        "generation_temperature": generation_temperature,
        "store_path": str(store_path),
        "gpu_memory_utilization": gpu_memory_utilization,
        "min_frequency": min_frequency,
        "min_input_length": min_input_length,
        "training_epochs": training_epochs,
    }
    with open(store_path / "config.json", "w") as f:
        json.dump(params, f, indent=4)
        command = f"CUDA_VISIBLE_DEVICES={CUDA_CONDITION} python3 main.py --config={str(store_path / 'config.json')}"
        print(command)
    required_paths = [
        store_path / "model",
        store_path / "result.txt",
        store_path / "inputs",
        store_path / "dataset",
    ]
    if not all(path.exists() for path in required_paths):
        os.system(
            f"CUDA_VISIBLE_DEVICES={CUDA_CONDITION} python3 main.py --config={store_path / 'config.json'}"
        )
