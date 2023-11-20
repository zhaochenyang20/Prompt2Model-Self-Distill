import json
import os
from itertools import product
from pathlib import Path
import csv


log_and_data_root = Path("/home/cyzhao") / "SQuAD_experiments_3"
# log_and_data_p2ms
# data_stash
ckpt_root = Path("/data/cyzhao/ckpt_data_p2ms")
log_and_data_root.mkdir(parents=True, exist_ok=True)
ckpt_root.mkdir(parents=True, exist_ok=True)
# 训练时能够用的显卡，加起来总共剩余的显存对于 7B model 需要接近 200G
CUDA_CONDITION = "0,1,2,3"
gpu_memory_utilization = 0.95
tensor_parallel_size = CUDA_CONDITION.count(",") + 1
# 进行 inference（除了训练之外的任何步骤）时，会分布在每张卡上，也即 tensor_parallel_size 就是所有能用的 CUDA
# gpu_memory_utilization 是在每张卡上的占比，比如 CUDA_CONDITION = "0,1,4,5", gpu_memory_utilization = 0.9
# 则每张卡都会占去全部显存的 0.9，会占用四张卡，推理效率极其高
# gpu_memory_utilization 越小，则 inference 越慢
# 然而，不是每张卡都是空的，比如 0 卡已经有人跑了 40G 了，那么 gpu_memory_utilization< 0.5


def read_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


# Function to write to CSV
def write_to_csv(file_path, header, data):
    with open(file_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        writer.writerows(data)


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
    # (20, 20, 50, 1.0, 0.3, 120, 1),
    # (20, 20, 50, 1.0, 0.4, 120, 3),
    # (20, 20, 50, 1.0, 0.6, 120, 3),
    # (40, 10, 50, 1.0, 0.3, 120, 3),
    # (20, 20, 50, 0.7, 0.3, 120, 3),
    # (20, 20, 50, 0.5, 0.3, 120, 3),
    # (40, 10, 50, 0.3, 0.3, 120, 3),
    # (20, 20, 40, 1.0, 0.3, 120, 3),
    # (20, 20, 30, 1.0, 0.3, 120, 3),
    # (10, 20, 30, 1.0, 0.3, 120, 3),
    # (30, 20, 30, 1.0, 0.3, 120, 3),
    # (40, 20, 30, 1.0, 0.3, 120, 3),
    # (50, 20, 30, 1.0, 0.3, 120, 3),
    (10, 20, 30, 1.0, 0.3, 120, 1),
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
        training_epochs,
    ) = parameter_tuple
    name = f"{task_name}_{generation_epochs}_{generation_batch_size}_{generation_top_k}_{generation_temperature}_{min_frequency}_{min_input_length}_{training_epochs}"
    log_and_data_path = log_and_data_root / name
    log_and_data_path.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_root / name
    ckpt_path.mkdir(parents=True, exist_ok=True)
    params = {
        "CUDA_CONDITION": CUDA_CONDITION,
        "task_name": task_name,
        "instruction": instruction,
        "examples": examples,
        "generation_epochs": generation_epochs,
        "generation_batch_size": generation_batch_size,
        "generation_top_k": generation_top_k,
        "generation_temperature": generation_temperature,
        "log_and_data_path": str(log_and_data_path),
        "ckpt_path": str(ckpt_path),
        "gpu_memory_utilization": gpu_memory_utilization,
        "min_frequency": min_frequency,
        "min_input_length": min_input_length,
        "training_epochs": training_epochs,
        "tensor_parallel_size": tensor_parallel_size,
    }
    with open(log_and_data_path / "config.json", "w") as f:
        json.dump(params, f, indent=4)
    required_paths = [
        log_and_data_path / "result_seed42_2.json",
        log_and_data_path / "inputs",
        log_and_data_path / "dataset",
    ]

    evaluate_result_path = log_and_data_path / "result_seed42_2.json"

    if evaluate_result_path.exists():
        evaluate_result = read_json(evaluate_result_path)
    else:
        evaluate_result = {}
        with open(evaluate_result_path, "w") as f:
            json.dump(evaluate_result, f, indent=4)

    csv_header = [
        "task_name",
        "generation_epochs",
        "generation_batch_size",
        "generation_top_k",
        "generation_temperature",
        "min_frequency",
        "min_input_length",
        "training_epochs",
    ] + ["epoch_" + str(i) for i in range(1, training_epochs + 1)]
    csv_data = []
    print(name)
    for experiment_folder in log_and_data_root.iterdir():
        if experiment_folder.is_dir():
            config_path = experiment_folder / "config.json"
            result_path = experiment_folder / "result_seed42_2.json"
            if config_path.exists() and result_path.exists():
                config = read_json(config_path)
                result = read_json(result_path)
                row = {key: config[key] for key in csv_header if key in config}
                row.update({"epoch_" + str(k): v for k, v in result.items()})
                csv_data.append(row)

    csv_file_path = log_and_data_root / "experiment_results.csv"
    # write_to_csv(csv_file_path, csv_header, csv_data)

    if (
        not all(path.exists() for path in required_paths)
        or len(list(evaluate_result.keys())) < training_epochs
    ):
        command = f"CUDA_VISIBLE_DEVICES={CUDA_CONDITION} python3 main.py --config={str(log_and_data_path / 'config.json')}"
        print(command)
        os.system(command)
