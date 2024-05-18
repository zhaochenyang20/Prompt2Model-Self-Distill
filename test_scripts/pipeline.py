import csv
import json
import os

# TODO: change card
os.environ["WANDB_MODE"] = "offline"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


TENSOR_SIZE = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

from pathlib import Path
import itertools
from prompt2model.utils.path import ROOT, STORE_ROOT, TEST_DATA_ROOT
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

# TODO change experiment rank
experiment_rank = 33
# 20 for scaling experiment
# 21 for 281 regeneration 
# 22 for debugging
# 31 for no filter generation
# 32 for no filter generation
# 33 for new generation

gpu_memory_utilization = 0.9
# 如果别人用了某张卡的不到一半，我们可以开 2 张卡，BS 开成 10；但是卡是空的，我们就单卡 bs = 1
per_device_train_batch_size = 1
# bs 为 2 的时候，单卡显存是 40G，然后如果能用一整张卡，就用 bs = 6 或者 4
# TODO 改动 max_training_epochs，这也是个可以搜的参数
max_training_epochs = 3
from main_without_validation import main

# TODO: change task name
for task_name in [ "task1562", ]: # "task1622","task281", "task1345",
    file_path = ROOT+"/main/NI_tasks/tasks.json"
    with open(file_path, "r", encoding="utf-8") as json_file:
        all_tasks = json.load(json_file)

    task_config_for_generation_tasks = None
    for task in all_tasks:
        if task["task_name"] == task_name:
            task_config_for_generation_tasks = (
                task["task_name"],
                task["task_instruction"],
                task["examples"],
                task["expected_content"],
                f"{TEST_DATA_ROOT}/prompt2model_test/testdataset/NI/eval/{task_name}",
                f"{TEST_DATA_ROOT}/prompt2model_test/testdataset/NI/test/{task_name}",
                task.get("optional_list", []),
                task.get("metric", "rouge"),
            )
            break


    # TODO 加expected content和metrics
    experiment_name = "NI_" + task_name + f"_exp_{experiment_rank}"
    # 训练时能够用的显卡，加起来总共剩余的显存对于 7B model 需要接近 200G
    # TODO 改显存配置

    file_path = ROOT+"/main/NI_tasks/tasks.json"

    log_and_data_root = Path(ROOT) / experiment_name
    evaluation_result_file_tail = "result.json"
    ckpt_root = Path(STORE_ROOT+"/ckpt_data_p2ms")
    best_ckpt_path = Path(STORE_ROOT+"/best_ckpt")
    best_validation_result_path = log_and_data_root / "best_validation_result.json"
    log_and_data_root.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    best_ckpt_path.mkdir(parents=True, exist_ok=True)

    def write_results(log_and_data_root, max_training_epochs):
        csv_header = [
            "task_name",
            "generation_temperature",
            "intput_length_constraint",
            "output_length_constraint",
            "generation_epochs",
        ] + ["epoch_" + str(i) for i in range(1, max_training_epochs + 1)]
        csv_data = []
        for experiment_folder in log_and_data_root.iterdir():
            if experiment_folder.is_dir():
                config_path = experiment_folder / "config.json"
                result_path = experiment_folder / evaluation_result_file_tail
                if config_path.exists() and result_path.exists():
                    config = read_json(config_path)
                    result = read_json(result_path)
                    row = {key: config.get(key, 0) for key in csv_header}
                    row.update(
                        {
                            "epoch_" + str(k): result.get(str(k), 0)
                            for k in range(1, max_training_epochs + 1)
                        }
                    )
                    csv_data.append(row)

        csv_file_path = log_and_data_root / "experiment_results.csv"
        with open(csv_file_path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=csv_header)
            writer.writeheader()
            writer.writerows(csv_data)

    def read_json(file_path):
        with open(file_path, "r") as file:
            return json.load(file)

    def print_and_execute_command(command):
        print(command)
        os.system(command)


    #! For generation tasks

    task_name, instruction, examples, expected_content, evaluation_dataset_path, test_set_path, optional_list, metric  = task_config_for_generation_tasks
    labels = []
    extraction_examples = []

    # TODO: change generation epoch
    def objective_function(
        generation_temperature,
        intput_length_constraint,
        output_length_constraint,
        generation_epoch,
    ):
        name = f"{task_name}_{generation_temperature}_{intput_length_constraint}_{output_length_constraint}_{generation_epoch}_{experiment_rank}"
        print(f"searching parameters: {name}")
        log_and_data_path = log_and_data_root / name
        log_and_data_path.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_root / name
        ckpt_path.mkdir(parents=True, exist_ok=True)

        assert optional_list != []
        assert expected_content != ""
        assert metric != ""
        params = {
            "CUDA_CONDITION": "0,1",
            "task_name": task_name,
            "instruction": instruction,
            "examples": examples,
            "expected_content": expected_content,
            "evaluation_dataset_path": evaluation_dataset_path,
            "test_set_path": test_set_path,
            "generation_epochs": generation_epoch,
            "generation_batch_size": int(10),
            "generation_top_k": int(40),
            "min_frequency": float(0.3),
            "generation_temperature": float(generation_temperature),
            "log_and_data_path": str(log_and_data_path),
            "ckpt_path": str(ckpt_path),
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "training_epochs": int(max_training_epochs),
            "tensor_parallel_size": TENSOR_SIZE,
            "evaluation_result_file_tail": evaluation_result_file_tail,
            "optional_list": optional_list,
            "metric": metric,
            "experiment_rank": experiment_rank,
            "per_device_train_batch_size": per_device_train_batch_size,
            "portion": 1,
            "intput_length_constraint": intput_length_constraint,
            "output_length_constraint": output_length_constraint,
            "conditional_labels": labels,
            "extraction_examples": extraction_examples,
        }
        with open(log_and_data_path / "config.json", "w") as f:
            json.dump(params, f, indent=4)
        main(str(log_and_data_path / "config.json"))
        return 0

    # TODO change params
    temperatures = [1.0]
    input_constraints = [False]
    output_constraints = [False]
    generation_epoches = [20]

    all_combinations = list(itertools.product(temperatures, input_constraints, output_constraints, generation_epoches))

    for combination in all_combinations:
        generation_temperature, input_length_constraint, output_length_constraint, generation_epoch = combination

        result = objective_function(
            generation_temperature,
            input_length_constraint,
            output_length_constraint,
            generation_epoch,
        )
    destroy_model_parallel()