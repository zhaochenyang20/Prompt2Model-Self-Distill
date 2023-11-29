import csv
import json
import os
from pathlib import Path

import optuna
# TODO change card name
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
# TODO change task name
task_name = "task1622"
# TODO change experiment rank
experiment_rank = 1
experiment_name = "NI_" + task_name + f"_exp_{experiment_rank}"
# 训练时能够用的显卡，加起来总共剩余的显存对于 7B model 需要接近 200G
gpu_memory_utilization = 0.90
tensor_parallel_size = os.environ["CUDA_VISIBLE_DEVICES"].count(",") + 1
# 进行 inference（除了训练之外的任何步骤）时，会分布在每张卡上，也即 tensor_parallel_size 就是所有能用的 CUDA
# gpu_memory_utilization 是在每张卡上的占比，比如 CUDA_CONDITION = "0,1,4,5", gpu_memory_utilization = 0.9
# 则每张卡都会占去全部显存的 0.9，会占用四张卡，推理效率极其高
# gpu_memory_utilization 越小，则 inference 越慢
# 然而，不是每张卡都是空的，比如 0 卡已经有人跑了 40G 了，那么 gpu_memory_utilization < 0.5

max_training_epochs = 3
file_path = "/home/cyzhao/main/NI_tasks/tasks.json"
with open(file_path, "r", encoding="utf-8") as json_file:
    all_tasks = json.load(json_file)

tasks = []

# Discuss 加入了 metric 需要改写
for task in all_tasks:
    if task["task_name"] == task_name:
        task_tuple = (
            task["task_name"],
            task["task_instruction"],
            task["examples"],
            task["expected_content"],
            f"/home/cyzhao/prompt2model_test/testdataset/NI/eval/{task_name}",
            f"/home/cyzhao/prompt2model_test/testdataset/NI/test/{task_name}",
            task.get("optional_list", []),
            task.get("metric", "rouge"),
        )
        tasks.append(task_tuple)


log_and_data_root = Path("/home/cyzhao") / experiment_name
evaluation_result_file_tail = "result.json"
ckpt_root = Path("/data2/cyzhao/ckpt_data_p2ms")
best_ckpt_path = Path("/data2/cyzhao/best_ckpt")
best_validation_result_path = log_and_data_root / "best_validation_result.json"
log_and_data_root.mkdir(parents=True, exist_ok=True)
ckpt_root.mkdir(parents=True, exist_ok=True)
best_ckpt_path.mkdir(parents=True, exist_ok=True)


def write_results(log_and_data_root, max_training_epochs):
    csv_header = [
        "task_name",
        "generation_epochs",
        "generation_batch_size",
        "generation_top_k",
        "generation_temperature",
        "min_frequency",
        "min_input_length",
        "max_input_length",
        "min_output_length",
        "max_output_length",
        "training_epochs",
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


from main import main, validate_or_test


def read_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def print_and_execute_command(command):
    print(command)
    os.system(command)


for task in tasks:
    (
        task_name,
        instruction,
        examples,
        expected_content,
        evaluation_dataset_path,
        test_set_path,
        optional_list,
        metric,
    ) = task

    print(task_name)

    def objective_function(
        generation_epochs,
        generation_batch_size,
        generation_top_k,
        generation_temperature,
        min_frequency,
        training_epochs,
    ):
        name = f"{task_name}_{generation_epochs}_{generation_batch_size}_{generation_top_k}_{generation_temperature}_{min_frequency}_{training_epochs}_{experiment_rank}"
        print(f"searching parameters: {name}")
        log_and_data_path = log_and_data_root / name
        log_and_data_path.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_root / name
        ckpt_path.mkdir(parents=True, exist_ok=True)

        params = {
            "CUDA_CONDITION": os.environ["CUDA_VISIBLE_DEVICES"],
            "task_name": task_name,
            "instruction": instruction,
            "examples": examples,
            "expected_content": expected_content,
            "evaluation_dataset_path": evaluation_dataset_path,
            "test_set_path": test_set_path,
            "generation_epochs": int(generation_epochs),
            "generation_batch_size": int(generation_batch_size),
            "generation_top_k": int(generation_top_k),
            "generation_temperature": float(generation_temperature),
            "log_and_data_path": str(log_and_data_path),
            "ckpt_path": str(ckpt_path),
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "min_frequency": float(min_frequency),
            "training_epochs": int(training_epochs),
            "tensor_parallel_size": int(tensor_parallel_size),
            "evaluation_result_file_tail": evaluation_result_file_tail,
            "optional_list": optional_list,
            "metric": metric,
            "experiment_rank": experiment_rank,
            "portion": 1,
        }
        with open(log_and_data_path / "config.json", "w") as f:
            json.dump(params, f, indent=4)
        required_paths = [
            log_and_data_path / evaluation_result_file_tail,
            log_and_data_path / "inputs",
            log_and_data_path / "dataset",
        ]

        evaluate_result_path = log_and_data_path / evaluation_result_file_tail

        if evaluate_result_path.exists():
            evaluate_result = read_json(evaluate_result_path)
        else:
            evaluate_result = {}
            with open(evaluate_result_path, "w") as f:
                json.dump(evaluate_result, f, indent=4)

        best_validation_result = 0
        validation_results = {}
        if best_validation_result_path.exists():
            validation_results = read_json(best_validation_result_path)
            best_validation_result = validation_results.get("validation_result", 0)
        else:
            best_validation_result = 0
            with open(best_validation_result_path, "w") as f:
                json.dump({}, f, indent=4)

        if (
            not all(path.exists() for path in required_paths)
            or len(list(evaluate_result.keys())) < training_epochs
        ):
            print(log_and_data_path)
            ckpt_paths_and_result = main(str(log_and_data_path / "config.json"))

            if ckpt_paths_and_result is None:
                return 0

            highest_result_path = max(
                ckpt_paths_and_result, key=ckpt_paths_and_result.get
            )
            highest_validation_result = ckpt_paths_and_result[highest_result_path]

            if highest_validation_result > best_validation_result:
                # Update the best validation result and write to file
                validation_results = {
                    "task_name": task_name,
                    "validation_result": highest_validation_result,
                    "evaluate_result_path": str(evaluate_result_path),
                    "ckpt_path": str(highest_result_path),
                }
                with open(best_validation_result_path, "w") as f:
                    json.dump(validation_results, f, indent=4)

                # Move the best checkpoint and delete others
                task_best_ckpt_path = Path(best_ckpt_path) / experiment_name
                if task_best_ckpt_path.exists():
                    print_and_execute_command(f"rm -rf {task_best_ckpt_path}")
                print_and_execute_command(
                    f"mv {highest_result_path} {task_best_ckpt_path}"
                )

                for ckpt_path in ckpt_paths_and_result:
                    if ckpt_path != highest_result_path:
                        print_and_execute_command(f"rm -rf {ckpt_path}")
            else:
                # If no new best result, delete all checkpoints
                for ckpt_path in ckpt_paths_and_result:
                    print_and_execute_command(f"rm -rf {ckpt_path}")
        else:
            highest_validation_result = max(evaluate_result.values())

        write_results(log_and_data_root, max_training_epochs)
        return highest_validation_result

    def objective(trial):
        generation_epochs = trial.suggest_categorical("generation_epochs", [10, 20, 30])
        generation_batch_size = trial.suggest_categorical(
            "generation_batch_size", [10, 15, 20]
        )
        generation_top_k = trial.suggest_categorical("generation_top_k", [40, 45, 50])
        generation_temperature = trial.suggest_categorical(
            "generation_temperature", [0.4, 0.5, 0.6, 0.7, 0.8]
        )
        min_frequency = trial.suggest_categorical("min_frequency", [0.3, 0.35, 0.4])
        training_epochs = trial.suggest_int("training_epochs", 3, max_training_epochs)

        return objective_function(
            generation_epochs,
            generation_batch_size,
            generation_top_k,
            generation_temperature,
            min_frequency,
            training_epochs,
        )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    best_params = study.best_params
    print(best_params)

    with open(best_validation_result_path, "r") as json_file:
        evaluate_result = json.load(json_file)
    if "test_result" in evaluate_result:
        print("Already tested")
        continue
    else:
        print("test best ckpt.")
        validate_or_test(
            test_set_path,
            best_ckpt_path / experiment_name,
            instruction,
            examples,
            gpu_memory_utilization,
            tensor_parallel_size,
            best_validation_result_path,
            test_content_store_path=log_and_data_root / "best_ckpt_generated_content",
            validation=False,
            metric=metric,
        )
