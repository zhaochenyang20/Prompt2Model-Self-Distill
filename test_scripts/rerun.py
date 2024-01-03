import csv
import json
import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from pathlib import Path
import itertools

# TODO change card name
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

experiments = [("task1631", 6)]

# ("task034", 4), ("task034", 5), ("task034", 6)

for task_name, experiment_rank in experiments:
    file_path = "/home/cyzhao/main/NI_tasks/tasks.json"
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
                f"/home/cyzhao/prompt2model_test/testdataset/NI/eval/{task_name}",
                f"/home/cyzhao/prompt2model_test/testdataset/NI/test/{task_name}",
                task.get("optional_list", []),
                task.get("metric", "rouge"),
            )


    # TODO 加expected content和metrics
    experiment_name = "NI_" + task_name + f"_exp_{experiment_rank}"
    # 训练时能够用的显卡，加起来总共剩余的显存对于 7B model 需要接近 200G
    # TODO 改显存配置
    gpu_memory_utilization = 0.85
    # 如果别人用了某张卡的不到一半，我们可以开 2 张卡，BS 开成 10；但是卡是空的，我们就单卡 bs = 1
    per_device_train_batch_size = 1
    # bs 为 2 的时候，单卡显存是 40G，然后如果能用一整张卡，就用 bs = 6 或者 4
    max_training_epochs = 3
    file_path = "/home/cyzhao/main/NI_tasks/tasks.json"

    from main import main, validate_or_test

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
            "generation_temperature",
            "intput_length_constraint",
            "output_length_constraint",
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

    #! For classification tasks
    # task_name = task.task_name
    # instruction = task.task_instruction
    # examples = task.examples
    # expected_content = task.expected_content
    # evaluation_dataset_path = f"/home/cyzhao/prompt2model_test/testdataset/NI/eval/{task_name}"
    # test_set_path = f"/home/cyzhao/prompt2model_test/testdataset/NI/test/{task_name}"
    # optional_list = task.optional_list
    # metric = task.metric
    # labels = task.labels
    # extraction_examples = task.extraction_examples

    #! For generation tasks

    task_name, instruction, examples, expected_content, evaluation_dataset_path, test_set_path, optional_list, metric  = task_config_for_generation_tasks
    labels = []
    extraction_examples = []

    def objective_function(
        generation_temperature,
        intput_length_constraint,
        output_length_constraint,
    ):
        name = f"{task_name}_{generation_temperature}_{intput_length_constraint}_{output_length_constraint}_{experiment_rank}"
        print(f"searching parameters: {name}")
        log_and_data_path = log_and_data_root / name
        log_and_data_path.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_root / name
        ckpt_path.mkdir(parents=True, exist_ok=True)

        assert optional_list != []
        assert expected_content != ""
        assert metric != ""
    # change generation epoch to 40 for 738
        params = {
            "CUDA_CONDITION": os.environ["CUDA_VISIBLE_DEVICES"],
            "task_name": task_name,
            "instruction": instruction,
            "examples": examples,
            "expected_content": expected_content,
            "evaluation_dataset_path": evaluation_dataset_path,
            "test_set_path": test_set_path,
            "generation_epochs": int(40) if experiment_rank <=3 else int(20),
            "generation_batch_size": int(10),
            "generation_top_k": int(40),
            "min_frequency": float(0.3),
            "generation_temperature": float(generation_temperature),
            "log_and_data_path": str(log_and_data_path),
            "ckpt_path": str(ckpt_path),
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "training_epochs": int(max_training_epochs),
            "tensor_parallel_size": int(1),
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
            or len(list(evaluate_result.keys())) < max_training_epochs
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

    temperatures = [0.6, 0.7, 0.8, 0.9, 1.0]
    input_constraints = [False, True]
    output_constraints = [False, True]

    all_combinations = list(itertools.product(temperatures, input_constraints, output_constraints))

    # 遍历每组参数组合
    for combination in all_combinations:
        generation_temperature, input_length_constraint, output_length_constraint = combination
        
        # if task.is_classification is not None:
        #     output_length_constraint = False

        result = objective_function(
            generation_temperature,
            input_length_constraint,
            output_length_constraint,
        )

    with open(best_validation_result_path, "r") as json_file:
        evaluate_result = json.load(json_file)
    if "test_result" in evaluate_result:
        print("Already tested.")
    else:
        print("test best ckpt.")
        validate_or_test(
                test_set_path,
                best_ckpt_path / experiment_name,
                instruction,
                examples,
                gpu_memory_utilization,
                1,
                best_validation_result_path,
                test_content_store_path=log_and_data_root / "best_ckpt_generated_content",
                validation=False,
                metric=metric,
            )
    destroy_model_parallel()