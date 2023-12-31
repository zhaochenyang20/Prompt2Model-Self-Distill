import json

from datasets import Dataset


def get_highest_digit(number):
    if number == 0:
        return 0
    temp = number
    while temp >= 10:
        temp //= 10

    highest_digit = temp * 10 ** (len(str(number)) - 1)

    return highest_digit


# TODO change json file path
files = [
    "/home/cyzhao/main/NI_tasks/task_json/processed/task121_zest_text_modification.json"
]

for file_name in files:
    with open(file_name) as json_file:
        data = json.load(json_file)

    instances = data["Instances"]

    if len(instances) < 2000:
        data_size = get_highest_digit(len(instances))
        test_dataset = instances[: data_size // 2]
        eval_dataset = instances[data_size // 2 : data_size]
    else:
        test_dataset = instances[:1000]
        eval_dataset = instances[1000:2000]
    task_name = test_dataset[0]["id"].split("-")[0]

    print(
        f"{task_name}: evaluation data size = {len(eval_dataset)}, test data size = {len(test_dataset)}"
    )

    test_data_dict = {
        "input_col": [item["input"] for item in test_dataset],
        "output_col": [item["output"][0] for item in test_dataset],
    }

    eval_data_dict = {
        "input_col": [item["input"] for item in eval_dataset],
        "output_col": [item["output"][0] for item in eval_dataset],
    }

    test_dataset = Dataset.from_dict(test_data_dict)
    test_dataset.save_to_disk(
        f"/home/cyzhao/prompt2model_test/testdataset/NI/test/{task_name}"
    )
    loaded_dataset = Dataset.load_from_disk(
        f"/home/cyzhao/prompt2model_test/testdataset/NI/test/{task_name}"
    )

    eval_dataset = Dataset.from_dict(eval_data_dict)
    eval_dataset.save_to_disk(
        f"/home/cyzhao/prompt2model_test/testdataset/NI/eval/{task_name}"
    )
    loaded_dataset = Dataset.load_from_disk(
        f"/home/cyzhao/prompt2model_test/testdataset/NI/eval/{task_name}"
    )
